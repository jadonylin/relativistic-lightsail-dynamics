"""
Main script for running twobox optimisation on multiple computer cores.

How to run:
    Set the parameters for optimisation in the parameters.py module 
    
    Set the hyperparameters for the global and local optimisation in this module
    
    Set the number of computer cores to use in parallel during optimisation

    Set the maximum number of function evaluations (for optimisation) per core
"""

# IMPORTS ################################################################################################################################################
import os
## Limit number of numpy threads (MUST GO BEFORE NUMPY IMPORT) ##
os.environ["OMP_NUM_THREADS"] = "1" 
os.environ["OPENBLAS_NUM_THREADS"] = "1" 
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 

from datetime import datetime
from multiprocessing import Pool

import numpy as np
from numpy import *

import pickle
import sys
sys.path.append("../")

import fom
import opt 
from parameters import Hyperparameters, Bounds


# Global optimisation parameters
choose_monofom = fom.monofom_asymp
num_cores = 5  # number of cores to run parallel optimisation
maxfev = 200  # global 1000
h1_min, h1_max, param_bounds = Bounds()
runID = "nsample_convergence_test"

# Local optimisation parameters
xtol_rel = 1e-4
ftol_rel = 1e-8

seed = 20250508  # LDS seed
sampling = 'sobol'  # 'sobol' or 'random'
n_samples = np.linspace(4,64,num_cores,endpoint=True, dtype=int)  # number of random samples per iteration, the best of which (in non-overlapping regions of attraction) are locally optimised
ndof = 10  # number of optimisation parameters


# Initial grating parameters and hyperparameters
wavelength, angle, Nx, nG, Qabs, goal, final_speed, return_grad, RCWA_engine, torcwa_sharpness, fixed_parameters = Hyperparameters()

# Objective function
def objective(grating,params):
    grating.params = params
    return fom.multifom_uniform(grating, choose_monofom, final_speed, goal, return_grad)



# RECORDING RESULTS ###########################################################################
# Parameters and hyperparameters for an optimisation run are recorded in dictionaries and saved to a text file for 
# convenient viewing. 

## Converting non-h1 parameter dicts to strings ##
# Fixed parameters
hyperparams_dict = {'wavelength': wavelength, 'angle': angle, 'Nx': Nx, 'nG': nG, 'Qabs': Qabs,
                     'RCWA engine': RCWA_engine, 'TORCWA edge sharpness': torcwa_sharpness,
                     'Fixed parameters': fixed_parameters}
hyperparams_line = str(hyperparams_dict)
FOM_params_dict = {'final_speed': final_speed, 'goal': goal}
FOM_params_line = str(FOM_params_dict)

# Bounded parameters
bounds_dict = {'param_bounds': param_bounds}
bounds_line = str(bounds_dict)

# Optimiser options
sampling_dict = {'Sampling method': sampling, 'n_sample': None, 'seed': seed}
sampling_line = str(sampling_dict)
LO_dict = {'xtol_rel': f"{xtol_rel:.1E}", 'ftol_rel': f"{ftol_rel:.1E}"}
LO_line = str(LO_dict)
GO_dict = {'number of cores': num_cores, 'maxfev per core': maxfev}
GO_line = str(GO_dict)

# Date and time at beginning of run
time_at_execution = str(datetime.now())

# Strings to write to file
lines_to_file = ["\n\n------------------------------------------------------------------------------------------------------------------------------------\n"
                , f"Date & time      | {time_at_execution}\n"
                ,  "\n"
                , f"Hyperparameters  | {hyperparams_line}\n"
                , f"FOM parameters   | {FOM_params_line}\n"
                , f"Bounds           | {bounds_line}\n"
                ,  "\n"
                , f"Sampling options | {sampling_line}\n"
                , f"LO options       | {LO_line}\n"
                , f"GO options       | {GO_line}\n"
                , "------------------------------------------------------------------------------------------------------------------------------------\n"]


## Writing to file ##
txt_fname = f'./Data/{runID}_maxfev{maxfev*num_cores}.txt'
with open(txt_fname, "a") as result_file:
    result_file.writelines(lines_to_file)



### RUN GLOBAL OPTIMISATION ###########################################################################
def optimise_nsample(nsample):
    return opt.global_optimise(choose_monofom, Hyperparameters(), objective, sampling, seed, nsample, maxfev, xtol_rel, ftol_rel, param_bounds, True)

# Run parallel optimisation
if __name__ == '__main__':
    with Pool(processes=num_cores) as pool:        
        print("Begun!")
        
        # Some processes run for too long, so we need to store the results of each process 
        # immediately once they become available.
        # From https://stackoverflow.com/questions/70317903/how-to-store-all-the-output-before-multiprocessing-finish
        for opt_index, opt_result in enumerate(pool.imap_unordered(optimise_nsample, n_samples)):
            
            opt_FOM = opt_result[0]
            opt_grating = opt_result[1]
            opt_params = opt_result[2]
            is_opt = opt_result[3]
            settings = opt_result[4]

            nsample = settings['n_sample']
            sampling_dict['n_sample'] = nsample
            time_at_completion = str(datetime.now())

            data = {'Optimised grating': opt_grating, 'FOM': opt_FOM, 'Real optimum?': is_opt,
                    'Optimised parameters': opt_params,
                    'FOM parameters': FOM_params_dict,  'Bounds': bounds_dict,
                    'Sampling settings': sampling_dict, 'LO settings': LO_dict, 'GO settings': GO_dict,
                    'Execution time': time_at_execution, 'Completion time': time_at_completion}
            
            pkl_fname = f'./Data/{runID}_maxfev{maxfev*num_cores}_process{opt_index}.pkl'
            with open(pkl_fname, 'wb') as data_file:
                pickle.dump(data, data_file)