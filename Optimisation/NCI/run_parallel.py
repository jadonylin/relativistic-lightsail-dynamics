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
from multiprocess import Pool

import numpy as np
from numpy import *

import pathlib
import dill as pickle

import sys
sys.path.append("../")

import opt 
from parameters import Initial_bigrating, opt_Parameters, Bounds


# Global optimisation parameters
num_cores = 90  # number of cores to run parallel optimisation
maxtime = 47*60  # Stop after maxtime minutes
maxstop = {'maxtime': maxtime}  # global 1000
h1_min, h1_max, param_bounds = Bounds()
runID = "MdSnpmin1_torcwa"

# Local optimisation parameters
xtol_rel = 1e-4  
ftol_rel = 1e-8  

seed = 20250515  # LDS seed
sampling = 'sobol'  # 'sobol' or 'random'
n_sample_exp = 4
n_sample = 2**n_sample_exp  # number of random samples per iteration, the best of which (in non-overlapping regions of attraction) are locally optimised
ndof = 10  # number of optimisation parameters


# Initial grating parameters and hyperparameters
wavelength, angle, Nx, nG, Qabs, goal, final_speed, return_grad, RCWA_engine, torcwa_sharpness = opt_Parameters()
grating_pitch, grating_depth, box1_width, box2_width, box_centre_dist, box1_eps, box2_eps, gaussian_width, substrate_depth, substrate_eps = Initial_bigrating()



# RECORDING RESULTS ###########################################################################
# Parameters and hyperparameters for an optimisation run are recorded in dictionaries and saved to a text file for 
# convenient viewing. 

## Converting non-h1 parameter dicts to strings ##
# Fixed parameters
fixed_params_dict = {'wavelength': wavelength, 'angle': angle, 'Nx': Nx, 'nG': nG, 'Qabs': Qabs,
                     'RCWA engine': RCWA_engine, 'TORCWA edge sharpness': torcwa_sharpness}
fixed_params_line = str(fixed_params_dict)
FOM_params_dict = {'final_speed': final_speed, 'goal': goal}
FOM_params_line = str(FOM_params_dict)

# Bounded parameters
bounds_dict = {'param_bounds': param_bounds}
bounds_line = str(bounds_dict)

# Optimiser options
sampling_dict = {'Sampling method': sampling, 'n_sample': f'2E+{n_sample_exp}', 'seed': seed}
sampling_line = str(sampling_dict)
LO_dict = {'xtol_rel': f"{xtol_rel:.1E}", 'ftol_rel': f"{ftol_rel:.1E}"}
LO_line = str(LO_dict)
GO_dict = {'number of cores': num_cores, 'maxstop per core': maxstop}
GO_line = str(GO_dict)

# Date and time at beginning of run
time_at_execution = str(datetime.now())

# Strings to write to file
lines_to_file = ["\n\n------------------------------------------------------------------------------------------------------------------------------------\n"
                , f"Date & time      | {time_at_execution}\n"
                ,  "\n"
                , f"Fixed parameters | {fixed_params_line}\n"
                , f"FOM parameters   | {FOM_params_line}\n"
                , f"Non-h1 bounds    | {bounds_line}\n"
                ,  "\n"
                , f"Sampling options | {sampling_line}\n"
                , f"LO options       | {LO_line}\n"
                , f"GO options       | {GO_line}\n"
                , "------------------------------------------------------------------------------------------------------------------------------------\n"]


## Writing to file ##
current_dir = pathlib.Path(__file__).resolve(strict=True).parent
txt_fname = f'{runID}_FOM_optimisation_maxtime{maxtime}.txt'
txt_dir = current_dir / "Data" / txt_fname
with open(txt_dir, "a") as result_file:
    result_file.writelines(lines_to_file)



### RUN GLOBAL OPTIMISATION ###########################################################################
# The parallel optimisation is run by partitioning the h1 parameter range into a number (num_cores) of non-intersecting 
# subsidiary parameter ranges whose union is the full h1 parameter range set by the user. Each core optimises over one of 
# those subsidiary h1 ranges, and saves the found most-optimal grating and all optimisation parameters into a dictionary 
# that is stored in .pkl file. The optimisation results for all cores are stored in the same .pkl file.

def optimise_partitioned_depth(h1_bounds):
    _param_bounds = param_bounds[:]
    _param_bounds[1] = tuple([*h1_bounds])  # Must unpack a single argument for pool.imap to be applied correctly
    return opt.global_optimise(Initial_bigrating(), opt_Parameters(), sampling, seed, n_sample, maxstop, xtol_rel, ftol_rel, _param_bounds)

h1_bounds = []
h1s = np.linspace(h1_min,h1_max,num_cores+1)
for p in range(0,num_cores):
    interval = (h1s[p], h1s[p+1])
    h1_bounds.append(interval)

# Run parallel optimisation
if __name__ == '__main__':
    with Pool(processes=num_cores) as pool:        
        print("Begun!")
        
        # Some processes run for too long, so we need to store the results of each process 
        # immediately once they become available.
        # From https://stackoverflow.com/questions/70317903/how-to-store-all-the-output-before-multiprocessing-finish
        for opt_index, opt_result in enumerate(pool.imap_unordered(optimise_partitioned_depth, h1_bounds)):
            
            opt_FOM = opt_result[0]
            opt_grating = opt_result[1]
            opt_params = opt_result[2]
            is_opt = opt_result[3]
            num_fev = opt_result[4]

            time_at_completion = str(datetime.now())

            data = {'Optimised grating': opt_grating, 'FOM': opt_FOM, 'Real optimum?': is_opt,
                    'Optimised parameters': opt_params, 'Function evaluations': num_fev,
                    'FOM parameters': FOM_params_dict,  'Bounds': bounds_dict,
                    'Sampling settings': sampling_dict, 'LO settings': LO_dict, 'GO settings': GO_dict,
                    'Execution time': time_at_execution, 'Completion time': time_at_completion}
            
            pkl_fname = f'{runID}_FOM_optimisation_maxtime{maxtime}_process{opt_index}.pkl'
            pkl_dir = current_dir / "Data" / pkl_fname
            with open(pkl_dir, 'wb') as data_file:
                pickle.dump(data, data_file)
