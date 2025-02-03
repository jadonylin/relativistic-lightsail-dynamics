# IMPORTS ################################################################################################################################################
import os
## Limit number of numpy threads (MUST GO BEFORE NUMPY IMPORT) ##
os.environ["OMP_NUM_THREADS"] = "1" 
os.environ["OPENBLAS_NUM_THREADS"] = "1" 
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 

from copy import deepcopy
from datetime import datetime
from multiprocessing import Pool

import numpy as np
from numpy import *

import pickle
import sys
sys.path.append("../")

from opt import FOM_uniform, global_optimise
from parameters import D1_ND, Parameters, Initial_bigrating
from twobox import TwoBox

Email_result = True

# GLOBAL OPTIMISATION ###########################################################################
## FIXED PARAMETERS ##
wavelength = 1. # Laser wavelength
angle = 0.
Nx = 100 # Number of grid points
nG = 25 # 25 # Number of Fourier components

# relaxation parameter, should be infinite unless you need to avoid singular matrix at grating cutoffs
# Also, optimiser finds large magnitude, noisy rNeg1 when Qabs = np.inf 
Qabs = 1e7  # 1e7


I0,L,m,c=Parameters()
grating_pitch, grating_depth, box1_width, box2_width, box_centre_dist, box1_eps, box2_eps, gaussian_width, substrate_depth, substrate_eps=Initial_bigrating()

# Initial twobox grating
grating = TwoBox(grating_pitch, grating_depth, box1_width, box2_width, box_centre_dist, box1_eps, box2_eps, 
                 gaussian_width, substrate_depth, substrate_eps,
                 wavelength, angle, Nx, nG, Qabs)


## FOM PARAMETERS ##
goal = 0.1 # Stopping criteria for adaptive sampling in the FOM (float is loss_goal, int is npoints_goal)
final_speed = 5 # percentage of c  # 20
return_grad = True # Return FOM and gradient of FOM



## OPTIMISATION ##
# Set up NLOPT
seed = 20240902 # LDS seed
sampling = 'sobol' # 'sobol' or 'random'
n_sample_exp = 3
n_sample = 2**n_sample_exp
ndof = 11

# Parameter bounds
pitch_min = np.round(1.001*wavelength/D1_ND([final_speed/100,0.]),3) # stay away from the cutoff divergences
pitch_max = 1.999 

h1_min = 0.01 # h1 = grating depth
h1_max = 1.5 * pitch_max

box_width_min = 0.
box_width_max = 1.*pitch_max # single box width must be smaller than pitch

box_centre_dist_min = 0.
box_centre_dist_max = 0.5*pitch_max # redundant space if > 0.5*pitch

box_eps_min = 1.5**2 # Minimum allowed grating permittivity
box_eps_max = 3.5**2 # Maximum allowed grating permittivity

gaussian_width_min=0.5*L 
gaussian_width_max=5*L

substrate_depth_min = h1_min 
substrate_depth_max = h1_max

substrate_eps_min = box_eps_min 
substrate_eps_max = box_eps_max

# Stopping criteria
xtol_rel = 1e-4 # local
ftol_rel = 1e-8 # local
num_cores = 32 # number of cores to run parallel optimisation
maxfev = 5000 # global 1000



# OBJECTIVE FUNCTION ###########################################################################
param_bounds = [(pitch_min, pitch_max), (h1_min, h1_max), 
                (box_width_min, box_width_max), (box_width_min, box_width_max),
                (box_centre_dist_min, box_centre_dist_max),
                (box_eps_min, box_eps_max), (box_eps_min, box_eps_max),
                (gaussian_width_min, gaussian_width_max),
                (substrate_depth_min, substrate_depth_max), 
                (substrate_eps_min, substrate_eps_max)]

def objective(params):
    grating_pitch, grating_depth, box1_width, box2_width, box_centre_dist, box1_eps, box2_eps, gaussian_width, substrate_depth, substrate_eps = params
    grating.grating_pitch = grating_pitch
    grating.grating_depth = grating_depth
    grating.box1_width = box1_width
    grating.box2_width = box2_width
    grating.box_centre_dist = box_centre_dist
    grating.box1_eps = box1_eps
    grating.box2_eps = box2_eps

    grating.gaussian_width=gaussian_width
    grating.substrate_depth = substrate_depth
    grating.substrate_eps = substrate_eps

    return FOM_uniform(grating, final_speed, goal, return_grad)



# RECORDING RESULTS ###########################################################################
## Converting non-h1 parameter dicts to strings ##
# Fixed parameters
fixed_params_dict = {'wavelength': wavelength,
                     'angle': angle,
                     'Nx': Nx, 'nG': nG, 'Qabs': Qabs}
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
GO_dict = {'number of cores': num_cores, 'maxfev per core': maxfev}
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
txt_fname = f'./Data/FOM_4th_bounds2_optimisation_maxfev{maxfev*num_cores}.txt'
with open(txt_fname, "a") as result_file:
    result_file.writelines(lines_to_file)



### RUN GLOBAL OPTIMISATION ###########################################################################
def optimise_partitioned_depth(partition_h1_min, partition_h1_max):
    _param_bounds = param_bounds[:]
    _param_bounds[1] = (partition_h1_min, partition_h1_max)
    return global_optimise(objective, sampling, seed, n_sample, maxfev, xtol_rel, ftol_rel, _param_bounds)

h1_bounds = []
h1s = np.linspace(h1_min,h1_max,num_cores+1)
for p in range(0,num_cores):
    interval = (h1s[p], h1s[p+1])
    h1_bounds.append(interval)


# Run parallel optimisation
pkl_fname = f'./Data/FOM_4th_bounds2_optimisation_maxfev{maxfev*num_cores}.pkl'
if __name__ == '__main__':
    with Pool(processes=num_cores) as pool:
        # Time checking
        import time 
        T1=time.time()
        print("Begun!")
        
        all_optima = pool.starmap(optimise_partitioned_depth, h1_bounds)
        opt_FOMs = []
        opt_params = []
        opt_gratings = []
        is_opt = []
        for optimum in all_optima:
            opt_FOMs.append(optimum[0])
            
            opt_param = optimum[1]
            opt_params.append(opt_param)
            
            grating_copy = deepcopy(grating)
            grating_copy.params = opt_param
            opt_gratings.append(grating_copy) 

            is_opt.append(optimum[2])

        data = {'Optimised grating': opt_gratings,  'FOM': opt_FOMs,        'Real optimum?': is_opt,
                'Optimised parameters': opt_params,
                'FOM parameters': FOM_params_dict,  'Bounds': bounds_dict,
                'Sampling settings': sampling_dict, 'LO settings': LO_dict, 'GO settings': GO_dict}
        
        # Save result
        try:
            with open(pkl_fname, 'wb') as data_file:
                pickle.dump(data, data_file)
        except:
            print("Couldn't save data")
            print(data)

        if Email_result:

            ## Send email to notify end of code

            # Import the following module 
            from email.mime.text import MIMEText 
            from email.mime.multipart import MIMEMultipart 
            import smtplib 


            smtp = smtplib.SMTP('smtp.gmail.com', 587) 
            smtp.ehlo() 
            smtp.starttls() 

            # Login with your email and password 
            from login import email_address, password
            smtp.login(email_address, password)

            def message(subject="Python Notification", 
                        text="", img=None, 
                        attachment=None): 
                
                # build message contents 
                msg = MIMEMultipart() 
                
                # Add Subject 
                msg['Subject'] = subject 
                
                # Add text contents 
                msg.attach(MIMEText(text)) 

                
                return msg 

            T2=str(round( (time.time()-T1)/ (60**2) ) )

            # Call the message function 
            msg = message(subject="Your code has finished in " + T2 + " hours!", text="Code finished! Yay!",img=None, 
                        attachment=None) 

            # Make a list of emails, where you wanna send mail 
            to = ["lvan0119@uni.sydney.edu.au", "mail.liam.vr@gmail.com"] 

            # Provide some data to the sendmail function! 
            smtp.sendmail(from_addr="midgetliamllama@gmail.com", 
                        to_addrs=to, msg=msg.as_string()) 

            # Finally, don't forget to close the connection 
            smtp.quit()

