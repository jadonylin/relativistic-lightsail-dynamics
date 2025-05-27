"""
A script to generate a lookup table for the efficiencies of a user-input twobox grating. Generates 
the efficiencies for a range of wavelengths and angles, and saves the data as a pickle file. The 
calculation is parallelised using the multiprocess module, so set the number of processes before
running.

IMPORTNT: set the maximum angle for the efficiency data close to the ±1 order grating cutoffs 
at the maximum Doppler-shifted wavelength. This ensures the dynamics results are physical and 
that the dynamics simulation terminates if the cutoff angle is exceeded.
"""

import itertools

import multiprocess as mp

import numpy as np
import dill as pickle

import sys
sys.path.append('../')

import time

from parameters import D1_ND
from twobox import TwoBox
import Optimisation.opt as opt

t_start = time.time()

## Initialise grating
runID = "MdSnpmin20_mirror"
num_cores = 90
# maxfev = 500
maxtime = 1410.0
# opt_grating_basefname = f"../Optimisation/Data/{runID}_FOM_optimisation_maxfev{num_cores*maxfev}"
opt_grating_basefname = f"../Optimisation/Data/{runID}_FOM_optimisation_maxtime{maxtime}"
_, _, _opt_grating = opt.extract_opt(opt_grating_basefname, num_processes=num_cores, output_opt_idx=0)
try:
    op = _opt_grating.all_params[:]
except AttributeError:
    op = _opt_grating.params

# Set custom parameters, if needed. If not needed, can just set "grating" to the extracted grating above.
wavelength       = 1.
angle            = 0.
Nx               = 100
numG             = 12
Qabs             = np.inf

# Fdmp_params = [1.226     , 2.90479687, 0.31178695, 0.0071848 , 0.9995    , 2.96553672, 4.30675011]
# grating = TwoBox(*Fdmp_params, gaussian_width=50., substrate_depth=1., substrate_eps=-1e6,
#                       wavelength=1., angle=0., Nx=100, nG=12, Qabs=np.inf,
#                       RCWA_engine="TORCWA", torcwa_edge_sharpness=45)

print(op)
grating = TwoBox(*op, wavelength=wavelength, angle=angle, Nx=Nx, nG=numG, Qabs=Qabs, 
                 RCWA_engine="TORCWA", torcwa_edge_sharpness=45, fixed_parameters=)


klambda = 1000  # Number of lambda' points
v_final = 20/100
lambda_final = 1/D1_ND(v_final)
lambda_array = np.linspace(wavelength, lambda_final, klambda)

kdelta = 1000  # Number of delta' points
delta_max = 5.*np.pi/180  # IMPORTANT: must be set according to the grating cutoffs
delta_min = -delta_max
delta_array = np.linspace(delta_min, delta_max, kdelta)

num_processes = 10

def eff_auto(*args):
    grating.wavelength = args[0][0]
    grating.angle = args[0][1]
    return grating.to_numpy(grating.return_Qs_auto())

param_inputs = ((l,d) for l,d in itertools.product(lambda_array,delta_array))

p = mp.Pool(num_processes)
efficiencies_ = p.map(eff_auto, param_inputs)
efficiencies = np.reshape(np.array(efficiencies_), (klambda, kdelta, 6))  # 6 ordered efficiencies: Q1, Q2, PD_Q1_delta, PD_Q2_delta, PD_Q1_lambda, PD_Q2_lambda
p.close()
p.join()

Q1_array           = efficiencies[:,:,0];   Q2_array           = efficiencies[:,:,1]
PD_Q1_delta_array  = efficiencies[:,:,2];   PD_Q2_delta_array  = efficiencies[:,:,3]
PD_Q1_lambda_array = efficiencies[:,:,4];   PD_Q2_lambda_array = efficiencies[:,:,5]

t_end = time.time() - t_start
t_end_sec = round(t_end)
t_end_min = round(t_end/60)
t_end_hours = round(t_end/60**2)
print(rf"Finished in {t_end_sec} seconds, or {t_end_min} minutes, or {t_end_hours} hours!")
print(rf"#lambda: {klambda}, #delta: {kdelta}")

pkl_fname = rf"./Data/{runID}_Lookup_table_lambda_{klambda}_by_delta_{kdelta}.pkl"
data = {'Q1': Q1_array, 'Q2': Q2_array, 'PD_Q1_delta': PD_Q1_delta_array, 'PD_Q2_delta': PD_Q2_delta_array, 
        'PD_Q1_lambda': PD_Q1_lambda_array, 'PD_Q2_lambda': PD_Q2_lambda_array, 
        'lambda array': lambda_array, 'delta array': delta_array}
with open(pkl_fname, 'wb') as data_file:
    pickle.dump(data, data_file)