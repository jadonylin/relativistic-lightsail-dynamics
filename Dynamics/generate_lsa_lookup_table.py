"""
A script to generate a lookup table of the efficiency factors and their derivatives with respect to lambda and delta
for a given grating, across a range of lambda values. Used to precompute efficiencies in dynamics simulation.

The coefficients are calculated in the linear stability regime, where the incident angle delta is fixed to zero.
"""


import sys
sys.path.append('../')

import numpy as np
import dill as pickle

import time

import Optimisation.opt as opt
from parameters import D1_ND
from twobox import TwoBox

t_start = time.time()

## Initialise grating
runID = "Fasymp20_fixgaussian20_50GW"
final_speed = 20.
num_cores = 200
maxtime = 1440
output_opt_idx = 0
opt_grating_basefname = f"../Optimisation/Data/{runID}_FOM_optimisation_maxtime{maxtime}"
_, _, _opt_grating = opt.extract_opt(opt_grating_basefname, num_processes=num_cores, output_opt_idx=output_opt_idx)
op = _opt_grating.all_params[:]

# Set custom parameters, if needed. If not needed, can just set "grating" to the extracted grating above.
wavelength       = 1.
angle            = 0.
Nx               = 100
numG             = 12
Qabs             = np.inf

grating = TwoBox(*op, wavelength=wavelength, angle=angle, Nx=Nx, nG=numG, Qabs=Qabs, 
                 RCWA_engine="TORCWA", torcwa_edge_sharpness=45)

## Number of lambda' points
klambda = 1000
v_final = final_speed/100 
lambda_final = 1/D1_ND(v_final)
lambda_array = np.linspace( wavelength, lambda_final, klambda )

## Storage arrays
Q1_array            = np.zeros( klambda )
Q2_array            = np.zeros( klambda )
PD_Q1_delta_array   = np.zeros( klambda )
PD_Q2_delta_array   = np.zeros( klambda )
PD_Q1_lambda_array  = np.zeros( klambda )
PD_Q2_lambda_array  = np.zeros( klambda )

for i in range(klambda):
    grating.wavelength   = lambda_array[i]
    # Call function
    Qs = grating.to_numpy(grating.return_Qs_auto())
    # Efficiency factors
    Q1_array[i] = Qs[0];             Q2_array[i] = Qs[1]
    # Derivatives
    PD_Q1_delta_array[i] = Qs[2];   PD_Q2_delta_array[i] = Qs[3]
    PD_Q1_lambda_array[i] = Qs[4];  PD_Q2_lambda_array[i] = Qs[5]

t_end = time.time()-t_start
t_end_sec = round(t_end)
t_end_min = round(t_end/60)
print(rf"Finished in {t_end_sec} seconds, or {t_end_min} minutes!")
print(rf"#lambda: {klambda}")

## Save data
pkl_fname = rf'./Data/{runID}_lsa_Lookup_table_lambda_{klambda}.pkl'
data = {'Q1': Q1_array, 'Q2': Q2_array, 'PD_Q1_delta': PD_Q1_delta_array, 'PD_Q2_delta': PD_Q2_delta_array, 'PD_Q1_lambda': PD_Q1_lambda_array, 'PD_Q2_lambda': PD_Q2_lambda_array, 
         'lambda array': lambda_array}
with open(pkl_fname, 'wb') as data_file:
    pickle.dump(data, data_file)