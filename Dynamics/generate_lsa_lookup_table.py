import sys
sys.path.append('../')

import numpy as np
import pickle
from twobox import TwoBox
from parameters import D1_ND
import time


t_start = time.time()

## Initialise grating
# TODO: extract gratings directly from optimised pkl, and print parameters
grating_pitch   = 1.4185910181100811
grating_depth   = 0.49190319526197407
box1_width      = 0.5719025530222406
box2_width      = 0.053439272331534775
box_centre_dist = 0.44458828885168056
box1_eps        = 11.029595778447616
box2_eps        = 6.100136959625866
gaussian_width  = 31.37144885298504
substrate_depth = 0.43257336002828756
substrate_eps   = 2.279546035418172

wavelength      = 1.
angle           = 0.
Nx              = 100
numG            = 25
Qabs            = np.inf

grating = TwoBox(grating_pitch, grating_depth, box1_width, box2_width, box_centre_dist, box1_eps, box2_eps, 
                 gaussian_width, substrate_depth, substrate_eps,
                 wavelength, angle, Nx, numG, Qabs)

## Number of lambda' points
klambda = 1000
v_final = 5/100 # 5/100
lambda_final = 1/D1_ND(v_final)
lambda_array = np.linspace( wavelength, lambda_final, klambda )

runID = "test"

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
    Qs = grating.return_Qs_auto(True)
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
pkl_fname = rf'Data/{runID}_lsa_Lookup_table_lambda_{klambda}.pkl'
data = {'Q1': Q1_array, 'Q2': Q2_array, 'PD_Q1_delta': PD_Q1_delta_array, 'PD_Q2_delta': PD_Q2_delta_array, 'PD_Q1_lambda': PD_Q1_lambda_array, 'PD_Q2_lambda': PD_Q2_lambda_array, 
         'lambda array': lambda_array}
with open(pkl_fname, 'wb') as data_file:
    pickle.dump(data, data_file)
