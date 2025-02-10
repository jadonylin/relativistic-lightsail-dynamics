import sys
sys.path.append('../')

import numpy as np
import pickle
from twobox import TwoBox
from parameters import D1_ND
import time


t_start = time.time()

## Ilic grating
# wavelength      = 1.5 / D1_ND(5.3/100)
# grating_pitch   = 1.8 / wavelength
# grating_depth   = 0.5 / wavelength
# box1_width      = 0.15 * grating_pitch
# box2_width      = 0.35 * grating_pitch
# box_centre_dist = 0.60 * grating_pitch
# box1_eps        = 3.5**2 
# box2_eps        = 3.5**2
# gaussian_width  = 2 * 10
# substrate_depth = 0.5 / wavelength
# substrate_eps   = 1.45**2

## Second
start_wavelength= 1
grating_pitch   = 1.5384469388251338   
grating_depth   = 0.5580762361523982    
box1_width      = 0.10227122552871484   
box2_width      = 0.07605954942866577   
box_centre_dist = 0.2669020979549422    
box1_eps        = 9.614975107945112
box2_eps        = 9.382304398409568
gaussian_width  = 33.916288616522735
substrate_depth = 0.17299998450776535   
substrate_eps   = 9.423032644325023

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
pkl_fname = rf'Data/Stability_Lookup_table_lambda_{klambda}.pkl'
data = {'Q1': Q1_array, 'Q2': Q2_array, 'PD_Q1_delta': PD_Q1_delta_array, 'PD_Q2_delta': PD_Q2_delta_array, 'PD_Q1_lambda': PD_Q1_lambda_array, 'PD_Q2_lambda': PD_Q2_lambda_array, 
         'lambda array': lambda_array}
with open(pkl_fname, 'wb') as data_file:
    pickle.dump(data, data_file)
