import numpy as np
import pickle
import time

import sys
sys.path.append('../')

from parameters import D1_ND
from twobox import TwoBox


t_start = time.time()

## Initialise grating
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


klambda = 100  # Number of lambda' points
v_final = 5/100
lambda_final = 1/D1_ND(v_final)
lambda_array = np.linspace(wavelength, lambda_final, klambda)

kdelta = 100  # Number of delta' points
delta_max = 15*np.pi/180
delta_min = -delta_max
delta_array = np.linspace(delta_min, delta_max, kdelta)

efficiencies = np.zeros((klambda, kdelta, 6))  # 6 ordered efficiencies: Q1, Q2, PD_Q1_delta, PD_Q2_delta, PD_Q1_lambda, PD_Q2_lambda

for i in range(klambda):
    grating.wavelength = lambda_array[i]
    for j in range(kdelta):
        grating.angle = delta_array[j]
        efficiencies[i,j,:] = grating.return_Qs_auto()

Q1_array           = efficiencies[:,:,0];   Q2_array           = efficiencies[:,:,1]
PD_Q1_delta_array  = efficiencies[:,:,2];   PD_Q2_delta_array  = efficiencies[:,:,3]
PD_Q1_lambda_array = efficiencies[:,:,4];   PD_Q2_lambda_array = efficiencies[:,:,5]

t_end = time.time() - t_start
t_end_sec = round(t_end)
t_end_min = round(t_end/60)
t_end_hours = round(t_end/60**2)
print(rf"Finished in {t_end_sec} seconds, or {t_end_min} minutes, or {t_end_hours} hours!")
print(rf"#lambda: {klambda}, #delta: {kdelta}")

pkl_fname = rf"./Data/Lookup_table_lambda_{klambda}_by_delta_{kdelta}.pkl"
data = {'Q1': Q1_array, 'Q2': Q2_array, 'PD_Q1_delta': PD_Q1_delta_array, 'PD_Q2_delta': PD_Q2_delta_array, 
        'PD_Q1_lambda': PD_Q1_lambda_array, 'PD_Q2_lambda': PD_Q2_lambda_array, 
        'lambda array': lambda_array, 'delta array': delta_array}
# with open(pkl_fname, 'wb') as data_file:
#             pickle.dump(data, data_file)