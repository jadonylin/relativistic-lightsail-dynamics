import numpy as np
import pickle
import time

from parameters import D1_ND
from twobox import TwoBox


t_start = time.time()

## Initialise grating
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


klambda = 1000  # Number of lambda' points
v_final = 5/100
lambda_final = 1/D1_ND(v_final)
lambda_array = np.linspace(wavelength, lambda_final, klambda)

kdelta = 1000  # Number of delta' points
delta_max = 15*np.pi/180
delta_min = -delta_max
delta_array = np.linspace(delta_min, delta_max, kdelta)

Q1_array           = np.zeros((klambda, kdelta));   Q2_array           = np.zeros((klambda, kdelta))
PD_Q1_delta_array  = np.zeros((klambda, kdelta));   PD_Q2_delta_array  = np.zeros((klambda, kdelta))
PD_Q1_lambda_array = np.zeros((klambda, kdelta));   PD_Q2_lambda_array = np.zeros((klambda, kdelta))


for i in range(klambda):
    grating.wavelength = lambda_array[i]
    for j in range(kdelta):
        grating.angle = delta_array[j]

        Q1, Q2, PD_Q1_delta, PD_Q2_delta, PD_Q1_lambda, PD_Q2_lambda = grating.return_Qs_auto()

        Q1_array[i,j] = Q1;                         Q2_array[i,j] = Q2
        PD_Q1_delta_array[i,j] = PD_Q1_delta;       PD_Q2_delta_array[i,j] = PD_Q2_delta
        PD_Q1_lambda_array[i,j] = PD_Q1_lambda;     PD_Q2_lambda_array[i,j] = PD_Q2_lambda


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
with open(pkl_fname, 'wb') as data_file:
            pickle.dump(data, data_file)