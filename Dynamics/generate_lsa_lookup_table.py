import sys
sys.path.append('../')

import numpy as np
import dill as pickle

from pathlib import PosixPath
user_home_path = PosixPath('~/')
user_home_path_full = user_home_path.expanduser()

import time

import Optimisation.opt as opt
from parameters import D1_ND
from twobox import TwoBox

t_start = time.time()

## Initialise grating
runID = "Fasymp20_cutoff"
final_speed = 20.
num_cores = 200
maxtime = 1440
output_opt_idx = 18

common_path = user_home_path_full / "Library/CloudStorage/OneDrive-TheUniversityofSydney(Students)/Doppler Damping - Jadon Lin/Documentation/Data/relativistic-lightsail-dynamics/Optimisation/Jadon's results"
custom_folder_path = f"Fasymp/final_speed{int(final_speed)}/maxtime{int(maxtime)}/{runID}"
fname_preamble = common_path / custom_folder_path

opt_grating_basefname = fname_preamble / f"{runID}_FOM_optimisation_maxtime{maxtime}"
_, _, _opt_grating = opt.extract_opt(opt_grating_basefname, num_processes=num_cores, output_opt_idx=output_opt_idx)
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

print(op)
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
