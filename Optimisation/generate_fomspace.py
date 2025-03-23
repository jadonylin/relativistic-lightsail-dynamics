"""
A script to generate FOM values over a 2D-parameter-space for a user-input twobox grating.
Saves the data as a pickle file. The calculation is parallelised using the multiprocess module, 
so set the number of processes before running.
"""

import numpy as np
import pickle

import sys
sys.path.append('../')

import time

import fomspace
from twobox import TwoBox

t_start = time.time()

## Initialise grating
standard_params = [1.4, 0.7, 0.15, 0.35, 0.6, 9., 9., 50., 1., 1.45**2]
standard_grating = TwoBox(*standard_params, wavelength=1., angle=0., Nx=100, nG=25, Qabs=np.inf)
runID = "standard_grating"  # String to add to .pkl filename
grating = standard_grating
num_processes = 6

num_points = 2
w_quantity = "pitch"  
w_range = (1.3, 1.9)
d_range = (0.01, 2.)
ws, ds, FOM_data = fomspace.generate_FOM_space(grating, w_quantity, w_range, d_range, 
                                               num_points, num_processes, 
                                               final_speed=5., goal=0.1, return_grad=False)

t_end = time.time() - t_start
t_end_sec = round(t_end)
t_end_min = round(t_end/60)
t_end_hours = round(t_end/60**2)
print(rf"Finished in {t_end_sec} seconds, or {t_end_min} minutes, or {t_end_hours} hours!")
print(rf"# points: {num_points}, # processes: {num_processes}")

pkl_fname = rf"./Data/{runID}_fomspace_{w_quantity}_by_depth_npoints{num_points}.pkl"
data = {'grating': grating, 'w quantity': w_quantity, 'ws': ws, 'ds': ds, 'FOM data': FOM_data}
with open(pkl_fname, 'wb') as data_file:
    pickle.dump(data, data_file)