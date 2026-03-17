"""
A script for simulating the dynamics of a twobox grating, calculated using a comoving integrator.

For the grating whose dynamics you wish to simulate, copy paste the grating twobox data (i.e. the optimisation .pkl file) 
and the grating-Qpr lookup table data into the ./Data directory. 
"""

import numpy as np

import pickle

import sys
sys.path.append("../")

from cmvint import odecmvint
import forces
from parameters import Parameters


I, L, m, c = Parameters()
wavelength = 1

# The efficiency factors are too expensive to calculate in real time, so pre-calculated tables are used.
klambda = 1000
kdelta = 1000
runID_load = "Fasymp20_fixgaussian20_50GW"
nonlinear_run = True  # Flag to load the nonlinear data and acceleration function
damping_scaler = 1.  # Only applies if nonlinear_run is False. Set to 0 to disable damping.

# Extract optimised grating (in particular, Gaussian width)
num_processes = 200  # number of processes used in the optimisation to produce opt_gratings_data_fname.pkl
maxtime = 1440
output_opt_idx = 17  # Index of the optimum grating to output, lower corresponding to larger FOM

if nonlinear_run:
    lookup_data_fname = f'./Data/{runID_load}_Lookup_table_lambda_{klambda}_by_delta_{kdelta}.pkl'
else:
    lookup_data_fname = f'./Data/{runID_load}_lsa_Lookup_table_lambda_{klambda}.pkl'

opt_gratings_data_fname = f'../Optimisation/Data/{runID_load}_FOM_optimisation_maxtime{maxtime}'
w, lookup_data = forces.load_essential_data(opt_gratings_data_fname, num_processes, output_opt_idx, lookup_data_fname)
interpolation_funcs = forces.create_interpolation_funcs(lookup_data, has_angle_data=nonlinear_run)

## Optimisation parameters and initial conditions ##
t0 = 0.
x0 = 0
vx0 = 0

# Moderate velocity
y0      = 0.001*w  # metres
phi0    = 0.01*np.pi/180  # degrees converted to radians
vy0     = -0.1  # metres per second
omega0  = -0.005*2*np.pi  # revolutions per second converted to radians per second

# time_MAX = 72.*60*60  # Maximum runtime (seconds)
time_MAX = 10  # testing
velocity_MAX = 0.2*c
h = 1e-3   # Step size  
runID = runID_load  # For saving dynamics data


if nonlinear_run:
    accel = forces.aM  # Choose time derivative state vector function
    accel_args = (w, interpolation_funcs)  # Passed to accel
else:
    accel = forces.aM_linear  
    accel_args = (w, interpolation_funcs, damping_scaler)  

Y0 = np.array([x0,y0,phi0,vx0,vy0,omega0])
positions, angles, times, accels, loop_data = odecmvint(accel, Y0, t0, time_MAX, velocity_MAX, args=accel_args, hstep=h, 
                                                        save_idx=10000, save_file=runID)

phi_nparray      = angles[0,:]
eps_nparray      = angles[1,:]
omega_nparray    = angles[2,:]
eps_rate_nparray = angles[3,:]
theta_nparray    = angles[4,:]
timeM_nparray    = times[0,:]
tau_nparray      = times[1,:]
timeL_nparray    = times[2,:]

runtime = loop_data["Runtime"]
steps = loop_data["Steps"]
STOPPED = loop_data["Stopped"]

save_data = {'YL': positions, 'phiM': phi_nparray, 'phidot': omega_nparray,
             'timeM': timeM_nparray, 'tau': tau_nparray, 'timeL': timeL_nparray, 
             'eps': eps_nparray, 'epsdot': eps_rate_nparray, 'theta': theta_nparray,
             'accel': accels,
             'step': h, 'Runtime (sec)': runtime, 'i': steps, 'Stopped': STOPPED,
             'Initial': Y0, 'Intensity': I}
save_fname = f'./Data/{runID}_Dynamics.pkl'

# Save result
with open(save_fname, 'wb') as data_file:
    pickle.dump(save_data, data_file)