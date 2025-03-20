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
runID_load = "MdSnpminOpt_lsa"
nonlinear_run = False  # Flag to load the nonlinear data and acceleration function

if nonlinear_run:
    lookup_data_fname = f'./Data/{runID_load}_Lookup_table_lambda_{klambda}_by_delta_{kdelta}.pkl'
else:
    lookup_data_fname = f'./Data/{runID_load}_Lookup_table_lambda_{klambda}.pkl'
opt_gratings_data_fname = f'./Data/FOM_optimisation_maxfev9000'
num_processes = 18  # number of processes used in the optimisation to produce opt_gratings_data_fname.pkl
output_opt_idx = 0  # Index of the optimum grating to output, lower corresponding to larger FOM

w, lookup_data = forces.load_essential_data(opt_gratings_data_fname, num_processes, output_opt_idx, lookup_data_fname)
interpolation_funcs = forces.create_interpolation_funcs(lookup_data)

if nonlinear_run:
    accel = forces.aM  # Choose time derivative state vector function
    accel_args = (w, interpolation_funcs)  # Passed to accel
else:
    accel = forces.aM_linear  
    accel_args = (w, interpolation_funcs, 0.)  
    # accel = forces.aM_forces_linear
    # accel_args = (w, interpolation_funcs)

## Optimisation parameters and initial conditions ##
x0 = 0
vx0 = 0

# # Standard perturbation
# y0      = 0.05*w  # metres
# phi0    = 1.*np.pi/180  # degrees converted to radians
# vy0     = -1.  # metres per second
# omega0  = -1.  # radians per second

# Reduced perturbation
y0      = 0.01*w  # metres
phi0    = 0.1*np.pi/180  # degrees converted to radians
vy0     = -0.1  # metres per second
omega0  = -0.05*2*np.pi  # rotations per second converted to radians per second

## Ilic - 1st
# y0      = 1.3183489420398592e-07
# phi0    = -3.858944981371387e-09
# vy0     = -1.7076261180956787
# omega0  = -1.0411235013620457
## Ilic - 2nd 
# y0      = 5.64330183341613e-08
# phi0    = 1.9109909983391035e-08
# vy0     = -1.998710957511268
# omega0  = -0.06679404481428496


Y0 = np.array([x0,y0,phi0,vx0,vy0,omega0])
time_MAX = 8.5*60*60  # Maximum runtime (seconds)
# time_MAX = 0.001  # Maximum runtime (seconds)
velocity_MAX = 0.05*c
h = 1e-4   # Step size  
runID = "MdSnpminOpt_lsa_no_damping"  # Added to the output data filename

positions, angles, times, accels, loop_data = odecmvint(accel, Y0, time_MAX, velocity_MAX, args=accel_args, hstep=h)

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