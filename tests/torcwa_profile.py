import sys
sys.path.append('../')
import fom
from parameters import D1_ND, Parameters
from twobox import TwoBox
import numpy as np
import matplotlib.pyplot as plt
import agfunc 
import torch
import time
npa = agfunc.agfunc('torch')
npaa = agfunc.agfunc('autograd')


# wavelength      = 1.5 #/ D1_ND(1.2/100)
# grating_pitch   = 1.8 / wavelength
# # grating_pitch   = 2.5 / wavelength # original Ilic grating is 1.8, this is for testing over broader wl range

# grating_depth   = 0.5 / wavelength
# box1_width      = 0.15 * grating_pitch
# box2_width      = 0.35 * grating_pitch
# box_centre_dist = 0.60 * grating_pitch
# box1_eps        = 3.5**2 
# box2_eps        = 3.5**2
# gaussian_width  = 2* 10   # 2.7180049942915896 * 10
# substrate_depth = 0.5 / wavelength
# substrate_eps   = 1.45**2

# wavelength      = 1.
# angle           = 0.0
# Qabs            = np.inf
# numpoints=1
# numG_torcwa=25
# numG=2*numG_torcwa

# Nx=300


# wavelength_range=np.array([1.01,1.25])
# # derivatives vs angle
# gratingTorcwa = TwoBox(grating_pitch, grating_depth, box1_width, box2_width, box_centre_dist, box1_eps, box2_eps, 
#                  gaussian_width, substrate_depth, substrate_eps,
#                  wavelength, angle, Nx, numG_torcwa, Qabs, RCWA_engine='TORCWA',torcwa_edge_sharpness=45)
# start_time=time.time()
# # gratingTorcwa.show_spectrum(efficiency_quantity='eig',num_plot_points=numpoints)
# gratingTorcwa.show_spectrum(efficiency_quantity='PDt',num_plot_points=100,wavelength_range=wavelength_range)
# print('Torcwa time = ',time.time()-start_time)

from twobox import TwoBox
from ilic import *
Nx=300

start = time.time()
numG=25
gratingTorcwa = TwoBox(grating_pitch, grating_depth, box1_width, box2_width, box_centre_dist, box1_eps, box2_eps, 
                 gaussian_width, substrate_depth, substrate_eps,
                 wavelength, angle, Nx, numG, Qabs, RCWA_engine='TORCWA', torcwa_edge_sharpness=50)

gratingTorcwa.show_angular_efficiency()
print('TORCWA time = ',time.time()-start)