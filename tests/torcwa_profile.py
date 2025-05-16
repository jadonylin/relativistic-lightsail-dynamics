import numpy as np

import sys
sys.path.append('../')

import time

import fom
from ilic import *
import Optimisation.opt as opt
from twobox import TwoBox

# wavelength_range=np.array([1.01,1.25])
# # derivatives vs angle
# gratingTorcwa = TwoBox(grating_pitch, grating_depth, box1_width, box2_width, box_centre_dist, box1_eps, box2_eps, 
#                  gaussian_width, substrate_depth, substrate_eps,
#                  wavelength, angle, Nx, numG_torcwa, Qabs, RCWA_engine='TORCWA',torcwa_edge_sharpness=45)
# start_time=time.time()
# # gratingTorcwa.show_spectrum(efficiency_quantity='eig',num_plot_points=numpoints)
# gratingTorcwa.show_spectrum(efficiency_quantity='PDt',num_plot_points=100,wavelength_range=wavelength_range)
# print('Torcwa time = ',time.time()-start_time)

# Nx=300

# start = time.time()
# numG=25
# gratingTorcwa = TwoBox(grating_pitch, grating_depth, box1_width, box2_width, box_centre_dist, box1_eps, box2_eps, 
#                  gaussian_width, substrate_depth, substrate_eps,
#                  wavelength, angle, Nx, numG, Qabs, RCWA_engine='TORCWA', torcwa_edge_sharpness=50)

# gratingTorcwa.show_angular_efficiency()
# print('TORCWA time = ',time.time()-start)


Nx = 100
numG = 12
gratingTorcwa = TwoBox(1.3, grating_depth, box1_width, box2_width, box_centre_dist, box1_eps, box2_eps, 
                       gaussian_width, substrate_depth, substrate_eps,
                       wavelength, angle, Nx, numG, Qabs, RCWA_engine='TORCWA', torcwa_edge_sharpness=50)

# start = time.time()
# ps = np.linspace(1.4, 1.6, 10)
# for p in ps:
#     gratingTorcwa.grating_pitch = p
#     val = fom.FOM_uniform(gratingTorcwa, final_speed=20., goal=0.1, return_grad=True)
# total = time.time()-start

# print(f'TORCWA time (average over {len(ps)} runs) = {total/len(ps)}')

# Profile this function to see which parts are slowest
val = fom.FOM_uniform(gratingTorcwa, final_speed=20., goal=20, return_grad=True)
print(val)