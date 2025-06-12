import numpy as np

import sys
sys.path.append('../')

import time

from ilic import *

import fom
from twobox import TwoBox

# wavelength_range=np.array([1.01,1.25])
# start_time=time.time()
# gratingGRCWA = TwoBox(grating_pitch, grating_depth, box1_width, box2_width, box_centre_dist, box1_eps, box2_eps, 
#                  gaussian_width, substrate_depth, substrate_eps,
#                  wavelength, angle, Nx, numG, Qabs, RCWA_engine='GRCWA')
# # gratingGRCWA.show_spectrum(efficiency_quantity='eig',num_plot_points=numpoints)
# gratingGRCWA.show_spectrum(efficiency_quantity='PDt',num_plot_points=100,wavelength_range=wavelength_range)
# print('GRCWA time = ',time.time()-start_time)

choose_fom = fom.FoM_asymp
Nx = 100
numG = 25
gratingGRCWA = TwoBox(1.3, grating_depth, box1_width, box2_width, box_centre_dist, box1_eps, box2_eps, 
                       gaussian_width, substrate_depth, substrate_eps,
                       wavelength, angle, Nx, numG, Qabs, RCWA_engine='GRCWA')

# start = time.time()
# ps = np.linspace(1.4, 1.6, 10)
# for p in ps:
#     gratingGRCWA.grating_pitch = p
#     val = fom.FOM_uniform(gratingGRCWA, choose_fom, final_speed=20., goal=0.1, return_grad=True)
# total = time.time()-start

# print(f'GRCWA time (average over {len(ps)} runs) = {total/len(ps)}')

# Profile this function to see which parts are slowest
val = fom.FOM_uniform(gratingGRCWA, choose_fom, final_speed=20., goal=20, return_grad=True)
print(val)