# File to defin ilic grating parameters
# use for twobox_test.ipynb: from ilic import all
import numpy as np
wavelength      = 1.5 #/ D1_ND(1.2/100)
grating_pitch   = 1.8 / wavelength
# grating_pitch   = 2.5 / wavelength # original Ilic grating is 1.8, this is for testing over broader wl range

grating_depth   = 0.5 / wavelength
box1_width      = 0.15 * grating_pitch
box2_width      = 0.35 * grating_pitch
box_centre_dist = 0.60 * grating_pitch
box1_eps        = 3.5**2 
box2_eps        = 3.5**2
gaussian_width  = 2* 10   # 2.7180049942915896 * 10
substrate_depth = 0.5 / wavelength
substrate_eps   = 1.45**2

wavelength      = 1.
angle           = 0.0
Qabs            = np.inf
numpoints=1
numG_torcwa=25
numG=2*numG_torcwa

Nx=300