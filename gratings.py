"""
A module to store diffraction grating parameters from lightsail literature.
"""
import numpy as np
import twobox

wavelength      = 1.
angle           = 0.0
Nx              = 100
numpoints       = 1
numG_torcwa     = 25
numG            = 2*numG_torcwa
Qabs            = np.inf

def ilic():
    """
    Ilic & Atwater 2019
    """
    wavelength      = 1.5 #/ D1_ND(1.2/100)
    
    grating_pitch   = 1.8 / wavelength
    grating_depth   = 0.5 / wavelength
    box1_width      = 0.15 * grating_pitch
    box2_width      = 0.35 * grating_pitch
    box_centre_dist = 0.60 * grating_pitch
    
    box1_eps        = 3.5**2 
    box2_eps        = 3.5**2
    
    gaussian_width  = 2*10   # 2.7180049942915896 * 10
    substrate_depth = 0.5 / wavelength
    substrate_eps   = 1.45**2

    params = [grating_pitch, grating_depth, box1_width, box2_width, box_centre_dist, \
              box1_eps, box2_eps, \
              gaussian_width, substrate_depth, substrate_eps]

    grating = twobox.TwoBox(*params, wavelength=1., angle=0., Nx=Nx, nG=numG_torcwa, Qabs=np.inf, RCWA_engine="TORCWA")
    return grating

def gao():
    """
    Gao 2024 Flexible lightsail Figure 6, TE region
    """
    wavelength      = 1.064

    grating_pitch   = 1.6/wavelength
    grating_depth   = 0.4/wavelength
    box1_width      = 0.6/wavelength
    box2_width      = 0.2/wavelength
    box_centre_dist = (0.19 + 0.5*(0.6 + 0.2))/wavelength
    
    box1_eps        = 2**2
    box2_eps        = 2**2
    
    gaussian_width  = 2*10
    substrate_depth = 0.2/wavelength
    substrate_eps   = 2**2

    params = [grating_pitch, grating_depth, box1_width, box2_width, box_centre_dist, \
              box1_eps, box2_eps, \
              gaussian_width, substrate_depth, substrate_eps]
    
    grating = twobox.TwoBox(*params, wavelength=1., angle=0., Nx=Nx, nG=numG_torcwa, Qabs=np.inf, RCWA_engine="TORCWA")
    return grating