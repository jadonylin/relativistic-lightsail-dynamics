"""
Test the runtime of figure of mert functions.
"""

import numpy as np

import sys
sys.path.append("../")
sys.path.append("../Optimisation")

import timeit

import Optimisation.opt as opt
from parameters import Parameters, D1_ND
I0, L, m, c = Parameters()
from twobox import TwoBox

Nx = 100
nG = 25
Doppler = D1_ND([0.2,0])


# Test gratings
standard_params = [1.4, 0.7, 0.15, 0.35, 0.6, 9., 9., 20., 1., 1.45**2]
standard_grating = TwoBox(*standard_params, wavelength=1., angle=0., Nx=Nx, nG=nG, Qabs=np.inf)
# standard_grating.show_permittivity(show_analytic_box=True)

near_cutoff_params = [1.999, 0.7, 0.15, 0.35, 0.6, 9., 9., 20., 1., 1.45**2] 
near_cutoff_grating = TwoBox(*near_cutoff_params, wavelength=1., angle=0., Nx=Nx, nG=nG, Qabs=np.inf)

thin_pillars_params = [1.4, 0.7, 0.01, 0.01, 0.6, 9., 9., 20., 1., 1.45**2]
thin_pillars_grating = TwoBox(*thin_pillars_params, wavelength=1., angle=0., Nx=Nx, nG=nG, Qabs=np.inf)

closebox_params = [1.4, 0.7, 0.15, 0.35, 0.27, 9., 9., 20., 1., 1.45**2]
closebox_grating = TwoBox(*closebox_params, wavelength=1., angle=0., Nx=Nx, nG=nG, Qabs=np.inf)

overlap_params = [1.4, 0.7, 0.15, 0.35, 0.15, 12., 9., 20., 1., 1.45**2]
overlap_grating = TwoBox(*overlap_params, wavelength=1., angle=0., Nx=Nx, nG=nG, Qabs=np.inf)

clipbox_params = [1.4, 0.7, 0.35, 0.9, 1.1, 9., 9., 20., 1., 1.45**2]
clipbox_grating = TwoBox(*clipbox_params, wavelength=1., angle=0., Nx=Nx, nG=nG, Qabs=np.inf)


def FD(grating: TwoBox) -> float:
    """
    Calculate the grating single-wavelength figure of merit FD.

    Parameters
    ----------
    grating :           TwoBox instance containing the grating parameters
    """    
    return grating.FoM(I0, grad_method="finite")

def FD_LvR(grating: TwoBox) -> float: 
    return grating.FoM_LvR(I0, grad_method="finite")

def FOM(grating: TwoBox, goal: float=0.1, return_grad: bool=True, FD: callable=FD):
    return opt.FOM_uniform(grating, 5., goal, return_grad, FD)



goal = 0.1
number = 2

s = """f = FOM(standard_grating, 0.1, False, FD)"""
runtime = timeit.timeit(stmt=s, number=number, globals=globals())
print(runtime)
s = """f = FOM(standard_grating, 0.1, False, FD_LvR)"""
runtime = timeit.timeit(stmt=s, number=number, globals=globals())
print(runtime)


s = """f = FOM(standard_grating, 0.1, True, FD)"""
runtime = timeit.timeit(stmt=s, number=number, globals=globals())
print(runtime)
# s = """f = FOM(standard_grating, 0.1, True, FD_LvR)"""
# runtime = timeit.timeit(stmt=s, number=number, globals=globals())
# print(runtime)