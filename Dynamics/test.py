import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True,
                 "text.usetex": True,
                 "font.family": "Computer Modern Roman"})


## Plotting font options ##
SMALL_SIZE = 16
MEDIUM_SIZE = 18
BIGGER_SIZE = 20

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

import numpy as np

import pathlib

import sys
sys.path.append("../")

import dynplot
import Optimisation.opt as opt
import parameters
_, L, m, c = parameters.Parameters()
from twobox import TwoBox


l_width = 2.5
colorY = "blue"
colorvY = "dodgerblue"  #cornflowerblue
colorphi = "red"  #"royalblue"
colorvphi = "tomato"  #coral
colorReal = (0.7, 0, 0)
colorImag = 'blue'
colorY2 = "blue"
colorvY2 = "red"  #cornflowerblue
colorphi2 = "black"  #"royalblue"
colorvphi2 = "lightgreen"  #coral



import autolib

# runID = "PureTranslationOpt"
runID = "MdSnpmin20_torcwa"
fname_preamble = '/Users/jlin0351/Library/CloudStorage/OneDrive-TheUniversityofSydney(Students)/Doppler Damping - Jadon Lin/Documentation/Data/relativistic-lightsail-dynamics'
I = 5e8
final_speed = 20.
speed_range = np.array([0.,final_speed])
num_points = 200

# Calculate the linear stability information for the chosen grating whose eigenvectors 
# you wish to plot alongside the dynamics
num_cores = 96
maxtime = 2760
output_opt_idx = 2

pkl_fname = fname_preamble + f'/Optimisation/Jadon\'s results/MdS npmin FoM/final_speed{int(final_speed)}/maxtime{maxtime}/{runID}_FOM_optimisation_maxtime{maxtime}'
# pkl_fname = fname_preamble + f'/Optimisation/Jadon\'s results/MdS npmin FoM/final_speed20/mirror/{runID}_FOM_optimisation_maxtime{maxtime}'
_, _, grating = opt.extract_opt(pkl_fname, num_processes=num_cores, output_opt_idx=output_opt_idx)
grating.npa = autolib.AutoLib('torch', device='cpu', precision='double')  # twobox.npa isn't saved during optimisation, must set manually

# Fdmp_params = [1.226     , 2.90479687, 0.31178695, 0.0071848 , 0.9995    , 2.96553672, 4.30675011]
# Fdmp_grating = TwoBox(*Fdmp_params, gaussian_width=5., substrate_depth=1., substrate_eps=-1e6,
#                       wavelength=1., angle=0., Nx=100, nG=12, Qabs=np.inf,
#                       RCWA_engine="TORCWA", torcwa_edge_sharpness=45, mirror_substrate=True)
# Fdmp_grating.invert_unit_cell = True

# Fdmp_grating.show_permittivity()
# Fdmp_grating.show_spectrum(0., "PDr", wavelength_range=[1,1.225], I=I, grad_method="grad")
# grating = Fdmp_grating

# restoring_coeffs, damping_coeffs, real_eigvals, imag_eigvals, eigvec_moduli = dynplot.generate_lsa_spectrum(grating, speed_range, I, num_points, normalise=True)


import dynplot

betas = np.linspace(0,final_speed/100, num_points)
vels = [[b,0] for b in betas]
Dopplers = np.array(parameters.D1_ND(vels))
gammas = np.array(parameters.gamma_ND(vels))
Dsq = Dopplers**2
D_plus_1 = (Dopplers+1)/(Dopplers*(gammas+1))
fig, ax = plt.subplots(1, figsize=(6,5))
ax = dynplot.plot_array_on_same_axes(ax, betas, Dopplers, linewidth=2.5, color="red", linestyle="--", label=r"$D$")
ax = dynplot.plot_array_on_same_axes(ax, betas, Dsq, linewidth=2.5, color="blue", linestyle="-", label=r"$D^2$")
ax = dynplot.plot_array_on_same_axes(ax, betas, D_plus_1, linewidth=2.5, color="green", linestyle="-.", label=r"$D+1/(D(\gamma+1))$")
ax = dynplot.show_standard_axes(ax, betas, xlabel=r"$\beta$", ylabel=r"Doppler factor", show_zero_line=False, ax_width=2.5)
fig.legend(loc='upper right')
fig.savefig("Doppler_factor.pdf", dpi=300, bbox_inches='tight')
fig.show()