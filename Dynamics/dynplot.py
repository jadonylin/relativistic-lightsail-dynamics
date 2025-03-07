"""
dynplot - dynamics plotting

A module to reduce repetitive code when plotting dynamics results.
"""

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True,
                 "text.usetex": True,
                 "font.family": "Computer Modern Roman"})
# matplotlib.rc('text.latex', preamble=r'\usepackage{amsmath, physics}')
SMALL_SIZE = 18
MEDIUM_SIZE = 20
BIGGER_SIZE = 22
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

import numpy as np
from numpy.linalg import norm as norm

from scipy.optimize import curve_fit
import sys
sys.path.append("../")

from parameters import D1_ND
from twobox import TwoBox


def wavelength_to_beta(wavelengths, wavelength_start=1.):
    """Convert np.array of wavelengths to normalised velocities, assuming a given starting wavelength"""
    Doppler_factors = wavelength_start/wavelengths
    betas = (1 - Doppler_factors**2)/(1 + Doppler_factors**2)
    return betas

def beta_to_wavelength(betas, wavelength_start):
    """Convert np.array of normalised velocities to wavelengths, assuming a given starting wavelength"""
    Doppler_factors = np.sqrt((1-betas)/(1+betas))
    wavelengths = wavelength_start/Doppler_factors
    return wavelengths


def func(x,a,b,c,d,e):
    """
    "x": vx/c
    """
    return a*x**4 + b*x**3 + c*x**2 + d*x + e

def v_to_t(v, v_data, t_data):
    """Convert velocities v to times t using polynomial fit on v_data and t_data"""
    popt, pcov = curve_fit(func, v_data, t_data)
    return func(v, *popt)

def t_to_v(t, v_data, t_data):
    """Convert times t to velocities v using polynomial fit on t_data and v_data"""
    popt, pcov = curve_fit(func, t_data, v_data)
    return func(t, *popt)


def generate_lsa_spectrum(grating: TwoBox, speed_range: list=(0.,5.), I: float=5e8, num_points: int=200):
    """
    Generate linear stability analysis information across a given spectrum of wavelengths.

    Parameters
    ----------
    grating :   Grating whose spectrum is generated
    speed_range :   Maximum and minimum speeds, between which the corresponding wavelengths form the spectrum 
    """

    wavelength_range = np.linspace(1/D1_ND(speed_range[0]/100), 1/D1_ND(speed_range[1]/100), num_points)
    
    restoring_coeffs = np.zeros((4,num_points))
    damping_coeffs = np.zeros((4,num_points))
    real_eigvals = np.zeros((4,num_points))
    imag_eigvals = np.zeros((4,num_points))
    eigvec_moduli = np.zeros((4,4,num_points))

    for i in range(num_points):
        wavelength = wavelength_range[i]
        _, rest, damp, real, imag, eigvecs = grating.lsa_info(wavelength, I)

        restoring_coeffs[:,i] = rest
        damping_coeffs[:,i] = damp
        real_eigvals[:,i] = real
        imag_eigvals[:,i] = imag
        print(imag)

        _eigvec_norms = norm(eigvecs, axis=0).T
        eigvec_norms = _eigvec_norms[:,None]
        # print(eigvecs)
        # print(_eigvec_norms)
        # print(eigvec_norms)

        eigvec_moduli[:,:,i] = np.abs(eigvecs)/norm(eigvecs, axis=0)[:,None]

        # print(eigvec_moduli[:,:,i])

    return restoring_coeffs, damping_coeffs, real_eigvals, imag_eigvals, eigvec_moduli


def plot_array_on_same_axes(ax, x, y, **kwargs):
    """
    Plot y vs x on axes ax. y can have multiple rows, which are all plotted on the same axes.

    kwargs are passed to plt.plot()
    """
    if len(y.shape) > 1:  
        if y.shape[0] > y.shape[1]:  # Data should be arranged with columns representing times
            y = y.T
        for y_row in y:
            ax.plot(x, y_row, **kwargs)
    else:
        ax.plot(x, y, **kwargs)
    return ax

def plot_twinx_array(ax, x, y, **kwargs):
    sec_ax = ax.twinx()
    sec_ax = plot_array_on_same_axes(sec_ax, x, y, **kwargs)
    return ax, sec_ax

def show_standard_axes(ax, x, xlabel, ylabel, show_zero_line, color, ax_width):
    """
    Show standard plot features on the axes of ax. 
    """
    ax.set_xlim(x[0],x[-1])
    ax.set(xlabel=xlabel, ylabel=ylabel)
    if show_zero_line:
        ax.axhline(0, linestyle="--", color=color)
    ax.tick_params(which="both", axis='both', width=ax_width, direction='in')
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(ax_width)
    return ax

def color_yaxis(ax, color):
    ax.yaxis.label.set_color(color)
    ax.tick_params(axis='y',labelcolor=color)
    return ax

def show_dynamics(nrows: int, ncols: int, times: list, coords: list, 
                  colors: list, xlabels: list, ylabels: list, show_zero_line: list,
                  second_yaxis_coords: list=[], second_yaxis_colors: list=[], second_yaxis_ylabels: list=[],
                  ax_width: float=2.5, linewidth: float=2.5):
    """
    Plot dynamics results (coordinates and velocities) over time in a subplots grid.

    Parameters
    ----------
    nrows                :   Number of subplot rows (passed to plt.subplots)
    ncols                :   Number of subplot columns (passed to plt.subplots)
    times                :   List of arrays of times to plot over. Should be length nrows*ncols
    coords               :   List of arrays of coordinates to plot over time. Should be length nrows*ncols
    colors               :   List of strings representing colours to plot for each coordinate. Should be length nrows*ncols
    xlabels              :   List of strings for each coordinate's X axis label. Should be length nrows*ncols
    ylabels              :   List of strings for each coordinate's Y axis label. Should be length nrows*ncols
    show_zero_line       :   List of bools to flag whether or not the Y = 0 line should be plotted. Should be length nrows*ncols
    second_yaxis_coords  :   List of arrays for second set of Y coordinates to show on the same X axis. Should be length nrows*ncols.
                             If array is None, no second set of Y coordinates is plotted.
    second_yaxis_colors  :   List of strings representing colours to plot for each second Y coordinate. Should be length nrows*ncols
    second_yaxis_ylabels :   List of strings for each coordinate's second Y axis label. Should be length nrows*ncols

    Returns
    -------
    fig :   matplotlib figure object for the dynamics subplots 
    axs :   matplotlib axis object for the dynamics subplots 
    """
    
    fig, dyn_axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*6,nrows*4))

    dyn_axs_flat = list(dyn_axs.flatten())
    for ax_idx, ax in enumerate(dyn_axs_flat):
        t = times[ax_idx]
        q = coords[ax_idx]
        col = colors[ax_idx]

        ax = plot_array_on_same_axes(ax, t, q, color=col, linewidth=linewidth)
        ax = show_standard_axes(ax, t, xlabels[ax_idx], ylabels[ax_idx], show_zero_line[ax_idx], col, ax_width)

        if len(second_yaxis_coords) != 0:
            q2 = second_yaxis_coords[ax_idx]
            if q2 is not None:  # Handle secondary Y-axis plot
                col2 = second_yaxis_colors[ax_idx]
                ax, sec_ax = plot_twinx_array(ax, t, q2, color=col2, linewidth=linewidth)
                sec_ax = show_standard_axes(sec_ax, t, xlabels[ax_idx], second_yaxis_ylabels[ax_idx], show_zero_line[ax_idx], col2, ax_width)
                sec_ax = color_yaxis(sec_ax, col2)
                ax = color_yaxis(ax, col)
    
    fig.tight_layout()
    return fig, dyn_axs