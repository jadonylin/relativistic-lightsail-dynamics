"""
dynplot - dynamics plotting

A module to extract and plot dynamics data.
"""

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True,
                 "text.usetex": True,
                 "font.family": "Computer Modern Roman"})
matplotlib.rc('text.latex', preamble=r'\usepackage{amsmath, physics}')
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

import pickle

import sys
sys.path.append("../")


def extract_dynamics(filename: str, start: int=0, end: int=-1, idx_to_print_state: int=0, use_M_time: bool=False):
    """
    Extract dynamics data from a .pkl file and return the data over a specified timespan.

    Parameters
    ----------
    filename           :   Name of the .pkl file containing the dynamics data relative to the current directory
    start              :   Start index of the data to extract. Indexes frame L time larger than timeL[start]. 
                           Can be an integer index or a float time
    end                :   End index of the data to extract. Indexes frame L time smaller than timeL[end].
                           Can be an integer index or a float time
    idx_to_print_state :   Index of the data to print the state vector
    use_M_time         :   Flag to truncate the time data using frame M time instead of frame L time
    """

    with open(filename, 'rb') as data_file:
        data = pickle.load(data_file)

    x0_dyn = data['Initial']
    h = data['step']
    I_dyn = data['Intensity']
    
    timeL = data['timeL']
    timeM = data['timeM']
    x,y,vx,vy = data['YL']
    phiM = data['phiM']
    phidotM = data['phidot']
    
    eps = data['eps']
    epsdot = data['epsdot']

    # Flag to check if acceleration and aberration data was recorded in the dynamics
    # of the given pkl data
    accel_and_theta_recorded = True 
    try:
        accel_shape = data['accel'].shape
        if accel_shape[0] > accel_shape[1]:
            # Transpose the acceleration data if it is in the wrong shape, only applies to older data
            data['accel'] = data['accel'].T
        ax, ay, aphi = data['accel']
        theta = data['theta']
    except KeyError:
        accel_and_theta_recorded = False
        print("Acceleration and aberration data weren't recorded in the dynamics associated with your chosen pkl file.")
    
    print(f"t0 = {timeL[idx_to_print_state]}")
    print(f"x0 = {x[idx_to_print_state]}")
    print(f"y0 = {y[idx_to_print_state]}")
    print(f"phi0 = {phiM[idx_to_print_state]}")
    print(f"vx0 = {vx[idx_to_print_state]}")
    print(f"vy0 = {vy[idx_to_print_state]}")
    print(f"omega0 = {phidotM[idx_to_print_state]}")

    if isinstance(start, float):
        t_start = start
    else:
        t_start = timeL[start]
    if isinstance(end, float):
        t_end = end
    else:
        t_end = timeL[end]
    
    frame_time = timeL
    if use_M_time:
        frame_time = timeM

    x_trunc = x[(timeL>t_start) & (timeL<=t_end)]
    y_trunc = y[(frame_time>t_start) & (frame_time<=t_end)]
    vx_trunc = vx[(timeL>t_start) & (timeL<=t_end)]
    vy_trunc = vy[(frame_time>t_start) & (frame_time<=t_end)]
    phiM_trunc = phiM[(timeM>t_start) & (timeM<=t_end)]
    phidotM_trunc = phidotM[(timeM>t_start) & (timeM<=t_end)]

    timeM_trunc = timeM[(timeM>t_start) & (timeM<=t_end)]
    timeL_trunc = timeL[(timeL>t_start) & (timeL<=t_end)]

    eps_trunc = eps[(timeM>t_start) & (timeM<=t_end)]
    epsdot_trunc = epsdot[(timeM>t_start) & (timeM<=t_end)]

    coords = [x0_dyn, h, I_dyn, timeM_trunc, timeL_trunc, 
              x_trunc, y_trunc, vx_trunc, vy_trunc, phiM_trunc, phidotM_trunc, 
              eps_trunc, epsdot_trunc]

    if accel_and_theta_recorded:
        ay_trunc = ay[(timeM>t_start) & (timeM<=t_end)]
        aphi_trunc = aphi[(timeM>t_start) & (timeM<=t_end)]
        theta_trunc = theta[(timeM>t_start) & (timeM<=t_end)]
        coords += [ay_trunc, aphi_trunc, theta_trunc]
        
    return coords

def plot_array_on_same_axes(ax: plt.Axes, x: np.ndarray, y: np.ndarray, **kwargs):
    """
    Plot y vs x on axes ax. y can have multiple columns, which are all plotted on the same axes.

    Parameters
    ----------
    ax     :   matplotlib axis object
    x      :   1D array of times
    y      :   Array of features and times. Can be up to 2 dimensions large. Data should be arranged with the 1 axis representing times.
    kwargs :   kwargs to pass to matplotlib.plt.plot()
    """
    if len(y.shape) == 1:
        ax.plot(x, y, **kwargs)
    elif len(y.shape) == 2:
        n_yrows, n_ycols = y.shape  
        if n_yrows < n_ycols:  # Number of times should be greater than the number of features to plot
            print("Warning: number of time data points was fewer than number of features. Check if y should be transposed.")
            y = y.T
        ax.plot(x, y, **kwargs)
    else:
        raise ValueError("y has too many dimensions (maximum 3).")
    return ax

def plot_twinx_array(ax: plt.Axes, x: np.ndarray, y: np.ndarray, **kwargs):
    sec_ax = ax.twinx()
    sec_ax = plot_array_on_same_axes(sec_ax, x, y, **kwargs)
    return ax, sec_ax

def show_standard_axes(ax: plt.Axes, x: np.ndarray, xlabel: str, ylabel: str, show_zero_line: bool=True, ax_width: float=2.5):
    """
    Show standard plot features on the axes of ax. 
    """
    if x is not None:
        ax.set_xlim(x[0],x[-1])
    ax.set(xlabel=xlabel, ylabel=ylabel)
    if show_zero_line:
        ax.axhline(0, color="black", linewidth=ax_width)
    ax.tick_params(which="both", axis='both', width=ax_width, direction='in')
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(ax_width)
    return ax

def color_yaxis(ax: plt.Axes, color: str):
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

    if nrows == 1 and ncols == 1:
        dyn_axs_flat = [dyn_axs]
    else:
        dyn_axs_flat = list(dyn_axs.flatten())
    
    for ax_idx, ax in enumerate(dyn_axs_flat):
        t = times[ax_idx]
        q = coords[ax_idx]
        col = colors[ax_idx]

        if q is None:
            ax.axis('off')
        else:
            ax = plot_array_on_same_axes(ax, t, q, color=col, linewidth=linewidth)
            ax = show_standard_axes(ax, t, xlabels[ax_idx], ylabels[ax_idx], show_zero_line[ax_idx], ax_width)

            if len(second_yaxis_coords) != 0:
                q2 = second_yaxis_coords[ax_idx]
                if q2 is not None:  # Handle secondary Y-axis plot
                    col2 = second_yaxis_colors[ax_idx]
                    ax, sec_ax = plot_twinx_array(ax, t, q2, color=col2, linewidth=linewidth)
                    sec_ax = show_standard_axes(sec_ax, t, xlabels[ax_idx], second_yaxis_ylabels[ax_idx], show_zero_line[ax_idx], ax_width)
                    sec_ax = color_yaxis(sec_ax, col2)
                    ax = color_yaxis(ax, col)
    
    fig.tight_layout()
    return fig, dyn_axs