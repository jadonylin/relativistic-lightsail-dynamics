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

from scipy.optimize import curve_fit
import sys
sys.path.append("../")

import fom
from parameters import D1_ND
from twobox import TwoBox


def wavelength_to_beta(wavelengths: np.ndarray, wavelength_start: float=1.):
    """Convert np.array of wavelengths to normalised velocities, assuming a given starting wavelength"""
    Doppler_factors = wavelength_start/wavelengths
    betas = (1 - Doppler_factors**2)/(1 + Doppler_factors**2)
    return betas

def beta_to_wavelength(betas: np.ndarray, wavelength_start: float=1.):
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



def hl_envelopes_idx(s: np.ndarray, dmin: int=1, dmax: int=1, split: bool=False):
    """
    From https://stackoverflow.com/questions/34235530/how-to-get-high-and-low-envelope-of-a-signal
    with updated documentation.
    
    Parameters
    ----------
    s          :   1d-array, data signal from which to extract high and low envelopes
    dmin, dmax :   Size of chunks in terms of indices. Increase to calculate extrema over a broader range
    split      :   If True, split the signal in half along its mean, might help to generate the envelope in some cases
    
    Returns
    -------
    lmin, lmax :   Low/high envelope indices of input signal s
    """

    # locals min      
    lmin = (np.diff(np.sign(np.diff(s))) > 0).nonzero()[0] + 1 
    # locals max
    lmax = (np.diff(np.sign(np.diff(s))) < 0).nonzero()[0] + 1 
    
    if split:
        # s_mid is zero if s centered around x-axis or more generally mean of signal
        s_mid = np.mean(s) 
        # pre-sorting of locals min based on relative position with respect to s_mid 
        lmin = lmin[s[lmin]<s_mid]
        # pre-sorting of local max based on relative position with respect to s_mid 
        lmax = lmax[s[lmax]>s_mid]

    # global min of dmin-chunks of locals min 
    lmin = lmin[[i+np.argmin(s[lmin[i:i+dmin]]) for i in range(0,len(lmin),dmin)]]
    # global max of dmax-chunks of locals max 
    lmax = lmax[[i+np.argmax(s[lmax[i:i+dmax]]) for i in range(0,len(lmax),dmax)]]
    
    return lmin, lmax


def coordinates_to_envelopes(coordinate_arrays: list[list], envelope_idx: int=10000):
    """
    Calculate the envelopes of a list of coordinate arrays. 
    The envelopes are calculated by taking the high and low envelope indices 
    of the coordinate arrays.
    
    Parameters
    ----------
    coordinate_arrays :   List of coordinate arrays to calculate envelopes for
    envelope_idx      :   Size of chunks in terms of indices. Increase to 
                          calculate extrema over a broader range
    
    Returns
    -------
    min_indices, max_indices :   Low/high envelope indices of input signal s
    """
    
    num_coordinates = len(coordinate_arrays)
    min_times = [[] for _ in range(num_coordinates)]
    max_times = [[] for _ in range(num_coordinates)]
    min_coordinates = [[] for _ in range(num_coordinates)]
    max_coordinates = [[] for _ in range(num_coordinates)]
    # TODO: Avoid hard coding the indices. Can we check which coordinates are 
    #       monotonic and avoid calculating the envelope for those?
    # Avoid calculating the envelopes for x, vx or frame times
    ignore_indices = [0,2,4,5,6]  # Of coordinate_arrays
    timeM = coordinate_arrays[4]
    timeL = coordinate_arrays[6]
    times = [timeL, timeL, timeL, timeL, timeM, timeM, timeL, 
             timeM, timeM, timeM, timeM, timeL, timeL, timeM, timeM]  # Frame associated with each coordinate
    
    for n in range(num_coordinates):
        time = np.array(times[n])
        coord = np.array(coordinate_arrays[n])

        if n in ignore_indices:
            min_idx = -1
            max_idx = -1
            min_times[n] += [time[min_idx]]
            max_times[n] += [time[max_idx]]
            min_coordinates[n] += [coord[min_idx]]
            max_coordinates[n] += [coord[max_idx]]
        else:
            min_idx, max_idx = hl_envelopes_idx(coord, dmin=envelope_idx, dmax=envelope_idx)
            min_times[n] += list(time[min_idx])
            max_times[n] += list(time[max_idx])
            min_coordinates[n] += list(coord[min_idx])
            max_coordinates[n] = max_coordinates[n] + list(coord[max_idx])
    return min_times, max_times, min_coordinates, max_coordinates


def load_coordinate_array_envelopes(filename: str, n_sets: int=7, n_chunks_per_set: int=100, envelope_idx: int=10000):
    """
    The data is saved as coordinate arrays in chunks of time (e.g. 100000 time points).
    Here, we load the data in a number of chunks n_chunks_per_set, concatenate the data, 
    then calculate the envelope of the data to reduce the array size.
    """
    num_coordinates = 15  # TODO: how to avoid hard coding this?
    min_time_arrays = [[] for _ in range(num_coordinates)]
    max_time_arrays = [[] for _ in range(num_coordinates)]
    min_coordinate_arrays = [[] for _ in range(num_coordinates)]
    max_coordinate_arrays = [[] for _ in range(num_coordinates)]
    with open(filename, 'rb') as storage:
        try:
            load_count = 0
            while load_count < n_sets:
                chunks_of_coords = []
                
                # Load n_chunks_per_set chunks of coordinate arrays sequentially to avoid memory issues
                # E.g. if cmvint saved coordinate arrays after every 10000 iterations, then:
                #   chunk n = [[coord 1, 10000 floats], [coord 2, 10000 floats], ..., [coord 15, 10000 floats]]
                #   chunks_of_coords = [chunk 1, chunk 2, ..., chunk n_chunks_per_set]
                for chunk_idx in range(n_chunks_per_set):
                    chunks_of_coords.append(pickle.load(storage))
                
                # Combine the chunks into coordinate-array standard for output
                #   coordinate_arrays = [[coord 1, 10000*n_chunks_per_set floats], [coord 2, 10000*n_chunks_per_set floats], 
                #                       ..., [coord 15, 10000*n_chunks_per_set floats]]
                coordinate_arrays = chunks_of_coords[0]
                num_coordinates = len(coordinate_arrays)
                for chunk in chunks_of_coords[1:]:
                    for n in range(num_coordinates):
                        coordinate_arrays[n] += chunk[n]
                
                # Truncate coordinate arrays of size (10000*n_chunks_per_set) to envelopes of size ni (1 <= i <= 15)
                #   LHS arrays = [[coord 1 envelopes, n1 floats], [coord 2 envelopes], ..., [coord 15 envelopes]]
                min_times, max_times, min_coordinates, max_coordinates = coordinates_to_envelopes(coordinate_arrays, envelope_idx)
                
                # Add to existing set of chunks to create coordinate arrays
                for n in range(num_coordinates):
                    min_time_arrays[n] += min_times[n]
                    max_time_arrays[n] += max_times[n]
                    min_coordinate_arrays[n] += min_coordinates[n]
                    max_coordinate_arrays[n] += max_coordinates[n]
                
                load_count += 1
                print(f"Loaded {n_chunks_per_set} chunks (set #{load_count}) with {n_chunks_per_set*len(coordinate_arrays[1])} points.")
        except EOFError:
            pass
    
    # Output arrays = [[coord 1 envelopes, n_sets*n1 floats], [coord 2 envelopes, n_sets*n2 floats], 
    #                 ..., [coord 15 envelopes, n_sets*n15 floats]]
    return min_time_arrays, max_time_arrays, min_coordinate_arrays, max_coordinate_arrays


def generate_lsa_spectrum(grating: TwoBox, speed_range: list=(0.,5.), I: float=5e8, num_points: int=200, 
                          normalise: bool=False, use_perturbed: bool=False):
    """
    Generate linear stability analysis information across a given spectrum of wavelengths.

    Parameters
    ----------
    grating       :   Grating whose spectrum is generated
    speed_range   :   Maximum and minimum speeds, between which the corresponding wavelengths form the spectrum 
    I             :   Laser intensity
    num_points    :   Number of points in the spectrum
    normalise     :   Flag to normalise the eigenvalues, eigenvectors and Jacobian coefficients
    use_perturbed :   Flag to use the analytic eigenvalues from first order perturbation theory.
    """

    wavelength_range = np.linspace(1/D1_ND(speed_range[0]/100), 1/D1_ND(speed_range[1]/100), num_points)
    
    restoring_coeffs = np.zeros((num_points,4))
    damping_coeffs = np.zeros((num_points,4))
    real_eigvals = np.zeros((num_points,4))
    imag_eigvals = np.zeros((num_points,4))
    eigvec_moduli = np.zeros((num_points,4,4))
    preal_eigvals = np.zeros((num_points,4))
    pimag_eigvals = np.zeros((num_points,4))
    for i in range(num_points):
        wavelength = wavelength_range[i]
        input_wavelength = grating.wavelength
        grating.wavelength = wavelength 
        lsa = grating.to_numpy(fom.lsa_info(grating, I, normalise, use_perturbed))
        if use_perturbed:
            _, rest, damp, real, imag, eigvecs, preal, pimag = lsa
            preal_eigvals[i,:] = preal
            pimag_eigvals[i,:] = pimag
        else:
            _, rest, damp, real, imag, eigvecs = lsa
        restoring_coeffs[i,:] = rest
        damping_coeffs[i,:] = damp
        real_eigvals[i,:] = real
        imag_eigvals[i,:] = imag
        eigvec_moduli[i,:,:] = np.abs(eigvecs)**2  # Eigenvectors are already normalised
    grating.wavelength = input_wavelength
    
    return restoring_coeffs, damping_coeffs, real_eigvals, imag_eigvals, eigvec_moduli, preal_eigvals, pimag_eigvals


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