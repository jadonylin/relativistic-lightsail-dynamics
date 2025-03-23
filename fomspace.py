"""
ewh: Efficiency-W-Height. 
"""

# IMPORTS ########################################################################################################################
from copy import deepcopy

import itertools

import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Helvetica"
LINE_WIDTH = 2.2
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

import multiprocess as mp

import numpy as np

import pickle

from twobox import TwoBox
import Optimisation.opt as opt


def generate_FOM_space(grating: TwoBox, w_quantity: str="pitch", w_range: tuple=(1.,2.), depth_range: tuple=(0.,1.), 
                       num_points_per_dimension: int=1000, num_processes: int=8, **kwargs):
    """
    Calculate the FOM for the twobox at variable w and height. w can be chosen from a preset array of parameters.  

    Parameters
    ----------
    grating                  :   TwoBox grating whose ewh you want to generate
    w_quantity               :   w quantity you want to generate. Can be "pitch" (unit cell width), "box_width" (width of box 1) or "wavelength". 
    w_range                  :   Upper and lower limit on values for w_quantity.
    depth_range              :   Depths range
    num_points_per_dimension :   Number of points to plot in each dimension
    num_processes            :   Number of processes to use in parallel calculation
    **kwargs                 :   Keyword arguments to pass to FOM
    """

    _grating = deepcopy(grating) # avoid modifying initial grating
    w_min, w_max = w_range
    depth_min, depth_max = depth_range
    ds = np.linspace(depth_min,depth_max,num_points_per_dimension)
    ws = np.linspace(w_min,w_max,num_points_per_dimension)

    def _FOM(*args):
        match w_quantity:
            case "pitch":
                _grating.grating_pitch = args[0][0]
            case "box_width":
                _grating.box1_width = args[0][0]
            case "wavelength":
                _grating.wavelength = args[0][0]
            case _:
                raise ValueError("Unrecognised w_quantity. Must be one of 'pitch', 'box_width' or 'wavelength'.")
        _grating.grating_depth = args[0][1]
        return opt.FOM_uniform(_grating, **kwargs) 

    param_inputs = ((w,d) for w,d in itertools.product(ws,ds))
    p = mp.Pool(num_processes)
    _data = p.map(_FOM, param_inputs)
    data = np.reshape(np.array(_data), (num_points_per_dimension, num_points_per_dimension))
    p.close()
    p.join()

    return ws, ds, data

def show_FOM_space(data_filename):
    """
    Show the FOM for the twobox as 2D colour plot over w and grating depth.  

    Parameters
    ----------
    data_filename :   Simulation parameters and FOM data
    """

    with open(data_filename, 'rb') as data_file:
        data = pickle.load(data_file)

    ws = data["ws"]
    grating_depths = data["ds"]
    FOM_data = data["FOM data"]
    w_quantity = data["w quantity"]

    w_min = ws[0]
    w_max = ws[-1]
    h1_min = grating_depths[0]
    h1_max = grating_depths[-1]

    # Maximum value to map the colourbar to
    fig, ax = plt.subplots(1, figsize=(10,8))
    
    match w_quantity:
        case "pitch":
            ax.set(xlabel=r"$\Lambda'/\lambda_0$", ylabel=r"$h_1'/\lambda_0$")
        case "box_width":
            ax.set(xlabel=r"$w'/\Lambda'$", ylabel=r"$h_1'/\lambda_0$")
        case "wavelength":
            ax.set(xlabel=r"$\lambda'/\Lambda'$", ylabel=r"$h_1'/\lambda_0$")
        case _:
            raise ValueError("Unrecognised w_quantity. Must be one of 'pitch', 'box_width' or 'wavelength'.")
    # ax.set_aspect('equal', adjustable='box')
    ax.set_xlim([w_min, w_max])
    ax.set_ylim([h1_min, h1_max])

    colorbar_label = rf"FOM"
    max_colour_scale = np.maximum(np.abs(np.min(FOM_data)), np.abs(np.max(FOM_data)))
    max_colour_scale = np.round(max_colour_scale, 1)
    pcolormesh_kwargs = {"cmap": 'bwr', "vmin": -max_colour_scale, "vmax": max_colour_scale}

    FOM_cplot = ax.pcolormesh(ws, grating_depths, FOM_data, 
                                    shading='nearest', **pcolormesh_kwargs)
    fig.colorbar(FOM_cplot, label=colorbar_label)

    return fig, ax