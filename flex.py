"""
A module to store helpful functions and classes for plotting.

In particular, the PlotBox class contains methods for plotting the permittivity profile, 
angular efficiency, spectrum, and field distributions of a TwoBox grating.
"""

import torch
from twobox import TwoBox

def Qpr2_elongated(grating: TwoBox, scale: float=1.0) -> float:
    """
    Calculate the radiation pressure efficiency Qpr2 as a function of elongation
    parallel to the grating.

    Parameters
    ----------
    grating :   The TwoBox grating object.
    scale   :   Scale factor for elongation, by default 1.0.

    Returns
    -------
    Qpr2 : float
    """
    
    # Store original parameters
    p = grating.grating_pitch
    w1 = grating.box1_width 
    w2 = grating.box2_width 
    bcd = grating.box_centre_dist 

    scale = grating.npa.array(scale)
    grating.grating_pitch = grating.grating_pitch * scale
    grating.box1_width = grating.box1_width * scale
    grating.box2_width = grating.box2_width * scale
    grating.box_centre_dist = grating.box_centre_dist * scale
    Qpr2 = grating.Q()[1]

    # TODO: Is there a better way to use the grating object without modifying the original?
    # Restore original parameters
    grating.grating_pitch = p
    grating.box1_width = w1
    grating.box2_width = w2
    grating.box_centre_dist = bcd
    return Qpr2

def dQpr2_dscale(grating: TwoBox, scale: float=1.0) -> float:
    """
    Calculate the derivative of the radiation pressure efficiency Qpr2 
    with respect to elongation parallel to the grating.

    Parameters
    ----------
    grating :   The TwoBox grating object.
    scale   :   Scale factor for elongation, by default 1.0.

    Returns
    -------
    dQpr2_dscale : float
    """
    scale = grating.npa.array(scale)
    grad_func = grating.npa.grad(Qpr2_elongated, argnum=1)
    return grad_func(grating, scale)