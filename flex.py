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

    grating.grating_pitch = p * scale
    grating.box1_width = w1 * scale
    grating.box2_width = w2 * scale
    grating.box_centre_dist = bcd * scale
    Qpr2 = grating.Q()[1]

    # TODO: Is there a better way to use the grating object without modifying the original?
    # Restore original parameters
    grating.grating_pitch = p
    grating.box1_width = w1
    grating.box2_width = w2
    grating.box_centre_dist = bcd
    return Qpr2

def dQpr2_dscale(grating: TwoBox, scale: float=1.0, grad_method: str="finite") -> float:
    """
    Calculate the derivative of the radiation pressure efficiency Qpr2 
    with respect to elongation parallel to the grating.

    Parameters
    ----------
    grating     :   The TwoBox grating object.
    scale       :   Scale factor for elongation, by default 1.0.
    grad_method :   Method for gradient calculation.

    Returns
    -------
    dQpr2_dscale : float
    """
    if grad_method == "finite":
        h = 1e-6
        Qpr2_plus = Qpr2_elongated(grating, scale + h)
        Qpr2_minus = Qpr2_elongated(grating, scale - h)
        return (Qpr2_plus - Qpr2_minus)/(2*h)
    elif grad_method == "grad":
        scale = grating.npa.array(scale)
        grad_func = grating.npa.grad(Qpr2_elongated, argnum=1)
        return grad_func(grating, scale)
    else:
        raise ValueError(f"Unknown grad_method: {grad_method}")