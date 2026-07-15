"""
A module to store helpful functions and classes for plotting.

In particular, the PlotBox class contains methods for plotting the permittivity profile, 
angular efficiency, spectrum, and field distributions of a TwoBox grating.
"""

def Qpr_elongated(grating, scale: float=1.0) -> float:
    """
    Calculate the radiation pressure efficiency Qprj as a function of elongation
    parallel to the grating.

    Parameters
    ----------
    grating :   The TwoBox grating object.
    scale   :   Scale factor for elongation, by default 1.0.

    Returns
    -------
    [Qpr1, Qpr2] : Radiation pressure efficiency factors
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
    Qprs = grating.Q()

    # TODO: Is there a better way to use the grating object without modifying the original?
    # Restore original parameters
    grating.grating_pitch = p
    grating.box1_width = w1
    grating.box2_width = w2
    grating.box_centre_dist = bcd
    return Qprs

def dQpr_dscale(grating, scale: float=1.0) -> float:
    """
    Calculate the derivative of the radiation pressure efficiency Qprj 
    with respect to elongation parallel to the grating.

    Parameters
    ----------
    grating     :   The TwoBox grating object.
    scale       :   Scale factor for elongation, by default 1.0.

    Returns
    -------
    [dQpr1_dscale, dQpr2_dscale] : Scale derivatives of radiation pressure efficiency factors
    """
    scale = grating.npa.array(scale)
    grad_func = grating.npa.grad(Qpr_elongated, argnum=1)
    return grad_func(grating, scale)


def Qprj_elongated(grating, j: int=2, scale: float=1.0) -> float:
    """
    Calculate the radiation pressure efficiency Qprj as a function of elongation
    parallel to the grating.

    Parameters
    ----------
    grating :   The TwoBox grating object.
    j       :   Index of the radiation pressure efficiency to calculate (1 for Qpr1, 2 for Qpr2).
    scale   :   Scale factor for elongation, by default 1.0.

    Returns
    -------
    Qprj : float
    """

    if j not in [1, 2]:
        raise ValueError(f"Invalid value for j: {j}. Must be 1 or 2.")
    return Qpr_elongated(grating, scale)[j-1]

def dQprj_dscale(grating, j: int=2, scale: float=1.0, grad_method: str="finite") -> float:
    """
    Calculate the derivative of the radiation pressure efficiency Qprj 
    with respect to elongation parallel to the grating.

    Parameters
    ----------
    grating     :   The TwoBox grating object.
    j           :   Index of the radiation pressure efficiency to calculate (1 for Qpr1, 2 for Qpr2).
    scale       :   Scale factor for elongation, by default 1.0.
    grad_method :   Method for gradient calculation.

    Returns
    -------
    dQprj_dscale : float
    """
    if j not in [1, 2]:
        raise ValueError(f"Invalid value for j: {j}. Must be 1 or 2.")
    if grad_method == "finite":
        h = 1e-6
        Qprj_plus = Qprj_elongated(grating, j, scale + h)
        Qprj_minus = Qprj_elongated(grating, j, scale - h)
        return (Qprj_plus - Qprj_minus)/(2*h)
    elif grad_method == "grad":
        return dQpr_dscale(grating, scale)[j-1]
    else:
        raise ValueError(f"Unknown grad_method: {grad_method}")


def absorption(grating, scale: float=1.0) -> float:
    """
    Calculate the absorption (1-R-T) as a function of elongation
    parallel to the grating.

    Parameters
    ----------
    grating :   The TwoBox grating object.
    scale   :   Scale factor for elongation, by default 1.0.

    Returns
    -------
    absorption : absorption of the grating
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
    Rs, Ts = grating.eff()
    absorption = 1 - grating.npa.sum(Rs + Ts)

    # TODO: Is there a better way to use the grating object without modifying the original?
    # Restore original parameters
    grating.grating_pitch = p
    grating.box1_width = w1
    grating.box2_width = w2
    grating.box_centre_dist = bcd
    return absorption

def dabsorption_dscale(grating, scale: float=1.0) -> float:
    """
    Calculate the derivative of the absorption
    with respect to elongation parallel to the grating.

    Parameters
    ----------
    grating     :   The TwoBox grating object.
    scale       :   Scale factor for elongation, by default 1.0.

    Returns
    -------
    dabsorption_dscale : Scale derivative of absorption
    """
    scale = grating.npa.array(scale)
    grad_func = grating.npa.grad(absorption, argnum=1)
    return grad_func(grating, scale)