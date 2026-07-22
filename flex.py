"""
A module to store Qpr functions with thermal and elastic coupling.
"""
import materials

def permittivity_scaled(grating, e, strain: float=0., temp: float=0., material: dict=materials.Si3N4) -> float:
    """
    Scale a given permittivity with material-dependent change in real and imaginary 
    refractive index due to strain and temperature change.

    Parameters
    ----------
    grating  :   The TwoBox grating object.
    e        :   Permittivity of the material.
    strain   :   strain factor for elongation
    temp     :   Temperature relative to rest temperature in Kelvin
    material :   Material properties dictionary

    Returns
    -------
    permittivity : scaled permittivity
    """
    n = grating.npa.sqrt(e) + 0j  # ensure n is complex to make PyTorch happy
    nr = grating.npa.real(n)
    ni = grating.npa.imag(n)
    dnrdT = material["thermorefract"]
    dnidT = material["thermoextinct"]
    nr_scaled = nr*(1 + dnrdT*temp)
    ni_scaled = ni*(1 + dnidT*temp)
    n_scaled = nr_scaled + 1j*ni_scaled
    return n_scaled**2

def grating_scaler(grating, strain: float=0., temp: float=0., material: dict=materials.Si3N4) -> float:
    """
    Scale the grating due to longitudinal-wave strain and temperature change.

    Parameters
    ----------
    grating  :   The TwoBox grating object.
    strain   :   strain factor for elongation
    temp     :   Temperature relative to rest temperature in Kelvin
    material :   Grating material

    Returns
    -------
    scaled_grating : grating with physical properties scaled
    """

    # Store original parameters
    p = grating.grating_pitch
    h = grating.grating_depth
    w1 = grating.box1_width 
    w2 = grating.box2_width 
    bcd = grating.box_centre_dist 
    hs = grating.substrate_depth
    eb1 = grating.box1_eps
    eb2 = grating.box2_eps
    es = grating.substrate_eps

    nu = material["Poisson"]
    alpha = material["thermal_expansion"]

    scale_from_strain = 1 + strain
    scale_from_temp = 1 + alpha*temp
    grating.grating_pitch = p*scale_from_strain*scale_from_temp
    grating.box1_width = w1*scale_from_strain*scale_from_temp
    grating.box2_width = w2*scale_from_strain*scale_from_temp
    grating.box_centre_dist = bcd*scale_from_strain*scale_from_temp
    
    zscale = 1 - nu*strain + (1+nu)*alpha*temp
    grating.grating_depth = h*zscale
    grating.substrate_depth = hs*zscale

    grating.box1_eps = permittivity_scaled(grating, eb1, strain, temp, material)
    grating.box2_eps = permittivity_scaled(grating, eb2, strain, temp, material)
    grating.substrate_eps = permittivity_scaled(grating, es, strain, temp, material)

    if strain != 0. or temp != 0.:
        raise ValueError("Error: input grating parameters have been modified due to request for non-zero strain or temperature.")
    return grating

def jacobian_maker(func, grating, strain: float=0., temp: float=0., material: dict=materials.Si3N4) -> float:
    """
    Generates the jacobian for func with respect to strain and temperature-change variables.

    Parameters
    ----------
    func     :   Function to calculate the jacobian for, with inputs (grating, strain, temp, material)
    grating  :   The TwoBox grating object.
    strain   :   strain factor for elongation
    temp     :   Temperature relative to rest temperature in Kelvin
    material :   Material properties dictionary

    Returns
    -------
    |∂func/∂ϵ, ∂func/∂θ|
    |∂func/∂ϵ, ∂func/∂θ|
    """
    strain = grating.npa.array(strain)
    temp = grating.npa.array(temp)
    def _func(p):
        strain, temp = p
        return func(grating, strain, temp, material)
    p = grating.npa.array([strain, temp])
    return grating.npa.array(grating.npa.jacobian(_func)(p).squeeze())


def Qpr(grating, strain: float=0., temp: float=0., material: dict=materials.Si3N4) -> float:
    """
    Calculate the radiation pressure efficiency Qprj as a function of strain
    parallel to the grating and temperature. Temperature changes are assumed
    to linearly expand the grating in the directions parallel and perpendicular
    to the grating.

    Parameters
    ----------
    grating  :   The TwoBox grating object.
    strain   :   strain factor for elongation
    temp     :   Temperature relative to rest temperature in Kelvin
    material :   Material properties dictionary

    Returns
    -------
    [Qpr1, Qpr2] : Radiation pressure efficiency factors
    """
    _grating = grating_scaler(grating, strain, temp, material)
    return _grating.Q()

def Qprj(grating, j: int=2, strain: float=0., material: dict=materials.Si3N4) -> float:
    """
    Calculate the radiation pressure efficiency Qprj as a function of elongation
    parallel to the grating.

    Parameters
    ----------
    grating  :   The TwoBox grating object.
    j        :   Index of the radiation pressure efficiency to calculate (1 for Qpr1, 2 for Qpr2).
    strain   :   strain factor for elongation
    material :   Material properties dictionary

    Returns
    -------
    Qprj : float
    """

    if j not in [1, 2]:
        raise ValueError(f"Invalid value for j: {j}. Must be 1 or 2.")
    return Qpr(grating, strain, 0., material)[j-1]

def dQpr(grating, strain: float=0., temp: float=0., material: dict=materials.Si3N4) -> float:
    """
    Calculate the derivative of the radiation pressure efficiency Qpr1 and Qpr2
    with respect to strain and temperature change.

    Parameters
    ----------
    grating  :   The TwoBox grating object.
    strain   :   strain factor for elongation
    temp     :   Temperature relative to rest temperature in Kelvin
    material :   Material properties dictionary

    Returns
    -------
    |∂Qpr1/∂ϵ, ∂Qpr1/∂θ|
    |∂Qpr2/∂ϵ, ∂Qpr2/∂θ|
    """
    return jacobian_maker(Qpr, grating, strain, temp, material)

def dQpr_dstrain(grating, strain: float=0., material: dict=materials.Si3N4) -> float:
    """
    Calculate the derivative of the radiation pressure efficiency Qprj 
    with respect to strain parallel to the grating.

    Parameters
    ----------
    grating     :   The TwoBox grating object.
    strain      :   strain factor for elongation
    material    :   Material properties dictionary

    Returns
    -------
    [dQpr1_dstrain, dQpr2_dstrain] : strain derivatives of radiation pressure efficiency factors
    """
    return dQpr(grating, strain, 0., material)[:,0]

def dQprj_dstrain(grating, j: int=2, strain: float=0., grad_method: str="finite", material: dict=materials.Si3N4) -> float:
    """
    Calculate the derivative of the radiation pressure efficiency Qprj 
    with respect to strain parallel to the grating.

    Parameters
    ----------
    grating     :   The TwoBox grating object.
    j           :   Index of the radiation pressure efficiency to calculate (1 for Qpr1, 2 for Qpr2).
    strain      :   strain factor for elongation
    grad_method :   Method for gradient calculation.
    material    :   Material properties dictionary

    Returns
    -------
    dQprj_dstrain : float
    """
    if j not in [1, 2]:
        raise ValueError(f"Invalid value for j: {j}. Must be 1 or 2.")
    if grad_method == "finite":
        h = 1e-6
        Qprj_plus = Qprj(grating, j, strain + h, material)
        Qprj_minus = Qprj(grating, j, strain - h, material)
        return (Qprj_plus - Qprj_minus)/(2*h)
    elif grad_method == "grad":
        return dQpr_dstrain(grating, strain, material)[j-1]
    else:
        raise ValueError(f"Unknown grad_method: {grad_method}")

def d2Qprj_dstrain2(grating, j: int=2, strain: float=0., grad_method: str="finite", material: dict=materials.Si3N4) -> float:
    """
    Calculate the second derivative of the radiation pressure efficiency Qprj 
    with respect to strain parallel to the grating.
    TODO: NOT UPDATED

    Parameters
    ----------
    grating     :   The TwoBox grating object.
    j           :   Index of the radiation pressure efficiency to calculate (1 for Qpr1, 2 for Qpr2).
    strain      :   strain factor for elongation
    grad_method :   Method for gradient calculation.
    material    :   Material properties dictionary

    Returns
    -------
    dQprj_dstrain : float
    """
    if j not in [1, 2]:
        raise ValueError(f"Invalid value for j: {j}. Must be 1 or 2.")
    if grad_method == "finite":
        h = 1e-6
        # Qprj_plus = dQprj_dstrain(grating, j, strain + h, grad_method="grad")
        # Qprj_minus = dQprj_dstrain(grating, j, strain - h, grad_method="grad")
        Qprj_plus = dQprj_dstrain(grating, j, strain + h, grad_method="finite", material=material)
        Qprj_minus = dQprj_dstrain(grating, j, strain - h, grad_method="finite", material=material)
        return (Qprj_plus - Qprj_minus)/(2*h)
    elif grad_method == "grad":
        strain = grating.npa.array(strain)
        func = lambda s: dQprj_dstrain(grating, j, s, grad_method="grad", material=material)
        grad_func = grating.npa.grad(func)
        return grad_func(strain)
    else:
        raise ValueError(f"Unknown grad_method: {grad_method}")
    

def absorption(grating, strain: float=0., temp: float=0., material: dict=materials.Si3N4) -> float:
    """
    Calculate the absorption (1-R-T) as a function of elongation
    parallel to the grating.

    Parameters
    ----------
    grating  :   The TwoBox grating object.
    strain   :   strain factor for elongation
    temp     :   Temperature relative to rest temperature in Kelvin
    material :   Material properties dictionary

    Returns
    -------
    absorption : absorption of the grating
    """
    _grating = grating_scaler(grating, strain, temp, material)
    Rs, Ts = _grating.eff()
    absorption = 1 - _grating.npa.sum(Rs + Ts)
    return absorption

def dabsorption(grating, strain: float=0., temp: float=0., material: dict=materials.Si3N4) -> float:
    """
    Calculate the derivative of the absorption with respect to strain and temperature change.

    Parameters
    ----------
    grating  :   The TwoBox grating object.
    strain   :   strain factor for elongation
    temp     :   Temperature relative to rest temperature in Kelvin
    material :   Material properties dictionary

    Returns
    -------
    [∂a/∂ϵ, ∂a/∂θ]
    """
    return jacobian_maker(absorption, grating, strain, temp, material)