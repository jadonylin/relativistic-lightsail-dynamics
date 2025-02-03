def Parameters():
    I0=10**9            # intensity - might need to be halved
    L=10                # grating width (10 m^2)
    m=1/1000            # mass (1g)
    c=299792458
    return I0, L, m, c

I0, L, m, c = Parameters()

def gamma(v):
    """
    Calculate the Lorentz gamma factor with an input speed/velocity.

    Parameters
    ----------
    v     : Speed or two/three-velocity

    Returns
    -------
    gamma : Lorentz gamma factor
    """
    gamma = pow(1-v**2/c**2,-0.5)
    return gamma

def D1(v):
    """
    Calculate the D_1 Doppler factor with an input velocity.

    Parameters
    ----------
    v  : Two/three-velocity of the moving frame

    Returns
    -------
    D1 : Doppler factor
    """
    D1 = gamma(v)*(1-v/c)
    return D1
