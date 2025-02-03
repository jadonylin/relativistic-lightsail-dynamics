import numpy as np
I0=10**9            # intensity - might need to be halved
L=10                # grating width (10 m^2)
m=1/1000            # mass (1g)
c=299792458

def Parameters():
    return I0, L, m, c

def gamma_ND(v):
    """
    Calculate the Lorentz gamma factor with an input non-dimensionalised speed/velocity.

    Parameters
    ----------
    v     : ND speed or two/three-velocity or list of two/three-velocities

    Returns
    -------
    gamma : Lorentz gamma factor
    """
    if not isinstance(v,(list,np.ndarray)):
        v = [v]
    v = np.array(v)
    
    if any(isinstance(i, np.ndarray) for i in v):
        vnorm = np.linalg.norm(v,axis=1)
    else:
        vnorm = np.linalg.norm(v)
    
    gamma = 1/np.sqrt(1-np.power(vnorm,2))
    return gamma

def D1_ND(v):
    """
    Calculate the D_1 Doppler factor with an input non-dimensionalised velocity.

    Parameters
    ----------
    v  : ND two/three-velocity of the moving frame or list of two/three-velocities

    Returns
    -------
    D1 : Doppler factor
    """
    if not isinstance(v,(list,np.ndarray)):
        v = [v]
    v = np.array(v)
    
    if any(isinstance(i, np.ndarray) for i in v):
        vx = np.array([i[0] for i in v])
    else:
        vx = np.array(v[0])

    D1 = gamma_ND(v)*(1-vx)
    return D1


## Initial grating

# Ilic: 1.5, 1.8
# Starting at v=5.39%
wavelength=1.5 /D1_ND(0.0539)

grating_pitch=1.8 / wavelength
grating_depth=0.5 / wavelength
box1_width=0.15 * grating_pitch
box2_width=0.35 * grating_pitch
box_centre_dist=0.60 * grating_pitch
box1_eps = 3.5**2 
box2_eps = 3.5**2
gaussian_width=2 * L
substrate_depth=0.5 / wavelength
substrate_eps=1.45**2

grating_pitch= np.float64(grating_pitch)
grating_depth=np.float64(grating_depth)
box1_width=np.float64(box1_width)
box2_width=np.float64(box2_width)
box_centre_dist=np.float64(box_centre_dist)
box1_eps=np.float64(box1_eps)
box2_eps=np.float64(box2_eps)
gaussian_width=np.float64(gaussian_width)
substrate_depth=np.float64(substrate_depth)
substrate_eps=np.float64(substrate_eps)

def Initial_bigrating():
    return grating_pitch, grating_depth, box1_width, box2_width, box_centre_dist, box1_eps, box2_eps, gaussian_width, substrate_depth, substrate_eps


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
    v = np.linalg.norm(v)
    c = 3e8 # speed of light in m/s
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
    vx = v[0]
    c = 3e8 # speed of light in m/s
    D1 = gamma(v)*(1-vx/c)
    return D1

def LT(a,v):
    """
    Calculate the Lorentz boost of an input four-vector.

    Parameters
    ----------
    a       : Four-vector (e.g. array([5,5,0,0]) momentum vector)
    v       : Three-velocity of the moving frame 

    Returns
    -------
    a_prime : Four-vector a components measured in the moving frame
    """
    c = 3e8
    gam = gamma(v)
    vx, vy, vz = v
    vnorm = np.linalg.norm(v)

    if vnorm == 0:
        return a

    LT_mtx = np.array([[gam, -gam*vx/c, -gam*vy/c, -gam*vz/c], 
                         [-gam*vx/c, 1+(gam-1)*vx**2/vnorm**2, (gam-1)*vx*vy/vnorm**2, (gam-1)*vx*vz/vnorm**2], 
                         [-gam*vy/c, (gam-1)*vx*vy/vnorm**2, 1+(gam-1)*vy**2/vnorm**2, (gam-1)*vy*vz/vnorm**2], 
                         [-gam*vz/c, (gam-1)*vx*vz/vnorm**2, (gam-1)*vy*vz/vnorm**2, 1+(gam-1)*vz**2/vnorm**2]])    
    
    a_prime = np.matmul(LT_mtx,a)
    return a_prime