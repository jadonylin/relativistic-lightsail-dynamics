import autograd
from autograd import numpy as np
from autograd.scipy.special import erf as autograd_erf
import scipy

#### All velocities are in m/s (NOT normalised by c)
c = 299792458

def norm_squared(v):
    """
    ## Inputs
    v: [vx, vy] 2D array
    ## Output
    """
    return v[0]**2 + v[1]**2

def Gamma(v):
    """
    ## Inputs
    v: [vx, vy] 2D array
    ## Output
    Lorentz gamma factor gamma(v)
    """
    return (1 - norm_squared(v)/c**2)**(-1/2)

def Dv(v):
    """
    ## Inputs
    v: [vx, vy] 2D array
    ## Output
    Doppler factor gamma(v)(1- vx/c)
    """
    return Gamma(v)*(1 - v[0]/c)

def vadd(v,u):
    """
    ## Inputs
    v: [vx, vy] 2D array
    u: [vx, vy] 2D array
    ## Output
    v + u - SR addition of velocities
    """
    g = Gamma(v)
    return (1/(1+np.dot(v/c,u/c))) * ( v + u/g + ( (g/(g+1)) * np.dot(v/c,u/c) * v ) )

def Lorentz(v,z):
    """
    ## Inputs
    v: [vx, vy] 2D array
    ## Output
    returns ([t',x',y'])
    """
    t = z[0]; x = z[1]; y = z[2]
    g = Gamma(v)
    f = g**2/(g+1)
    
    t2 = g * ( t - (v[0]*x)/c**2 - (v[1]*y)/c**2 )
    x2 = -(g*v[0]*t) + (1+f*(v[0]/c)**2) * x + f*((v[0]/c)*(v[1]/c)) * y
    y2 = -(g*v[1]*t) + f*((v[0]/c)*(v[1]/c)) * x + (1+f*(v[1]/c)**2) * y

    return np.array([t2,x2,y2])

def SinCosTheta(v):
    """
    ## Inputs
    v: [vx, vy] 2D array
    ## Output
    sin(theta'), cos(theta') - relativistic aberration
    """
    D = Dv(v)
    g = Gamma(v)
    sin = (1/D)*( -g*v[1]/c + (g**2/(g+1))*(v[0]*v[1])/c**2 )
    cos = (1/D)*( -g*v[0]/c + 1 + (g**2/(g+1))*(v[0]/c)**2 )
    theta = np.arcsin(sin)
    return sin, cos, theta

def SinCosEpsilon(v,u):
    """
    ## Inputs
    v: [vx, vy] 2D array
    u: [ux, uy] 2D array
    ## Output
    sin(eps), cos(eps), eps - rotation angle from M to M+1
    """
    gv = Gamma(v)
    gu = Gamma(u)
    g = gv * gu * (1 + np.dot(v/c,u/c) )
    cross = ( (u[0]/c) * (v[1]/c) - (u[1]/c) * (v[0]/c) )
    sin = cross * (gv*gu*(1+g+gv+gu)) / ( (1+g)*(1+gv)*(1+gu) )
    cos = (1+g+gv+gu)**2 /( (1+g)*(1+gv)*(1+gu) ) - 1
    eps = np.arcsin(sin)
    return sin, cos, eps

# def EpsRateTest(v,u,h):

def ABSC(v,phi):
    """
    ## Inputs
    v: [vx, vy] 2D array \n
    phi': grating angle
    ## Output
    A,B,S,C - sin(theta'), cos(theta'), S,C - linear corrections
    """
    sintheta = SinCosTheta(v)[0]
    costheta = SinCosTheta(v)[1]
    cos = np.cos(phi)
    sin = np.sin(phi)
    g = Gamma(v)
    D = Dv(v)
    bx = v[0] / c
    by = v[1] / c
    dot = (bx*cos +g*by*sin)
    dot2 = (by*cos +g*bx*sin)
    
    A = sintheta * cos/(g*D) - sin/D - (
        sintheta*dot) + dot2/(D*(g+1))- (
        g*by*dot/D ) + (
        ((g**2*(g+2))/(D*(g+1)**2))*bx*by*dot  )
    
    B = -costheta*dot + (2*bx*cos)/(D*(g+1)) - (
        (g*bx*dot)/D + (g*by**2 * cos)/(D**2 *(g+1)) ) + (
        ( (g**2 *(g+2))/(D*(g+1)**2) ) * bx**2 * dot )
    S = cos * A - sin * B
    C = sin * A + cos * B
    return A, B, S, C

def E_eps(v, phi):
    """
    ## Inputs
    v: [vx, vy] 2D array \n
    phi': grating angle
    ## Output
    $\mathcal(E)$ - epsilon linear correction
    """
    g = Gamma(v)
    return (g/(g+1)) * ( np.sin(phi)*v[0]/c - np.cos(phi)*v[1]/c )

def erf(x):
    return autograd_erf(x)

## Parameters
I0 = 0.5 * 10**9      # intensity 
L = 10                # grating width (10 m^2)
m = 1/1000            # mass (1g)
c = 299792458

def Parameters():
    return I0, L, m, c

## Optimised grating
def gaussian_width(grating_type):
    if grating_type=="Ilic":
        return 2 * L
    if grating_type=="Second":
        return 33.916288616522735
