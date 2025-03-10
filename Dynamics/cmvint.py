"""
cmvint - comoving integrator

A module containing the comoving integrator function. Similar syntax to scipy's 
odeint.
"""

import numpy as np

import time

from specrel import vadd, SinCosEpsilon, Lorentz, SinCosTheta

class InterpolateError(ValueError):
    pass


def create_coordinate_arrays():
    x_array = []
    y_array = []
    vx_array = []
    vy_array = []

    timeM_array = []
    tau_array = []
    timeL_array = []

    # Angles are stored in frame M
    phi_array = []
    omega_array = []

    # Frame Wigner-rotation angle
    eps_array = []
    eps_rate_array = []

    # Aberration angle
    theta_array = []

    # Accelerations
    accels = []

    return [x_array, y_array, vx_array, vy_array, 
            timeM_array, tau_array, timeL_array, 
            phi_array, omega_array, eps_array, eps_rate_array, theta_array,
            accels]

def append_coordinate_arrays(coordinate_arrays: list[list], items: list[float]):
    """Append items to coordinate arrays element by element"""
    for item, arr in zip(items,coordinate_arrays):
        arr.append(item)
    return coordinate_arrays

def _func(func, tn, yn, vL, i, args):  
    if args == ():
        return func(tn,yn,vL,i)
    else:
        return func(tn,yn,vL,i,args)

def Mstep(func: callable, h: float, tn: float, yn: np.ndarray, vL: np.ndarray, i: int, args: tuple=()):
    """
    Step in the direction of the state-vector derivative, func, in frame Mn using fourth-order Runge-Kutta.
    """
    k1 = h*_func(func, tn      , yn       , vL, i, args)
    k2 = h*_func(func, tn+0.5*h, yn+0.5*k1, vL, i, args)
    k3 = h*_func(func, tn+0.5*h, yn+0.5*k2, vL, i, args)
    k4 = h*_func(func, tn+h    , yn+k3    , vL, i, args)
    yNew = yn + 1/6*(k1 + 2*k2 + 2*k3 + k4)
    tNew = tn + h
    ydotNew = k1/h
    return tNew,yNew,ydotNew

def update_frame_L(timeMNew, YNew: np.ndarray, vn: np.ndarray):
    """
    Convert YNew (M-frame state vector) to frame L using inverse Lorentz transformation 
    with new velocity vn.
    """
    xNew, yNew, phiNew, uxNew, uyNew, omegaNew = YNew
    zNew  = np.array([timeMNew,xNew,yNew]) 
    uNew  = np.array([uxNew,uyNew])  # Velocity induced by the M-frame forces over small time step
    zLNew = Lorentz(-vn,zNew)
    return zLNew, uNew

def create_frame_MNext(zLNew: np.ndarray, uNew: np.ndarray, vn: np.ndarray):
    # Defining new M_{n+1} frame as a boost from L
    vLNew   = vadd(vn,uNew)  # Uses relativistic velocity addition
    zM_NEXT = Lorentz(vLNew,zLNew)  # Won't be at the origin anymore due to forces
    vM_NEXT = np.array([0,0])  # New velocity is 0 since we've just boosted into the new frame 
    thetan  = SinCosTheta(vLNew)[2]
    epsn    = SinCosEpsilon(vn,uNew)[2]  # Wigner rotation angle
    return vLNew, zM_NEXT, vM_NEXT, thetan, epsn

def update_frame_M_state(zM_NEXT: np.ndarray, vM_NEXT: np.ndarray, phiNew: float, omegaNew: float, epsn: float, eps_raten: float):
    # Updating state vector and M-frame velocity
    timeMn, xM2, yM2 = zM_NEXT
    phiM2            = phiNew - epsn  
    vxM2, vyM2       = vM_NEXT
    omegaM2          = omegaNew - eps_raten
    YMn = np.array([xM2,yM2,phiM2,vxM2,vyM2,omegaM2])
    return timeMn, YMn

def odecmvint(func, state0, t_max, v_max, args: tuple=(), hstep: float=1e-4):
    """
    Integrate func over time by passing between the comoving reference frame (frame M) and
    the light source reference frame (frame L).

    Parameters
    ----------
    func   :   callable(t,y,vL,i,args). Computes the derivative of M-frame-state-vector, y, at time t
                measured in the comoving M-frame.
                vL is the velocity of the object in frame L, needed for Lorentz transformation. 
                The integration step (i) is used for debugging.
    state0 :   Array of initial conditions for the state vector ([x, y, phi, vx, vy, vphi])
    t_max  :   Maximum integration runtime (seconds)
    v_max  :   Maximum object velocity before stopping integration (metres/second)
    args   :   Extra arguments to pass to func
    hstep  :   Integration step size
    
    Returns
    -------
    positions :   Array containing translational coordinates [x,y,vx,vy], each coordinate an array 
                  measured in frame L
    angles    :   Array containing rotational coordinates [phi, wigner eps, dphi/dt, dwignereps/dt], 
                  each coordinate an array measured in frame M
    times     :   Array containing times [t_M, tau, t_L], each an array of times measured in certain 
                  frames (M, centre-of-mass frame, L)
    loop_data :   Dictionary with keywords 
                    "Runtime" - float, total integration time (seconds)
                    "Steps" - int, total number of steps
                    "Stopped" - bool, if the integrator ran out of time or encountered a force-calculation 
                    error
    """
    
    # Flag for if the optimisation took too long or an integration step failed
    STOPPED = False

    coordinate_arrays = create_coordinate_arrays()
    x_array, y_array, vx_array, vy_array, timeM_array, tau_array, timeL_array,\
    phi_array, omega_array, eps_array, eps_rate_array, theta_array,\
    accels = coordinate_arrays

    timeLn = 0.  # starting time
    x0, y0, phi0, vx0, vy0, omega0 = state0
    vn = np.array([vx0, vy0])

    # Initial four-position in frame L (without the third spatial component)
    z0 = np.array([timeLn, x0, y0]) 

    # Initial four-position in frame M
    zM0 = Lorentz(vn,z0)
    timeMn, x0M, y0M = zM0

    # Initial state vectors
    YMn = np.array([x0M, y0M, phi0, 0, 0, omega0])            
    taun = 0

    # At non-relativistic initial velocities, the correction angles are negligible, set to 0
    epsn = 0.
    eps_raten = 0.
    thetan = 0.  
    
    _, _, _, ax0, ay0, aphi0 = _func(func, timeMn, YMn, vn, 0, args)
    acceln = [ax0,ay0,aphi0]

    coordinates = [x0, y0, vx0, vy0, timeMn, taun, timeLn, phi0, omega0, epsn, eps_raten, thetan, acceln]
    coordinate_arrays = append_coordinate_arrays(coordinate_arrays, coordinates)

    timeSTART = time.time()
    
    i = 1
    while (vn[0] < v_max):
        """
        The main loop of the comoving integrator. Information is passed between frames L, Mn and Mn+1.

        The loop:
            Object starts in M_n and accelerates into state n+1 via M-frame forces. This updates position, angle and velocities.
            
            The n+1 information is sent back to L, then into a new frame M_{n+1} defined by relativistic addition of the M_n velocity 
            (in L) and the new small velocity due to M-frame forces (over the time step Delta tau).
        
            Rotation information is transferred from M_n to M_{n+1} purely via Wigner rotation.
        """
        
        timeDIFF = time.time() - timeSTART
        
        if timeDIFF >= t_max:
            STOPPED = True
            print("Integration time limit reached.")
        
        if STOPPED:
            break    
        else:                                  
            try:
                timeMNew, YNew, YdotNew = Mstep(func,hstep,timeMn,YMn,vn,i,args) 
            except InterpolateError as ie:
                STOPPED = True
                print("\n" + str(ie) + "odecmvint: M-frame step failed. Successfully stopped early.")
                break
            
            phiNew = YNew[2]
            omegaNew = YNew[5]
            zLNew, uNew = update_frame_L(timeMNew, YNew, vn)
            vLNew, zM_NEXT, vM_NEXT, thetan, epsn = create_frame_MNext(zLNew, uNew, vn)
            if i==1:
                eps_raten = (epsn - 0)/hstep
            else:    
                eps_raten = (epsn - eps_array[i-2])/(2*hstep)
            timeMn, YMn = update_frame_M_state(zM_NEXT, vM_NEXT, phiNew, omegaNew, epsn, eps_raten)
            
            phiM2 = YMn[2]
            omegaM2 = YMn[5]

            vn = vLNew
            taun = taun + hstep

            acceln = [*YdotNew[3:]]
            coordinates = [zLNew[1], zLNew[2], *vLNew, 
                            timeMn, taun, zLNew[0], phiM2, omegaM2, epsn, eps_raten, thetan, acceln]
            append_coordinate_arrays(coordinate_arrays, coordinates)
        
        i += 1

    positions = np.array([x_array, y_array, vx_array, vy_array])  # Frame L
    angles    = np.array([phi_array, eps_array, omega_array, eps_rate_array, theta_array])  # Frame M
    times     = np.array([timeM_array,tau_array,timeL_array])
    accels    = np.array(accels)
    loop_data = {'Runtime': round(timeDIFF), 'Steps': i, 'Stopped': STOPPED}
    
    return positions, angles, times, accels, loop_data