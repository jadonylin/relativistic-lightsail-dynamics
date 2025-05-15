"""
cmvint - comoving integrator

A module containing the comoving integrator function. Similar syntax to scipy's 
odeint.

TODO: I'm still not sure if finite difference is valid for wigner rotation derivative calculations.
      Also, using finite differences requires saving previous 2 elements of the wigner array, which
      further requires hard-coding the wigner index into saving methods.
"""

import numpy as np

import dill as pickle

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
    ax_array = []
    ay_array = []
    aphi_array = []

    coordinate_arrays = [x_array, y_array, vx_array, vy_array, 
                         timeM_array, tau_array, timeL_array, 
                         phi_array, omega_array, eps_array, eps_rate_array, theta_array,
                         ax_array, ay_array, aphi_array]

    return coordinate_arrays

def append_coordinate_arrays(coordinate_arrays: list[list], items: list[float]):
    """Append items to coordinate arrays element by element"""
    for item, arr in zip(items,coordinate_arrays):
        arr.append(item)
    return coordinate_arrays

def store_coordinate_arrays(coordinate_arrays: list[list], filename: str, final_save: bool):
    """Store coordinate arrays in a pre-existing file"""
    wigner_idx = 9  # index of wigner rotation angle list in coordinate_arrays list
    with open(filename, "ab+") as storage:
        arr_to_store = coordinate_arrays[:]
        for idx, arr in enumerate(arr_to_store):
            if final_save:  # TODO: can we avoid nested if statements?
                break
            else:
                # TODO: need wigner array to have the same length as other arrays. To fix,
                #       store up to [:-1], but don't discard the -2 element.
                if idx == wigner_idx or idx == wigner_idx+1:
                    arr_to_store[idx] = arr[:-2]
                else:
                    arr_to_store[idx] = arr[:-1]
        pickle.dump(arr_to_store, storage)
    return coordinate_arrays

def clear_coordinate_arrays(coordinate_arrays: list[list]):
    """Clear all entries except for 1 (possibly more) in coordinate arrays"""
    wigner_idx = 9  # index of wigner rotation angle list in coordinate_arrays list
    for idx, arr in enumerate(coordinate_arrays):
        if idx == wigner_idx or idx == wigner_idx+1:  # keep wigner derivative at same step as wigner angle
            coordinate_arrays[idx] = arr[-2:]  # keep the last 2 values for central difference
        else:
            coordinate_arrays[idx] = arr[-1:]
    return coordinate_arrays

def load_coordinate_arrays(filename: str):
    """Load all coordinate arrays. Useful to extract data from unfinished runs"""
    data = []
    with open(filename, 'rb') as storage:
        try:
            while True:
                data.append(pickle.load(storage))
        except EOFError:
            pass
    coordinate_arrays = data[0]
    num_coordinates = len(coordinate_arrays)
    for chunk in data[1:]:
        for n in range(num_coordinates):
            coordinate_arrays[n] += chunk[n]
    
    return coordinate_arrays

def reformat_coordinate_arrays(coordinate_arrays: list[list]):
    """Reformat coordinate arrays into a physically meaningful list of np arrays"""
    x_array, y_array, vx_array, vy_array, timeM_array, tau_array, timeL_array,\
    phi_array, omega_array, eps_array, eps_rate_array, theta_array,\
    ax_array, ay_array, aphi_array = coordinate_arrays

    positions = np.array([x_array, y_array, vx_array, vy_array])  # Frame L
    angles    = np.array([phi_array, eps_array, omega_array, eps_rate_array, theta_array])  # Frame M
    times     = np.array([timeM_array,tau_array,timeL_array])
    accels    = np.array([ax_array, ay_array, aphi_array])
    
    return positions, angles, times, accels


def _func(func, tn, yn, vL, i, args):  
    if args == ():
        return func(tn,yn,vL,i)
    else:
        return func(tn,yn,vL,i,*args)

def Mstep(func: callable, h: float, tn: float, yn: np.ndarray, vL: np.ndarray, i: int, args: tuple=()):
    """
    Step in the direction of the state-vector derivative, func, in frame Mn using fourth-order Runge-Kutta.
    """
    k1 = h*_func(func, tn    , yn     , vL, i, args)
    k2 = h*_func(func, tn+h/2, yn+k1/2, vL, i, args)
    k3 = h*_func(func, tn+h/2, yn+k2/2, vL, i, args)
    k4 = h*_func(func, tn+h  , yn+k3  , vL, i, args)
    yNew = yn + 1/6*(k1 + 2*k2 + 2*k3 + k4)
    tNew = tn + h
    ydotNew = k1/h
    return tNew,yNew,ydotNew


def update_frame_L(timeMNew: float, YNew: np.ndarray, vn: np.ndarray):
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
    """Create M frame at the next step (M_{n+1}) via a boost from frame L"""
    vLNew   = vadd(vn,uNew)  # Uses relativistic velocity addition
    zM_NEXT = Lorentz(vLNew,zLNew)  # Won't be at the origin anymore due to forces
    vM_NEXT = np.array([0,0])  # New velocity is 0 since we've just boosted into the new frame 
    thetan  = SinCosTheta(vLNew)[2]
    epsn    = SinCosEpsilon(vn,uNew)[2]  # Wigner rotation angle
    return vLNew, zM_NEXT, vM_NEXT, thetan, epsn

def update_frame_M_state(zM_NEXT: np.ndarray, vM_NEXT: np.ndarray, phiNew: float, omegaNew: float, epsn: float, eps_raten: float):
    """Update M-frame state vector including Wigner rotation angles"""
    timeMn, xM2, yM2 = zM_NEXT
    phiM2            = phiNew - epsn  
    vxM2, vyM2       = vM_NEXT
    omegaM2          = omegaNew - eps_raten
    YMn = np.array([xM2,yM2,phiM2,vxM2,vyM2,omegaM2])
    return timeMn, YMn

def update_finite_difference(current_value: float, all_values: list, iteration: int, hstep: float):
    """Calculate finite difference on an updating array"""
    # Still need to pass iteration to this function so that the first-iteration derivative 
    # is calculated as an edge case separately from the central differences.
    if iteration == 1:
        derivative = (current_value - 0)/hstep
    else:
        # TODO: epsn at time 0 is the same for any step size (for a given set of initial 
        #       conditions), hence the derivative is inversely proportional to the step size. 
        #       This means the wigner rotation and wigner time derivative are not consistent 
        #       across different step sizes. We could try to calculate the wigner time 
        #       derivative analytically, but the expression contains velocities in frame L, 
        #       which would also require finite differences to calculate.
        # TODO: since all_values is updating after every save, I've hard-coded the index to
        #       -2 (two spaces back from the current index). Is there a way to do it 
        #       without hard coding?
        # TODO: When I compare this derivative with np.gradient, I get consistent results.
        #       This indicates that we are calculating central differences (with the factor 2 
        #       in the denominator). However, since we are calculating the derivative at 
        #       "iteration", this looks more like backward difference, which shouldn't have 
        #       the factor 2 in the denominator. Which one is correct?
        derivative = (current_value - all_values[-2])/(2*hstep)
    return derivative



def odecmvint(func: callable, state0: np.ndarray, t0: float, t_max: float, v_max: float, args: tuple=(), hstep: float=1e-4, 
              save_idx: int=1000, save_file: str="odecmvint"):
    """
    Integrate func over time by passing between the comoving reference frame (frame M) and
    the light source reference frame (frame L).

    Parameters
    ----------
    func      :   callable(t,y,vL,i,args). Computes the derivative of M-frame-state-vector, y, at time t
                  measured in the comoving M-frame.
                  vL is the velocity of the object in frame L, needed for Lorentz transformation. 
                  The integration step, i, is used for debugging.
    state0    :   Initial conditions for the state vector ([x, y, phi, vx, vy, vphi])
    t0        :   Initial time in frame L (seconds), corresponds to the initial condition. 
                  NOTE: Must be nonzero when your initial condition comes from partway through some other 
                        dynamics run.
    t_max     :   Maximum integration runtime (seconds)
    v_max     :   Maximum object velocity before stopping integration (metres/second)
    args      :   Extra arguments to pass to func
    hstep     :   Integration step size
    save_idx  :   Save the data after every save_idx loops. This is more memory efficient and safer
                 for long integration times to avoid losing data. Must be larger than 3.
    save_file :   Filename to save the data to during runtime. The filetype is added automatically.
    
    Returns
    -------
    positions :   Array containing translational coordinates [x,y,vx,vy], each coordinate an array 
                  measured in frame L
    angles    :   Array containing rotational coordinates [phi, wigner eps, dphi/dt, dwignereps/dt], 
                  each coordinate an array measured in frame M
    times     :   Array containing times [t_M, tau, t_L], each an array of times measured in certain 
                  frames (M, centre-of-mass frame, L)
    accels    :   Array containing accelerations [ax, ay, aphi], each an array of accelerations 
                  measured in frame M
    loop_data :   Dictionary with keywords 
                    "Runtime" - float, total integration time (seconds)
                    "Steps" - int, total number of steps
                    "Stopped" - bool, if the integrator ran out of time or encountered a force-
                    calculation error
    """
    
    # Flag for if the optimisation took too long or an integration step failed
    STOPPED = False

    fname = save_file + "_temp.pkl"
    coordinate_arrays = create_coordinate_arrays()
    with open(fname, "wb") as storage:  # needed to overwrite temp file if it exists
        pickle.dump(coordinate_arrays, storage)
    wigner_idx = 9  # index of wigner rotation angle list in coordinate_arrays list

    timeLn = t0  # starting time
    x0, y0, phi0, vx0, vy0, omega0 = state0
    vn = np.array([vx0, vy0])

    # Initial four-position in frame L (without the third spatial component)
    z0 = np.array([timeLn, x0, y0]) 

    # Initial four-position in frame M
    zM0 = Lorentz(vn,z0)
    timeMn, x0M, y0M = zM0

    # Initial state vector
    YMn = np.array([x0M, y0M, phi0, 0., 0., omega0])            
    taun = 0.  # Simulation time/proper time of the comoving sail

    # At non-relativistic initial velocities, the correction angles are negligible, set to 0
    epsn = 0.
    eps_raten = 0.
    thetan = 0.  
    
    i = 0  # Integrator iteration index
    _, _, _, ax0, ay0, aphi0 = _func(func, timeMn, YMn, vn, i, args)

    coordinates = [x0, y0, vx0, vy0, timeMn, taun, timeLn, phi0, omega0, epsn, eps_raten, thetan, ax0,ay0,aphi0]
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
                print("\n" + str(ie) + "\nodecmvint: M-frame step failed. Successfully stopped early.")
                break
            
            zLNew, uNew = update_frame_L(timeMNew, YNew, vn)
            vLNew, zM_NEXT, vM_NEXT, thetan, epsn = create_frame_MNext(zLNew, uNew, vn)
            
            phiNew = YNew[2]
            omegaNew = YNew[5]
            eps_raten = update_finite_difference(epsn, coordinate_arrays[wigner_idx], i, hstep)
            timeMn, YMn = update_frame_M_state(zM_NEXT, vM_NEXT, phiNew, omegaNew, epsn, eps_raten)

            vn = vLNew
            taun = taun + hstep

            # TODO: find a better way to store and return arrays that doesn't require explicitly extracting coordinates.
            #       This should also simplify the number of subfunction arguments.
            phiM2 = YMn[2]
            omegaM2 = YMn[5]
            coordinates = [zLNew[1], zLNew[2], *vLNew, 
                            timeMn, taun, zLNew[0], phiM2, omegaM2, epsn, eps_raten, thetan, *YdotNew[3:]]
            append_coordinate_arrays(coordinate_arrays, coordinates)
            
            if i % save_idx == 0:
                store_coordinate_arrays(coordinate_arrays, fname, final_save=False)
                clear_coordinate_arrays(coordinate_arrays)
        
        i += 1
    
    # TODO: handle edge case where final save happens immediately after previous save
    store_coordinate_arrays(coordinate_arrays, fname, final_save=True)  # Final save
    coordinate_arrays = load_coordinate_arrays(fname)
    positions, angles, times, accels = reformat_coordinate_arrays(coordinate_arrays)
    loop_data = {'Runtime': round(timeDIFF), 'Steps': i, 'Stopped': STOPPED}
    
    return positions, angles, times, accels, loop_data