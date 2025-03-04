
"""
cmvint - comoving integrator

A module containing the comoving integrator function. Similar syntax to scipy's 
odeint.
"""

import numpy as np

import time

from specrel import vadd, SinCosEpsilon, Lorentz


def Mstep(func,h,tn,yn,vL,i,args=()):
    """
    Step in the direction of the state-vector derivative, func, in frame Mn using fourth-order Runge-Kutta.
    """

    k1 = h*func(tn      , yn       , vL, i)
    k2 = h*func(tn+0.5*h, yn+0.5*k1, vL, i)
    k3 = h*func(tn+0.5*h, yn+0.5*k2, vL, i)
    k4 = h*func(tn+h    , yn+k3    , vL, i)

    yNew = yn + 1/6*(k1 + 2*k2 + 2*k3 + k4)
    tNew = tn + h

    return tNew,yNew

def odecmvint(func, y0, t_max, v_max, args=(), hstep=1e-4):
    """
    Integrate func over time by passing between the comoving reference frame (frame M) and
    the light source reference frame (frame L).


    Parameters
    ----------
    func  :   callable(t,y,vL,i,args). Computes the derivative of M-frame-state-vector, y, at time t
              measured in the comoving M-frame.
              vL is the velocity of the object in frame L, needed for Lorentz transformation. 
              The integration step (i) is used for debugging.
    y0    :   Array of initial conditions on y ([x, y, phi, vx, vy, vphi])
    t_max :   Maximum integration runtime (seconds)
    v_max :   Maximum object velocity before stopping integration (metres/second)
    args  :   Extra arguments to pass to func
    hstep :   Integration step size
    
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

    timeLn = 0  # starting time
    x0, y0, phi0, vx0, vy0, omega0 = y0

    time_MAX = t_max
    vFINAL = v_max
    h = hstep   # Step size  

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

    # Flag if the optimisation took too long
    STOPPED = False


    vn = np.array([vx0, vy0])
    z0 = np.array([timeLn, x0, y0])  # Four-position of the object (without the third spatial component)           

    # Initial four-position in frame M
    zM0    = Lorentz(vn,z0)
    timeMn, x0M, y0M = zM0

    # Initial state vectors
    YMn = np.array([x0M, y0M, phi0, 0, 0, omega0])            
    YL0 = np.array([x0, y0, vx0, vy0])       
    taun = 0


    x_array.append(x0)
    y_array.append(y0)
    vx_array.append(vx0)
    vy_array.append(vy0)

    phi_array.append(phi0)
    omega_array.append(omega0)

    eps_array.append(0.)
    eps_rate_array.append(0.)

    timeM_array.append(timeMn)
    tau_array.append(taun)
    timeL_array.append(timeLn)

    timeSTART = time.time()
    i = 1

    while (vn[0] < vFINAL):
        """
        The main loop of the comoving integrator. Information is passed between frames L, Mn and Mn+1.

        The loop:
            Object starts in M_n and accelerates into state n+1 via M-frame forces. This updates position, angle and velocities.
            
            The n+1 information is sent back to L, then into a new frame M_{n+1} defined by relativistic addition of the M_n velocity 
            (in L) and the new small velocity due to M-frame forces (over the time step Delta tau).
        
            Rotation information is transferred from M_n to M_{n+1} purely via Wigner rotation.
        """
        
        timeDIFF = time.time() - timeSTART
        
        if timeDIFF >= time_MAX: # Finished
            STOPPED = True
            print("Integration time limit reached.")
            break
        
        if STOPPED:
            break    
        else:                                  
            # Take a step in M and evolve state there
            try:
                timeMNew, YNew = Mstep(func,h,timeMn,YMn,vn,i,args) 
            except:  # TODO: catch specific exception
                STOPPED = True
                print("Derivative vector evaluation failed. Successfully stopped early")
                break

            # Store new M
            xNew     = YNew[0]
            yNew     = YNew[1]
            phiNew   = YNew[2]
            uxNew    = YNew[3]  
            uyNew    = YNew[4]
            omegaNew = YNew[5]

            # Convert time and position variables to frame L using inverse Lorentz transformation
            zNew  = np.array([timeMNew,xNew,yNew]) 
            uNew  = np.array([uxNew,uyNew])  # Velocity induced by the M-frame forces over small time step
            zLNew = Lorentz(-vn,zNew)

            # Defining new M_{n+1} frame as a boost from L
            vLNew   = vadd(vn,uNew)  # Uses relativistic velocity addition
            zM_NEXT = Lorentz(vLNew,zLNew)  # Won't be at the origin anymore due to forces
            vM_NEXT = np.array([0,0])  # New velocity is 0 since we've just boosted into the new frame 
            eps     = SinCosEpsilon(vn,uNew)[2]  # Wigner rotation angle
            if i==1:
                eps_rate = (eps - 0)/h
            else:    
                eps_rate = (eps - eps_array[i-2])/h
            

            # Updating state vector and M-frame velocity
            timeMn  = zM_NEXT[0]
            xM2     = zM_NEXT[1]
            yM2     = zM_NEXT[2]
            phiM2   = phiNew - eps  
            vxM2    = 0
            vyM2    = 0
            omegaM2 = omegaNew - eps_rate

            YMn = np.array([xM2,yM2,phiM2,vxM2,vyM2,omegaM2])
            vn = vLNew


            # Saving L data
            timeL_array.append(zLNew[0])
            x_array.append(zLNew[1])
            y_array.append(zLNew[2])
            vx_array.append(vLNew[0])
            vy_array.append(vLNew[1])

            # Saving M data
            timeM_array.append(timeMn)
            tau_array.append(tau_array[i-1] + h)
            
            phi_array.append(phiM2)
            omega_array.append(omegaM2)
            
            eps_array.append(eps)
            eps_rate_array.append(eps_rate)
        
        iFINAL = i
        i += 1

    positions = np.array([x_array, y_array, vx_array, vy_array])  # Frame L
    angles    = np.array([phi_array, eps_array, omega_array, eps_rate_array])  # Frame M
    times     = np.array([timeM_array,tau_array,timeL_array])
    loop_data = {'Runtime': round(timeDIFF), 'Steps': i, 'Stopped': STOPPED}
    
    return positions, angles, times, loop_data