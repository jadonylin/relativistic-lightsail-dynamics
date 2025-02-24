"""
A script for simulating the dynamics of a twobox grating, calculated using a comoving integrator.
"""

import numpy as np

import pickle

from scipy.interpolate import RegularGridInterpolator

import sys
sys.path.append("../")

import time

from SR_functions import Gamma, Dv, vadd, SinCosTheta, SinCosEpsilon, ABSC, E_eps, erf, Parameters, gaussian_width, Lorentz, norm_squared


# The efficiency factors are too expensive to calculate in real time, so pre-calculated tables are used.
grating_type = "Second"

if grating_type == "Ilic":
    klambda = 650
    kdelta = 1000
    pkl_load_name = rf'Data/Tables/Ilic_Lookup_table_lambda_{klambda}_by_delta_{kdelta}.pkl'
if grating_type == "Second":
    klambda = 1000
    kdelta = 1000
    pkl_load_name = rf'Data/Tables/Lookup_table_lambda_{klambda}_by_delta_{kdelta}.pkl'


with open(pkl_load_name, 'rb') as f: 
    data = pickle.load(f)


Q1 = data['Q1']
Q2 = data['Q2']
PD_Q1_delta = data['PD_Q1_delta']
PD_Q2_delta = data['PD_Q2_delta']
PD_Q1_lambda = data['PD_Q1_lambda']
PD_Q2_lambda = data['PD_Q2_lambda']
lambda_array = data['lambda array']
delta_array = data['delta array']

interp_Q1           =   RegularGridInterpolator( (lambda_array,delta_array), Q1)
interp_Q2           =   RegularGridInterpolator( (lambda_array,delta_array), Q2)
interp_PD_Q1_delta  =   RegularGridInterpolator( (lambda_array,delta_array), PD_Q1_delta)
interp_PD_Q2_delta  =   RegularGridInterpolator( (lambda_array,delta_array), PD_Q2_delta)
interp_PD_Q1_lambda =   RegularGridInterpolator( (lambda_array,delta_array), PD_Q1_lambda)
interp_PD_Q2_lambda =   RegularGridInterpolator( (lambda_array,delta_array), PD_Q2_lambda)


def Q1_call(delta, lam):
    return interp_Q1( np.array( [lam,delta] ) )[0]
def Q2_call(delta, lam):
    return interp_Q2( np.array( [lam,delta] ) )[0]

def PD_Q1_delta_call(delta, lam):
    return interp_PD_Q1_delta( np.array( [lam,delta] ) )[0]
def PD_Q2_delta_call(delta, lam):
    return interp_PD_Q2_delta( np.array( [lam,delta] ) )[0]

def PD_Q1_lambda_call(delta, lam):
    return interp_PD_Q1_lambda( np.array( [lam,delta] ) )[0]
def PD_Q2_lambda_call(delta, lam):
    return interp_PD_Q2_lambda( np.array( [lam,delta] ) )[0]


I, L, m, c = Parameters()
I = 10e9
I_string = "10G"
w = gaussian_width(grating_type)
wavelength = 1


def aM(t,yvec,vL,i):
    """
    Calculate the lightsail state-vector acceleration using the frame M forces. Implements acceleration
    equations derived in the paper.


    Parameters
    ----------
    t    : Time measured in Frame Mn
    yvec : State vector measured in Frame Mn - [x, y, phi, vx, vy, vphi].
    vL   : Velocity of Frame Mn relative to Frame L - [vx, vy]
    i    : Input step (for troubleshooting)
    
    Returns
    -------
    [vx,vy,vphi,fx,fy,fphi] :   The derivative of the state vector with respect to the sail's proper time
    """
    
    # State vector information. Is transferred from Mn to L to Mn+1
    # All lengths and velocities are dimensional.
    xM  = yvec[0];     yM = yvec[1];    phiM = yvec[2]
    vxM = yvec[3];    vyM = yvec[4];   vphiM = yvec[5]

    vx = vL[0];       vy = vL[1]

    # Convenience factors in the equations of motion
    sintheta, costheta, theta = SinCosTheta(vL)
    A, B, S, C = ABSC(vL,phiM)
    E = E_eps(vL,phiM)

    # Rotation information. Is transferred from Mn directly to Mn+1
    delta    = theta - phiM
    sindelta = np.sin(delta)
    cosdelta = np.cos(delta)

    sinphi   = np.sin(phiM)
    cosphi   = np.cos(phiM)


    D   = Dv(vL)
    g   = Gamma(vL)
    lam = wavelength / D  # Incident wavelength in Frame Mn
    
    try:
        Q1R = Q1_call(delta,lam);    Q2R =  Q2_call(delta,lam);    
        Q1L = Q1_call(-delta,lam);   Q2L = -Q2_call(-delta,lam);   

        dQ1ddeltaR  =  PD_Q1_delta_call(delta,lam);     dQ2ddeltaR  = PD_Q2_delta_call(delta,lam)
        dQ1ddeltaL  = -PD_Q1_delta_call(-delta,lam);    dQ2ddeltaL  = PD_Q2_delta_call(-delta,lam)

        dQ1dlambdaR = PD_Q1_lambda_call(delta,lam);     dQ2dlambdaR =  PD_Q2_lambda_call(delta,lam)
        dQ1dlambdaL = PD_Q1_lambda_call(-delta,lam);    dQ2dlambdaL = -PD_Q2_lambda_call(-delta,lam)

        # Define T_{pr,j}' as angular correction terms, containing the Q1 and Q2 dispersion derivatives
        T1R = (A/costheta - E) * dQ1ddeltaR + cosphi * lam * dQ1dlambdaR
        T1L = (A/costheta - E) * dQ1ddeltaL + cosphi * lam * dQ1dlambdaL
        T2R = (A/costheta - E) * dQ2ddeltaR + cosphi * lam * dQ2dlambdaR
        T2L = (A/costheta - E) * dQ2ddeltaL + cosphi * lam * dQ2dlambdaL
    except:  # TODO: catch specific exception
        print(rf"Failed on delta'={delta}, lambda'={lam}")
        print(rf"Data boundaries: delta' in ({delta_array[0]}, {delta_array[-1]}), lambda' in ({lambda_array[0]}, {lambda_array[-1]})")
        print(rf"Failed on i={i}, t={t}, v={vL}")
        STOPPED = True

    # Transformed Gaussian intensity distribution from Frame L to Frame Mn
    A_int = yM     * (1 + g**2/(g+1)*vy**2/c**2) + xM     * g**2/(g+1)*vx*vy/c**2 + g*vy*t
    B_int = cosphi * (1 + g**2/(g+1)*vy**2/c**2) + sinphi * g**2/(g+1)*vx*vy/c**2

    XR = A_int + B_int*L/2
    XL = A_int - B_int*L/2

    expR = np.exp(-2/w**2 * XR**2)
    expL = np.exp(-2/w**2 * XL**2)
    
    erfR = erf(np.sqrt(2)/w*XR)
    erfL = erf(np.sqrt(2)/w*XL)
    
    expMID = np.exp(-2*A_int**2/w**2)
    erfMID = erf(np.sqrt(2)/w*A_int)

    # Integrated moments of intensity
    I0R =  w/(2*B_int) * np.sqrt(np.pi/2) * (erfR - erfMID)
    I0L = -w/(2*B_int) * np.sqrt(np.pi/2) * (erfL - erfMID)
    
    I1R = w/(4*B_int**2) * ( w*(expMID - expR) - np.sqrt(2*np.pi)*A_int*(erfR - erfMID) )
    I1L = w/(4*B_int**2) * ( w*(expMID - expL) - np.sqrt(2*np.pi)*A_int*(erfL - erfMID) )
    
    I2R = w/(16*B_int**3) * ( -4*w*(A_int*expMID - XL*expR) + np.sqrt(2*np.pi)*(4*A_int**2 + w**2)*(erfR - erfMID) )
    I2L = w/(16*B_int**3) * (  4*w*(A_int*expMID - XR*expL) - np.sqrt(2*np.pi)*(4*A_int**2 + w**2)*(erfL - erfMID) )

    # Forces
    fx = ((1/m)*(D**2*I/c)
            * ( (Q1R*costheta - Q2R*sintheta)*I0R + (Q1L*costheta - Q2L*sintheta)*I0L
                + (vphiM/c)
                    * ( (costheta*(2*cosphi*Q1R - T1R) - sintheta*(2*cosphi*Q2R - T2R))*I1R
                        - (costheta*(2*cosphi*Q1L - T1L) - sintheta*(2*cosphi*Q2L - T2L))*I1L
                        + (-(B - sintheta*E)*Q1R + (A + costheta*E)*Q2R)*I1R
                        - (-(B - sintheta*E)*Q1L + (A + costheta*E)*Q2L)*I1L
                    ) 
            )
    )
    
    fy = ((1/m)*(D**2*I/c)
            * ( (Q1R*sintheta + Q2R*costheta)*I0R + (Q1L*sintheta + Q2L*costheta)*I0L
                + (vphiM/c)
                    * ( (sintheta*(2*cosphi*Q1R - T1R) + costheta*(2*cosphi*Q2R - T2R))*I1R
                        - (sintheta*(2*cosphi*Q1L - T1L) + costheta*(2*cosphi*Q2L - T2L))*I1L
                        + (-(A + costheta*E)*Q1R - (B - sintheta*E)*Q2R)*I1R
                        - (-(A + costheta*E)*Q1L - (B - sintheta*E)*Q2L)*I1L
                    ) 
            )
    ) 
    
    fphi = (-(12/(m*L**2))*(D**2*I/c) 
            * ( (Q1R*cosdelta - Q2R*sindelta)*I1R - (Q1L*cosdelta - Q2L*sindelta)*I1L 
                + (vphiM/c)
                    * ( (cosdelta*(2*cosphi*Q1R - T1R) - sindelta*(2*cosphi*Q2R - T2R))*I2R 
                        + (cosdelta*(2*cosphi*Q1L - T1L) - sindelta*(2*cosphi*Q2L - T2L))*I2L  
                        + (-(C - sindelta*E)*Q1R + (S + cosdelta*E)*Q2R)*I2R
                        + (-(C - sindelta*E)*Q1L + (S + cosdelta*E)*Q2L)*I2L
                    ) 
            )
    )


    F = np.array([vxM,vyM,vphiM,fx,fy,fphi])
    
    return F


def Mstep(h,tn,yn,vL,i):
    """Step in the direction of the acceleration vector aM in frame Mn using fourth-order Runge-Kutta."""

    k1 = h*aM(tn      , yn       , vL, i)
    k2 = h*aM(tn+0.5*h, yn+0.5*k1, vL, i)
    k3 = h*aM(tn+0.5*h, yn+0.5*k2, vL, i)
    k4 = h*aM(tn+h    , yn+k3    , vL, i)

    yNew = yn + 1/6*(k1 + 2*k2 + 2*k3 + k4)
    tNew = tn + h

    return tNew,yNew


## Optimisation parameters and initial conditions ##
timeLn = 0
x0 = 0
vx0 = 0

## Optimised - 1st
# y0      = 4.246092324538898e-07
# phi0    = 1.662492890429048e-07
# vy0     = -1.8065865332297213
# omega0  = -0.85798762975541
## Optimised - 2nd
y0     = 3.590704173892898e-07
phi0   = 2.978897761047781e-08
vy0    = -1.9990833152857805
omega0 = -0.036371913744121846

## Ilic - 1st
# y0      = 1.3183489420398592e-07
# phi0    = -3.858944981371387e-09
# vy0     = -1.7076261180956787
# omega0  = -1.0411235013620457
## Ilic - 2nd 
# y0      = 5.64330183341613e-08
# phi0    = 1.9109909983391035e-08
# vy0     = -1.998710957511268
# omega0  = -0.06679404481428496


# x0=0;   y0=-(5/100)*L;      phi0=0            #y0=-0.05*L
# vx0=0;  vy0=0;              omega0=0

Y0 = np.array([x0,y0,phi0,vx0,vy0,omega0])

time_MAX = 8.5*60*60  # Maximum runtime (seconds)
 
h = 1e-4   # Step size  
runID = 2  # Added to the output data filename


################################
## Integration ##
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
z0 = np.array([timeLn, x0, y0])  # Four-position of the sail (without the third spatial component)           

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

timeM_array.append(timeMn)
tau_array.append(taun)
timeL_array.append(timeLn)

timeSTART = time.time()
i = 1
i_STOP = 100  # for debugging
vFINAL = 0.027*c


while (vn[0] < vFINAL):
    """
    The main loop of the comoving integrator. Information is passed between frames L, Mn and Mn+1.

    The loop:
        Sail starts in M_n and accelerates into state n+1 via M-frame forces. This updates position, angle and velocities.
        
        The n+1 information is sent back to L, then into a new frame M_{n+1} defined by relativistic addition of the M_n velocity 
        (in L) and the new small velocity due to M-frame forces (over the time step Delta tau).
    
        Rotation information is transferred from M_n to M_{n+1} purely via Wigner rotation.
    """
    
    timeDIFF = time.time() - timeSTART
    
    if timeDIFF >= time_MAX: # Finished
        STOPPED = True
        print("Stopped yay :)")
        break
    
    if STOPPED:
        break    
    else:                                  
        # Take a step in M and evolve state there
        try:
            timeMNew, YNew = Mstep(h,timeMn,YMn,vn,i) 
        except:  # TODO: catch specific exception
            STOPPED = True
            print("Force failed: Successfully stopped early")
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
        
        ###############################################
        ### Repeating
        # New M coordinates
        timeMn   = zM_NEXT[0]                  
        xM2      = zM_NEXT[1]
        yM2      = zM_NEXT[2]
        phiM2    = phiNew - eps  
        vxM2     = 0
        vyM2     = 0
        omegaM2  = omegaNew - eps_rate                

        YMn = np.array([xM2,yM2,phiM2,vxM2,vyM2,omegaM2])
        vn = vLNew

        ###############################################
        ### Saving L data
        timeL_array.append(zLNew[0])
        
        x_array.append(zLNew[1])
        y_array.append(zLNew[2])
        vx_array.append(vLNew[0])
        vy_array.append(vLNew[1])

        #### Saving M data
        timeM_array.append(timeMn)
        tau_array.append(tau_array[i-1] + h)
        
        phi_array.append(phiM2)
        omega_array.append(omegaM2)
        
        eps_array.append(eps)
        eps_rate_array.append(eps_rate)
    
    iFINAL=i
    i+=1

t_end = timeDIFF
t_end_sec = round(t_end)
t_end_min = round(t_end/60)
t_end_hours = round(t_end/60**2)

YL                  = np.array( [x_array, y_array, vx_array, vy_array] )
phi_nparray         = np.array(phi_array)
omega_nparray       = np.array(omega_array)
timeM_nparray       = np.array(timeM_array)
tau_nparray         = np.array(tau_array)
timeL_nparray       = np.array(timeL_array)
eps_nparray         = np.array(eps_array)
eps_rate_nparray    = np.array(eps_rate_array)

data = {'YL': YL, 'phiM': phi_nparray, 'phidot': omega_nparray, 
        'timeM': timeM_nparray, 'tau': tau_nparray, 'timeL': timeL_nparray, 
        'eps': eps_nparray, 'epsdot': eps_rate_nparray, 
        'step': h, 'duration (min)':t_end_min, 'i': iFINAL, 'Stopped': STOPPED,
        'Initial': Y0, 'Intensity': I}
pkl_fname = f'./Data/{grating_type}_Dynamics_run{runID}_I{I_string}.pkl'

# Save result
with open(pkl_fname, 'wb') as data_file:
    pickle.dump(data, data_file)