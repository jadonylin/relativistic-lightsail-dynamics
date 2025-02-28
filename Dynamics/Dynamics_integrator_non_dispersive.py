"""
Efficiency factors are evaluated at v=0 (lambda=1) throughout dynamics
"""
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import pickle
import sys
sys.path.append("../")
c=299792458

from SR_functions import Gamma, Dv, vadd, SinCosTheta, SinCosEpsilon, ABSC, E_eps, erf, Parameters, gaussian_width, Lorentz

grating_type = "Second"
pkl_load_name = rf"./Data/Tables/{grating_type}_Lookup_table_angle.pkl"

## Load data
with open(pkl_load_name, 'rb') as f: 
    data = pickle.load(f)

Q1 = data['Q1']
Q2 = data['Q2']
PD_Q1_delta = data['PD_Q1_delta']
PD_Q2_delta = data['PD_Q2_delta']
PD_Q1_lambda = data['PD_Q1_lambda']
PD_Q2_lambda = data['PD_Q2_lambda']

delta_array = data['delta array']

################################
# Interpolation
def Q1_call(delta):
    return np.interp( delta, delta_array, Q1)
def Q2_call(delta):
    return np.interp( delta, delta_array, Q2)

def PD_Q1_delta_call(delta):
    return np.interp( delta, delta_array, PD_Q1_delta)
def PD_Q2_delta_call(delta):
    return np.interp( delta, delta_array, PD_Q2_delta)

def PD_Q1_lambda_call(delta):
    return np.interp( delta, delta_array, PD_Q1_lambda)
def PD_Q2_lambda_call(delta):
    return np.interp( delta, delta_array, PD_Q2_lambda)

################################
# Force equations

# Laser parameters
I, L, m, c = Parameters()
I = 10e9
I_string = "10G"
w = gaussian_width(grating_type)
wavelength = 1


def aM(t,yvec,vL,i):
    """
    ## Inputs
    t: Frame Mn time
    yvec: Frame Mn - [x, y, phi, vx, vy, vphi]
    vL: Frame L - [vx, vy]
    i: input step (for troubleshooting)
    ## Outputs
    Returns the vector d/dtau(yvec):\n
    [vx,vy,vphi,fx,fy,fphi]
    """
    ## Y state vectors
    xM  = yvec[0];     yM = yvec[1];    phiM = yvec[2]
    vxM = yvec[3];    vyM = yvec[4];   vphiM = yvec[5]

    ## Velocity in L
    vx = vL[0];       vy = vL[1]

    ## Angles
    sintheta, costheta, theta = SinCosTheta(vL)
    A, B, S, C = ABSC(vL,phiM)
    E = E_eps(vL,phiM)

    delta    = theta - phiM
    sindelta = np.sin(delta)
    cosdelta = np.cos(delta)

    sinphi   = np.sin(phiM)
    cosphi   = np.cos(phiM)

    ## Find  M wavelength
    D   = Dv(vL)
    g   = Gamma(vL)
    lam = wavelength / D  # incoming wavelength
    try:
        Q1R = Q1_call(delta);    Q2R =  Q2_call(delta);    
        Q1L = Q1_call(-delta);   Q2L = -Q2_call(-delta);   

        dQ1ddeltaR  =  PD_Q1_delta_call(delta);     dQ2ddeltaR  = PD_Q2_delta_call(delta)
        dQ1ddeltaL  = -PD_Q1_delta_call(-delta);    dQ2ddeltaL  = PD_Q2_delta_call(-delta)

        dQ1dlambdaR = PD_Q1_lambda_call(delta);     dQ2dlambdaR =  PD_Q2_lambda_call(delta)
        dQ1dlambdaL = PD_Q1_lambda_call(-delta);    dQ2dlambdaL = -PD_Q2_lambda_call(-delta)

        ## Define T_{pr,j}'
        T1R = (A/costheta - E) * dQ1ddeltaR + cosphi * 1 * dQ1dlambdaR
        T1L = (A/costheta - E) * dQ1ddeltaL + cosphi * 1 * dQ1dlambdaL
        T2R = (A/costheta - E) * dQ2ddeltaR + cosphi * 1 * dQ2dlambdaR
        T2L = (A/costheta - E) * dQ2ddeltaL + cosphi * 1 * dQ2dlambdaL
    except:
        print(rf"Failed on delta'={delta}, lambda'={lam}")
        print(rf"Data boundaries: delta' in ({delta_array[0]}, {delta_array[-1]})")
        print(rf"Failed on i={i}, t={t}, v={vL}")
        STOPPED = True

    ## Base factors
    A_int=yM*       ( 1 + (g**2/(g+1))*(vy**2/c**2) )   + xM*       (g**2/(g+1))*(vx*vy)/c**2 + g*vy*t
    B_int=cosphi*   ( 1 + (g**2/(g+1))*(vy**2/c**2) )   + sinphi*   (g**2/(g+1))*(vx*vy)/c**2

    expR=np.exp(-2*(( A_int + B_int*(L/2) )**2)/w**2 )
    expL=np.exp(-2*(( A_int - B_int*(L/2) )**2)/w**2 )
    
    erfR=erf( (np.sqrt(2)/w)*( A_int + B_int*(L/2) ) )
    erfL=erf( (np.sqrt(2)/w)*( A_int - B_int*(L/2) ) )
    
    expMID=np.exp(-2*(A_int**2)/w**2 )
    erfMID=erf( (np.sqrt(2)/w)*A_int )
    
    XR=A_int + B_int*(L/2)
    XL=A_int - B_int*(L/2)

    ## Integrals
    I0R =  (w/(2*B_int))*np.sqrt(np.pi/2)* ( erfR - erfMID )
    I0L = -(w/(2*B_int))*np.sqrt(np.pi/2)* ( erfL - erfMID )
    
    I1R = (w/(4*B_int**2))* ( w*( expMID - expR ) - np.sqrt(2*np.pi)*A_int*( erfR - erfMID ) )
    I1L = (w/(4*B_int**2))* ( w*( expMID - expL ) - np.sqrt(2*np.pi)*A_int*( erfL - erfMID ) )
    
    I2R = (w/(16*B_int**3))* ( -4*w*(A_int*expMID - XL*expR) + np.sqrt(2*np.pi)*(4*A_int**2 + w**2)* ( erfR - erfMID) )
    I2L = (w/(16*B_int**3))* (  4*w*(A_int*expMID - XR*expL) - np.sqrt(2*np.pi)*(4*A_int**2 + w**2)* ( erfL - erfMID) )

    ## Forces
    fx=(1/m)*(D**2*I/c) * ( ( Q1R*costheta - Q2R*sintheta )*I0R + ( Q1L*costheta - Q2L*sintheta )*I0L
                           + (vphiM/c)*( ( costheta*( 2*cosphi*Q1R - T1R ) - sintheta*( 2*cosphi*Q2R - T2R ) )*I1R
                                       - ( costheta*( 2*cosphi*Q1L - T1L ) - sintheta*( 2*cosphi*Q2L - T2L ) )*I1L
                                       + (-( B - sintheta*E )*Q1R + ( A + costheta*E )*Q2R)*I1R
                                       - (-( B - sintheta*E )*Q1L + ( A + costheta*E )*Q2L)*I1L
                           ) )
    
    fy=(1/m)*(D**2*I/c) * ( ( Q1R*sintheta + Q2R*costheta )*I0R + ( Q1L*sintheta + Q2L*costheta )*I0L
                           + (vphiM/c)*( ( sintheta*( 2*cosphi*Q1R - T1R ) + costheta*( 2*cosphi*Q2R - T2R ) )*I1R
                                       - ( sintheta*( 2*cosphi*Q1L - T1L ) + costheta*( 2*cosphi*Q2L - T2L ) )*I1L
                                       + (-( A + costheta*E )*Q1R - ( B - sintheta*E )*Q2R)*I1R
                                       - (-( A + costheta*E )*Q1L - ( B - sintheta*E )*Q2L)*I1L
                           ) ) 
    
    fphi=-(12/(m*L**2))*(D**2*I/c)*( ( Q1R*cosdelta - Q2R*sindelta )*I1R - ( Q1L*cosdelta - Q2L*sindelta )*I1L 
                           + (vphiM/c)*( ( cosdelta*( 2*cosphi*Q1R - T1R ) - sindelta*( 2*cosphi*Q2R - T2R ) )*I2R 
                                       + ( cosdelta*( 2*cosphi*Q1L - T1L ) - sindelta*( 2*cosphi*Q2L - T2L ) )*I2L  
                                       + (-( C - sindelta*E )*Q1R + ( S + cosdelta*E )*Q2R)*I1R
                                       + (-( C - sindelta*E )*Q1L + ( S + cosdelta*E )*Q2L)*I1L
                           ) )

    ## Store as d/dtau (Y)=F=[vx,vy,vphi,fy,fy,fphi]
    F=np.array([vxM,vyM,vphiM,fx,fy,fphi])

    return F

def Mstep(h,tn,yn,vL,i):

    k1=h*aM(tn       , yn        , vL,i)
    k2=h*aM(tn+0.5*h , yn+0.5*k1 , vL,i)
    k3=h*aM(tn+0.5*h , yn+0.5*k2 , vL,i)
    k4=h*aM(tn+h     , yn+k3     , vL,i)

    yNew=yn+(1/6)*(k1 + 2*k2 + 2*k3 + k4)
    tNew=tn+h

    return tNew,yNew

################################
## Parameters
timeLn = 0
x0 = 0; vx0 = 0
## Optimised - 1st
y0      = 4.246092324538898e-07
phi0    = 1.662492890429048e-07
vy0     = -1.8065865332297213
omega0  = -0.85798762975541
## Optimised - 2nd
# y0      = 3.590704173892898e-07
# phi0    = 2.978897761047781e-08
# vy0     = -1.9990833152857805
# omega0  = -0.036371913744121846

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

Y0=np.array([x0,y0,phi0,vx0,vy0,omega0])

# Maximum runtime
import time
time_MAX=8.5*60*60   

## Step size   
h=1e-4      
Email_result = True
runID = 1

################################
# Frame M integration
x_array = []
y_array = []
vx_array = []
vy_array = []

timeM_array = []
tau_array = []
timeL_array = []

## Storing angles in frame M
phi_array = []
omega_array = []

## Frame Rotation angle
eps_array = []
eps_rate_array = []

## Checking whether took too long
STOPPED = False

vn = np.array([vx0, vy0])
z0 = np.array([timeLn, x0, y0])           

# Initial space (and time) in frame M
zM0     = Lorentz(vn,z0)
timeMn  = zM0[0]
x0M     = zM0[1]
y0M     = zM0[2]

## Initial Y in frame M
YMn = np.array([x0M, y0M, phi0, 0, 0, omega0])            
YL0 = np.array([x0, y0, vx0, vy0])       
taun = 0

#### Storing Initial values
x_array.append(x0)
y_array.append(y0)
vx_array.append(vx0)
vy_array.append(vy0)

phi_array.append(phi0)
omega_array.append(omega0)

timeM_array.append(timeMn)
tau_array.append(taun)
timeL_array.append(timeLn)

timeSTART=time.time()
i=1
i_STOP = 100
vFINAL= 0.027*c

################################
# Integration
while (vn[0] < vFINAL):# and (i<i_STOP): 
    timeDIFF=time.time()-timeSTART
    
    if timeDIFF>=time_MAX: # Finished
        STOPPED = True
        print("Stopped yay :)")
        break
    if STOPPED:
        break    

    else:                                  
        ###############################################
        ### Take a step in M and solve dynamics there
        try:
            timeMNew, YNew = Mstep(h,timeMn,YMn,vn,i)                # t,[x,y,phi,vx,vy,vphi]
        except:
            STOPPED = True
            print("Force failed: Successfuly stopped early")
            break

        ## Store new M
        xNew     = YNew[0]
        yNew     = YNew[1]
        phiNew   = YNew[2]
        uxNew    = YNew[3]
        uyNew    = YNew[4]
        omegaNew = YNew[5]

        ###############################################
        ### Convert position variables to frame L

        # Inverse Lorentz to store time, position variables in frame L
        zNew     = np.array([timeMNew,xNew,yNew])         # [t,x,y]
        uNew     = np.array([uxNew,uyNew])            # [ux, uy]
        zLNew    = Lorentz(-vn,zNew)               # [t,x,y]

        ###############################################
        #### Defining new M+1 frame as boost from L
        
        ## Velocity addition to find new incoming velocity
        vLNew    = vadd(vn,uNew)
        ## Boost from L
        zM_NEXT  = Lorentz(vLNew,zLNew)   # Won't be at origin anymore due to forces
        ## Velocity
        vM_NEXT  = np.array([0,0])          # New velocity is 0 since boosted into frame 
        ## Frame Rotation angle
        eps      = SinCosEpsilon(vn,uNew)[2]
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
pkl_fname = f'./Data/non-dispersive/{grating_type}_Dynamics_run{runID}_I{I_string}.pkl'

# Save result
with open(pkl_fname, 'wb') as data_file:
    pickle.dump(data, data_file)

## Send email to notify end of code
if Email_result:
    # Import the following modules
    from email.mime.text import MIMEText 
    from email.mime.multipart import MIMEMultipart 
    import smtplib 
    from login import email_address, password, from_address, to


    smtp = smtplib.SMTP('smtp.gmail.com', 587) 
    smtp.ehlo() 
    smtp.starttls() 

    # Login with your email and password 
    smtp.login(email_address,password)

    def message(subject="Python Notification", 
                text="", img=None, 
                attachment=None): 
        # build message contents 
        msg = MIMEMultipart() 
        # Add Subject 
        msg['Subject'] = subject 
        # Add text contents 
        msg.attach(MIMEText(text)) 
        return msg 

    # Call the message function 
    msg = message(subject=rf"Linux: your code has finished in {t_end_sec} seconds, or {t_end_min} minutes, or {t_end_hours} hours!" ,
                  text=rf"Did it stop early? {STOPPED}",img=None, 
                  attachment=None) 

    # Provide some data to the sendmail function! 
    smtp.sendmail(from_addr=from_address, to_addrs=to, msg=msg.as_string()) 

    # Finally, don't forget to close the connection 
    smtp.quit()

