import numpy as np
from scipy.interpolate import RegularGridInterpolator
import pickle
import sys
sys.path.append("../")
c=299792458

from SR_functions import Gamma, Dv, vadd, SinCosTheta, SinCosEpsilon, ABSC, E_eps, erf, Parameters, gaussian_width, Lorentz, norm_squared


grating_type = "Ilic"
klambda = 1000

if grating_type == "Ilic":
    pkl_load_name = rf'Data/Tables/Ilic_Stability_Lookup_table_lambda_{klambda}.pkl'
if grating_type == "Second":
    pkl_load_name = rf'Data/Tables/Stability_Lookup_table_lambda_{klambda}.pkl'
## Load data
with open(pkl_load_name, 'rb') as f: 
    data = pickle.load(f)

Q1 = data['Q1']
Q2 = data['Q2']
PD_Q1_delta = data['PD_Q1_delta']
PD_Q2_delta = data['PD_Q2_delta']
PD_Q1_lambda = data['PD_Q1_lambda']
PD_Q2_lambda = data['PD_Q2_lambda']

lambda_array = data['lambda array']

################################
## Interpolation
def Q1_call(lam):
    return np.interp( lam, lambda_array, Q1)
def Q2_call(lam):
    return np.interp( lam, lambda_array, Q2)
def PD_Q1_delta_call(lam):
    return np.interp( lam, lambda_array, PD_Q1_delta)
def PD_Q2_delta_call(lam):
    return np.interp( lam, lambda_array, PD_Q2_delta)
def PD_Q1_lambda_call(lam):
    return np.interp( lam, lambda_array, PD_Q1_lambda)
def PD_Q2_lambda_call(lam):
    return np.interp( lam, lambda_array, PD_Q2_lambda)

################################
# Force equations

# Laser parameters
I, L, m, c = Parameters()
I = 10e9
I_string = "10G"
w = gaussian_width(grating_type)
wavelength = 1


def aU(t,yvec,vL,i):
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
    theta = SinCosTheta(vL)[2]
    delta    = theta - phiM

    sinphi   = np.sin(phiM)
    cosphi   = np.cos(phiM)

    ## Find  M wavelength
    D   = Dv(vL)
    g   = Gamma(vL)
    lam = wavelength / D  # incoming wavelength
    w_bar = w/L

    try:
        Q1R = Q1_call(lam);     Q2R =  Q2_call(lam);    
        Q1L = Q1R;              Q2L = -Q2R;   

        dQ1ddeltaR  =  PD_Q1_delta_call(lam);       dQ2ddeltaR  = PD_Q2_delta_call(lam)
        dQ1ddeltaL  = -dQ1ddeltaR;                  dQ2ddeltaL  = dQ2ddeltaR

        dQ1dlambdaR = PD_Q1_lambda_call(lam);       dQ2dlambdaR =  PD_Q2_lambda_call(lam)
        dQ1dlambdaL = dQ1dlambdaR;                  dQ2dlambdaL = -dQ2dlambdaR

    except:
        print(rf"Failed on delta'={delta}, lambda'={lam}")
        print(rf"Data boundaries: lambda' in ({lambda_array[0]}, {lambda_array[-1]})")
        print(rf"Failed on i={i}, t={t}, v={vL}")
        STOPPED = True

    ## Base factors
    A_int=yM #*       ( 1 + (g**2/(g+1))*(vy**2/c**2) )   + xM*       (g**2/(g+1))*(vx*vy)/c**2 + g*vy*t
    B_int=cosphi #*   ( 1 + (g**2/(g+1))*(vy**2/c**2) )   + sinphi*   (g**2/(g+1))*(vx*vy)/c**2

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

    ####################################
    # y acceleration
    fy_y= -     D**2 * (I/(m*c)) *  ( Q2R - Q2L ) * ( 1 - np.exp(-1/(2*w_bar**2) ) )
    fy_phi= -   D**2 * (I/(m*c)) * ( dQ2ddeltaR + dQ2ddeltaL ) * (w/2) * np.sqrt( np.pi/2 ) * erf( 1/(w_bar*np.sqrt(2)) )
    fy_vy= -    D**2 * (I/(m*c)) * (1/c) * ( (D+1)/(D*(g+1)) ) * ( Q1R + Q1L  + dQ2ddeltaR + dQ2ddeltaL ) * (w/2) * np.sqrt( np.pi/2 ) * erf( 1/(w_bar*np.sqrt(2)) )
    fy_vphi=    D**2 * (I/(m*c)) * (1/c) * ( 2*( Q2R - Q2L ) - lam*( dQ2dlambdaR - dQ2dlambdaL ) ) * (w/2)**2 * ( 1 - np.exp( -1/(2*w_bar**2) ))

    # phi acceleration
    fphi_y=     D**2 * (12*I/( m*c*L**2)) * ( Q1R + Q1L ) * (  (w/2)*np.sqrt( np.pi/2 )  * erf( 1/(w_bar*np.sqrt(2)))  - (L/2)* np.exp( -1/(2*w_bar**2) )  ) 
    fphi_phi=   D**2 * (12*I/( m*c*L**2)) * ( dQ1ddeltaR - dQ1ddeltaL - ( Q2R - Q2L ) ) * (w/2)**2 * ( 1 - np.exp( -1/(2*w_bar**2) ))
    fphi_vy=    D**2 * (12*I/( m*c*L**2)) * (1/c) * ( (D+1)/(D*(g+1)) ) * ( dQ1ddeltaR - dQ1ddeltaL - ( Q2R - Q2L ) ) * (w/2)**2 * ( 1 - np.exp( -1/(2*w_bar**2) ))
    fphi_vphi= -D**2 * (12*I/( m*c*L**2)) * (1/c) * ( 2*( Q1R + Q1L ) - lam*( dQ1dlambdaR + dQ1dlambdaL ) ) * (w/2)**2 * (  (w/2)*np.sqrt( np.pi/2 )  * erf( 1/(w_bar*np.sqrt(2)))  - (L/2)* np.exp( -1/(2*w_bar**2) )  ) 

    # k = (xM/c) * (g**2/(g+1)) * (vx/c) + g*t
    # fy_diff = D**2 * (I/(m*c)) * ( Q2R - Q2L ) * - k * ( 1 - np.exp(-1/(2*w_bar**2) ) )
    # fphi_diff = - D**2 * (12*I/( m*c*L**2)) * ( Q1R + Q1L ) * - k * (  (w/2)*np.sqrt( np.pi/2 )  * erf( 1/(w_bar*np.sqrt(2)))  - (L/2)* np.exp( -1/(2*w_bar**2) )  ) 
    # print("fy_diff:", fy_diff)
    # print("fphi_diff:", fphi_diff)
    
    # fy_vy = fy_vy + fy_diff
    # fphi_vy = fphi_vy + fphi_diff

    # Build the Jacobian matrix
    J00=fy_y;   J01=fy_phi;     J02=fy_vy;    J03=fy_vphi
    J10=fphi_y; J11=fphi_phi;   J12=fphi_vy;  J13=fphi_vphi

    ## Forces
    fx=(1/m)*(D**2*I/c) * ( ( Q1R + delta*dQ1ddeltaR - Q2R*theta )  * I0R + ( Q1L + delta*dQ1ddeltaL - Q2L*theta )   * I0L
              + (vphiM/c)*(     ( 2*Q1R - lam*dQ1dlambdaR )         * I1R - (   ( 2*Q1L - lam*dQ1dlambdaL ) )        * I1L    ) )
    fy   = J00 * yM + J01 * phiM   +     J02 * vyM + J03 * vphiM
    fphi = J10 * yM + J11 * phiM   +     J12 * vyM + J13 * vphiM
    # ## Linearised forces
    # fy=(1/m)*(D**2*I/c) * ( ( Q2R + delta*dQ2ddeltaR + Q1R*theta )  * I0R + ( Q2L + delta*dQ2ddeltaL + Q1L*theta )  * I0L
    #           + (vphiM/c)*(     ( 2*Q2R - lam*dQ2dlambdaR )         * I1R - (   ( 2*Q2L - lam*dQ2dlambdaL ) )       * I1L    ) )
    
    # fphi=-(12/(m*L**2))*(D**2*I/c) * ( ( Q1R + delta*(dQ1ddeltaR - Q2R) )   * I1R - ( Q1L + delta*(dQ1ddeltaL - Q2L) )  * I1L
    #                      + (vphiM/c)*( ( 2*Q1R - lam*dQ1dlambdaR )          * I2R + ( ( 2*Q1L - lam*dQ1dlambdaL ) )     * I2L    ) )

    ## Store as d/dtau (Y)=F=[vx,vy,vphi,fy,fy,fphi]
    F=np.array([vxM,vyM,vphiM,fx,fy,fphi])
    return F

def Ustep(h,tn,yn,vL,i):

    k1=h*aU(tn       , yn        , vL,i)
    k2=h*aU(tn+0.5*h , yn+0.5*k1 , vL,i)
    k3=h*aU(tn+0.5*h , yn+0.5*k2 , vL,i)
    k4=h*aU(tn+h     , yn+k3     , vL,i)

    yNew=yn+(1/6)*(k1 + 2*k2 + 2*k3 + k4)
    tNew=tn+h

    return tNew,yNew

################################
## Parameters
timeLn = 0
x0 = 0; vx0 = 0
## Optimised - 1st
# y0      = 4.246092324538898e-07
# phi0    = 1.662492890429048e-07
# vy0     = -1.8065865332297213
# omega0  = -0.85798762975541
## Optimised - 2nd
# y0      = 3.590704173892898e-07
# phi0    = 2.978897761047781e-08
# vy0     = -1.9990833152857805
# omega0  = -0.036371913744121846

## Ilic - 1st
y0      = 1.3183489420398592e-07
phi0    = -3.858944981371387e-09
vy0     = -1.7076261180956787
omega0  = -1.0411235013620457
## Ilic - 2nd 
# y0      = 5.64330183341613e-08
# phi0    = 1.9109909983391035e-08
# vy0     = -1.998710957511268
# omega0  = -0.06679404481428496

# x0=0;   y0=-(0.5/100)*L;       phi0=0            #y0=-0.05*L
# vx0=0;  vy0=0;      omega0=0

Y0=np.array([x0,y0,phi0,vx0,vy0,omega0])

# Maximum runtime
import time
time_MAX=8.5*60*60   

## Step size   
h=1e-4      
Email_result = True
runID = 1

################################
# Frame U integration
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

# ## Frame Rotation angle
# eps_array = []
# eps_rate_array = []

## Checking whether took too long
STOPPED = False

vn = np.array([vx0, 0]) # Frame U
gamma0 = Gamma(vn)
z0 = np.array([timeLn, x0, y0])           

# Initial space (and time) in frame M
zM0     = Lorentz(vn,z0)
timeMn  = zM0[0]
x0M     = zM0[1]
y0M     = zM0[2]

## Initial Y in frame M
YMn = np.array([x0M, y0M, phi0, 0, gamma0*vy0, omega0])            
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
        ### Take a step in U and solve dynamics there
        try:
            timeMNew, YNew = Ustep(h,timeMn,YMn,vn,i)                # t,[x,y,phi,vx,vy,vphi]
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
        vLNew_U = np.array( [vLNew[0], 0] ) # Purely upwards
        ## Boost from L
        zM_NEXT  = Lorentz( vLNew_U, zLNew)   # Won't be at origin anymore due to forces
        ## Velocity
        vM_NEXT  = np.array([0, Gamma(vLNew) * vLNew[1] ])          # New velocity is (0,gamma vy) since boosted into frame U
        
        # ## Frame Rotation angle
        # eps      = SinCosEpsilon(vn,uNew)[2]
        # if i==1:
        #     eps_rate = (eps - 0)/h
        # else:    
        #     eps_rate = (eps - eps_array[i-2])/h
        
        ###############################################
        ### Repeating
        # New M coordinates
        timeMn   = zM_NEXT[0]                  
        xM2      = zM_NEXT[1]
        yM2      = zM_NEXT[2]
        phiM2    = phiNew #- eps  
        vxM2     = vM_NEXT[0]
        vyM2     = vM_NEXT[1]
        omegaM2  = omegaNew #- eps_rate                

        YMn = np.array([xM2,yM2,phiM2,vxM2,vyM2,omegaM2])
        vn = vLNew_U

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
        
        # eps_array.append(eps)
        # eps_rate_array.append(eps_rate)
    
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
# eps_nparray         = np.array(eps_array)
# eps_rate_nparray    = np.array(eps_rate_array)

data = {'YL': YL, 'phiM': phi_nparray, 'phidot': omega_nparray, 
        'timeM': timeM_nparray, 'tau': tau_nparray, 'timeL': timeL_nparray, 
        # 'eps': eps_nparray, 'epsdot': eps_rate_nparray, 
        'step': h, 'duration (min)':t_end_min, 'i': iFINAL, 'Stopped': STOPPED,
        'Initial': Y0, 'Intensity': I}
# pkl_fname = f'./Data/Linearised/Dynamics_Linear_Opt_run{runID}_I{I_string}.pkl'
pkl_fname = f'./Data/FrameU/{grating_type}_Dynamics_run{runID}_I{I_string}.pkl'

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

