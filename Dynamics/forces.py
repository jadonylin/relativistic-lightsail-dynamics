"""
A module containing the force functions to be integrated by odecmvint.

Add your own custom force functions in here and call them in Dynamics_integrator.py or your own 
dynamics-solving script.
"""
# test
import numpy as np

import pickle

from scipy.interpolate import RegularGridInterpolator

import sys
sys.path.append("../")

from cmvint import InterpolateError
from Optimisation.opt import extract_opt
from parameters import Parameters
from specrel import Gamma, Dv, SinCosTheta, ABSC, E_eps, erf, Lorentz


I, L, m, c = Parameters()
wavelength = 1


def load_essential_data(opt_gratings_data_fname: str, num_processes: int, output_opt_idx: int, lookup_data_fname: str):
    """
    Load optimised grating and Qpr lookup table data.

    The Qpr vs efficiency and wavelength lookup data should be called in your main dynamics script and passed to create_interpolation_funcs.

    Parameters
    ----------
    opt_gratings_data_fname :   Optimisation data .pkl filename(s). Passed to extract_opt.
    num_processes           :   Number of processes used in the optimisation whose results were stored in opt_gratings_data_fname
    output_opt_idx          :   Index for the optimised grating twobox object to extract directly. Index 0 
                                is the best grating out of those stored in opt_gratings_data_fname.
    lookup_data_fname       :   Lookup table data filename.

    Returns
    -------
    gaussian_width :   Gaussian-beam width
    lookup_data    :   Qpr lookup table data
    """
    _, _, opt_grating = extract_opt(opt_gratings_data_fname, num_processes, output_opt_idx)
    gaussian_width = opt_grating.gaussian_width
    with open(lookup_data_fname, 'rb') as lookup_file: 
        lookup_data = pickle.load(lookup_file)
    return gaussian_width, lookup_data
    

def create_interpolation_funcs(data: dict, has_angle_data: bool=True):
    """
    Take Qpr vs angle and wavelength lookup data and return interpolation functions for Qpr and their derivatives.
    If has_angle_data is False, the interpolation functions will only interpolate over wavelength.
    """
    
    Q1 = data['Q1']
    Q2 = data['Q2']
    PD_Q1_delta = data['PD_Q1_delta']
    PD_Q2_delta = data['PD_Q2_delta']
    PD_Q1_lambda = data['PD_Q1_lambda']
    PD_Q2_lambda = data['PD_Q2_lambda']
    lambda_array = data['lambda array']
    try:
        delta_array = data['delta array']
    except KeyError:
        pass
    
    if has_angle_data:
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
    else:
        def Q1_call(lam):
            return np.interp(lam, lambda_array, Q1)
        def Q2_call(lam):
            return np.interp(lam, lambda_array, Q2)
        def PD_Q1_delta_call(lam):
            return np.interp(lam, lambda_array, PD_Q1_delta)
        def PD_Q2_delta_call(lam):
            return np.interp(lam, lambda_array, PD_Q2_delta)
        def PD_Q1_lambda_call(lam):
            return np.interp(lam, lambda_array, PD_Q1_lambda)
        def PD_Q2_lambda_call(lam):
            return np.interp(lam, lambda_array, PD_Q2_lambda)
    
    return Q1_call, Q2_call, PD_Q1_delta_call, PD_Q2_delta_call, PD_Q1_lambda_call, PD_Q2_lambda_call


def aM(t: float, yvec: np.ndarray, vL: np.ndarray, i: int, w: float, interpolation_funcs: callable):
    """
    Calculate the lightsail state-vector acceleration using the frame M forces. Implements acceleration
    equations derived in the paper.

    All lengths and velocities are dimensional.

    Parameters
    ----------
    t                   :   Time measured in Frame Mn (seconds)
    yvec                :   State vector measured in Frame Mn - [x, y, phi, vx, vy, vphi]
    vL                  :   Velocity of Frame Mn relative to Frame L - [vx, vy] (metres/second)
    i                   :   Input step (for troubleshooting)
    w                   :   Gaussian-beam width
    interpolation_funcs :   Interpolation functions for Qprj and their derivatives
    
    Returns
    -------
    [vx,vy,vphi,fx,fy,fphi] :   The derivative of the state vector with respect to the sail's proper time
    """
    
    # State vector information. Is transferred from Mn to L to Mn+1
    xM  = yvec[0];     yM = yvec[1];    phiM = yvec[2]
    vxM = yvec[3];    vyM = yvec[4];   vphiM = yvec[5]
    vx  = vL[0];       vy = vL[1]

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
      
    Q1_call, Q2_call, PD_Q1_delta_call, PD_Q2_delta_call, PD_Q1_lambda_call, PD_Q2_lambda_call = interpolation_funcs
    try:  
        # TODO: incorporate these interpolation calculations into a function
        # TODO: user script shouldn't need to manually catch exceptions, that should all be handled by cmvint. 
        #       Moving the interpolation calculators into a function might help with that.
        Q1R = Q1_call(delta,lam);    Q2R =  Q2_call(delta,lam);    
        Q1L = Q1_call(-delta,lam);   Q2L = -Q2_call(-delta,lam);   

        dQ1ddeltaR  =  PD_Q1_delta_call(delta,lam);     dQ2ddeltaR  = PD_Q2_delta_call(delta,lam)
        dQ1ddeltaL  = -PD_Q1_delta_call(-delta,lam);    dQ2ddeltaL  = PD_Q2_delta_call(-delta,lam)

        dQ1dlambdaR = PD_Q1_lambda_call(delta,lam);     dQ2dlambdaR =  PD_Q2_lambda_call(delta,lam)
        dQ1dlambdaL = PD_Q1_lambda_call(-delta,lam);    dQ2dlambdaL = -PD_Q2_lambda_call(-delta,lam)
    except ValueError as ve:  
        # Should be caught when interpolator tries to extrapolate outside the lookup table bounds
        print(f"Failed on delta' = {delta}, lambda' = {lam}")
        print(f"Failed on i = {i}, t = {t}, v = {vL}")
        print(f"\nOriginal error: {ve}")
        raise InterpolateError("Interpolator moved out of bounds")
    
    # Define T_{pr,j}' as angular correction terms, containing the Q1 and Q2 dispersion derivatives
    T1R = (A/costheta - E) * dQ1ddeltaR + cosphi * lam * dQ1dlambdaR
    T1L = (A/costheta - E) * dQ1ddeltaL + cosphi * lam * dQ1dlambdaL
    T2R = (A/costheta - E) * dQ2ddeltaR + cosphi * lam * dQ2dlambdaR
    T2L = (A/costheta - E) * dQ2ddeltaL + cosphi * lam * dQ2dlambdaL

    # Transformed Gaussian intensity distribution from Frame L to Frame Mn
    # TODO: move repetitive intensity integrals into a function
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


def aM_linear(t: float, yvec: np.ndarray, vL: np.ndarray, i: int, w: float, interpolation_funcs: callable, 
              damping_scaler: float=1.):
    """
    Calculate the lightsail state-vector acceleration using the Jacobian-stiffness-derived forces.
    See aM for further documentation.

    Parameters
    ----------
    t                   :   Time measured in Frame Mn (seconds)
    yvec                :   State vector measured in Frame Mn - [x, y, phi, vx, vy, vphi]
    vL                  :   Velocity of Frame Mn relative to Frame L - [vx, vy] (metres/second)
    i                   :   Input step (for troubleshooting)
    w                   :   Gaussian-beam width
    interpolation_funcs :   Interpolation functions for Qprj and their derivatives
    damping_scaler      :   Factor to scale damping forces. Default is 1 (no scaling, true linear dynamics). 
                            Set to 0 for undamped dynamics.
    
    Returns
    -------
    [vx,vy,vphi,fx,fy,fphi] :   The derivative of the state vector with respect to the sail's proper time
    """
    
    # State vector information. Is transferred from Mn to L to Mn+1
    xM  = yvec[0];     yM = yvec[1];    phiM = yvec[2]
    vxM = yvec[3];    vyM = yvec[4];   vphiM = yvec[5]
    vx  = vL[0];       vy = vL[1]

    # Convenience factors in the equations of motion
    theta = SinCosTheta(vL)[2]

    # Rotation information. Is transferred from Mn directly to Mn+1
    delta    = theta - phiM
    sinphi   = np.sin(phiM)
    cosphi   = np.cos(phiM)

    D   = Dv(vL)
    g   = Gamma(vL)
    lam = wavelength / D  # Incident wavelength in Frame Mn
    w_bar = w/L
      
    Q1_call, Q2_call, PD_Q1_delta_call, PD_Q2_delta_call, PD_Q1_lambda_call, PD_Q2_lambda_call = interpolation_funcs
    try:  
        # TODO: incorporate these interpolation calculations into a function
        # TODO: user script shouldn't need to manually catch exceptions, that should all be handled by cmvint. 
        #       Moving the interpolation calculators into a function might help with that.
        Q1R = Q1_call(lam);     Q2R =  Q2_call(lam);    
        Q1L = Q1R;              Q2L = -Q2R;   

        dQ1ddeltaR  =  PD_Q1_delta_call(lam); dQ2ddeltaR  = PD_Q2_delta_call(lam)
        dQ1ddeltaL  = -dQ1ddeltaR;            dQ2ddeltaL  = dQ2ddeltaR

        dQ1dlambdaR = PD_Q1_lambda_call(lam); dQ2dlambdaR =  PD_Q2_lambda_call(lam)
        dQ1dlambdaL = dQ1dlambdaR;            dQ2dlambdaL = -dQ2dlambdaR
    except ValueError as ve:  
        # Should be caught when interpolator tries to extrapolate outside the lookup table bounds
        print(f"Failed on delta' = {delta}, lambda' = {lam}")
        print(f"Failed on i = {i}, t = {t}, v = {vL}")
        print(f"\nOriginal error: {ve}")
        raise InterpolateError("Interpolator moved out of bounds")
    
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

    # NOTE: derivatives with respect to lambda differ from derivatives with respect to frequency offset, the latter
    # being presented in Liam's thesis
    fy_y    = - D**2 * I/(m*c) * (Q2R - Q2L) * (1 - np.exp(-1/(2*w_bar**2)))
    fy_phi  = - D**2 * I/(m*c) * (dQ2ddeltaR + dQ2ddeltaL) * w/2 * np.sqrt(np.pi/2) * erf(1/(w_bar*np.sqrt(2)))
    fy_vy   = - D**2 * I/(m*c) * 1/c * (D+1)/(D*(g+1)) * (Q1R + Q1L + dQ2ddeltaR + dQ2ddeltaL) * w/2 * np.sqrt(np.pi/2) * erf(1/(w_bar*np.sqrt(2)))
    fy_vphi =   D**2 * I/(m*c) * 1/c * (2*(Q2R - Q2L) - lam*(dQ2dlambdaR - dQ2dlambdaL)) * (w/2)**2 * (1 - np.exp(-1/(2*w_bar**2)))

    # TODO: generalise for non-flat-geometry moments of inertia
    fphi_y    =  D**2 * 12*I/(m*c*L**2) * (Q1R + Q1L) * (w/2*np.sqrt(np.pi/2) * erf(1/(w_bar*np.sqrt(2))) - L/2*np.exp(-1/(2*w_bar**2))) 
    fphi_phi  =  D**2 * 12*I/(m*c*L**2) * (dQ1ddeltaR - dQ1ddeltaL - (Q2R - Q2L)) * (w/2)**2 * (1 - np.exp(-1/(2*w_bar**2)))
    fphi_vy   =  D**2 * 12*I/(m*c*L**2) * 1/c * (D+1)/(D*(g+1)) * (dQ1ddeltaR - dQ1ddeltaL - (Q2R - Q2L)) * (w/2)**2 * (1 - np.exp(-1/(2*w_bar**2)))
    fphi_vphi = -D**2 * 12*I/(m*c*L**2) * 1/c * (2*(Q1R + Q1L) - lam*(dQ1dlambdaR + dQ1dlambdaL)) * (w/2)**2 * (w/2*np.sqrt(np.pi/2) * erf(1/(w_bar*np.sqrt(2))) - L/2*np.exp(-1/(2*w_bar**2))) 

    # x-acceleration is unchanged from nonlinear version
    fx = (1/m)*(D**2*I/c) * ( (Q1R + delta*dQ1ddeltaR - theta*Q2R)*I0R + (Q1L + delta*dQ1ddeltaL - theta*Q2L)*I0L
                              + (vphiM/c)*((2*Q1R - lam*dQ1dlambdaR)*I1R - (2*Q1L - lam*dQ1dlambdaL)*I1L) 
                            )
    
    # Jacobian-derived forces
    phidot = vphiM/L  # Derivatives are with respect to phidot, not the linear velocity vphi
    spacetime = np.array([t, xM, yM]) 
    _, _, yL = Lorentz(-vL,spacetime)  # Derivatives are with respect to y'' (U frame), not y' (M frame), so yL must be calculated
    # Gamma factor in front of the vy derivatives because the vy derivative is the derivative 
    # with respect to vy in the accelerating-frame U, not L. They are related to each other by
    # a gamma factor.
    fy = fy_y*yL + fy_phi*phiM + damping_scaler*(g*fy_vy*vy + fy_vphi*phidot) 
    fphi = fphi_y*yL + fphi_phi*phiM + damping_scaler*(g*fphi_vy*vy + fphi_vphi*phidot)

    F = np.array([vxM,vyM,vphiM,fx,fy,fphi])
    
    return F


def aM_forces_linear(t: float, yvec: np.ndarray, vL: np.ndarray, i: int, w: float, interpolation_funcs: callable):
    """
    Calculate the lightsail state-vector acceleration using the linearised frame M forces. Linearisation
    with respect to position, angle and velocities are applied to the explicit force coefficients, but 
    not to the intensity distribution.
    See aM for further documentation.

    Parameters
    ----------
    t                   :   Time measured in Frame Mn (seconds)
    yvec                :   State vector measured in Frame Mn - [x, y, phi, vx, vy, vphi]
    vL                  :   Velocity of Frame Mn relative to Frame L - [vx, vy] (metres/second)
    i                   :   Input step (for troubleshooting)
    w                   :   Gaussian-beam width
    interpolation_funcs :   Interpolation functions for Qprj and their derivatives
    
    Returns
    -------
    [vx,vy,vphi,fx,fy,fphi] :   The derivative of the state vector with respect to the sail's proper time
    """
    
    # State vector information. Is transferred from Mn to L to Mn+1
    xM  = yvec[0];     yM = yvec[1];    phiM = yvec[2]
    vxM = yvec[3];    vyM = yvec[4];   vphiM = yvec[5]
    vx  = vL[0];       vy = vL[1]

    # Convenience factors in the equations of motion
    theta = SinCosTheta(vL)[2]

    # Rotation information. Is transferred from Mn directly to Mn+1
    delta    = theta - phiM
    sinphi   = np.sin(phiM)
    cosphi   = np.cos(phiM)

    D   = Dv(vL)
    g   = Gamma(vL)
    lam = wavelength / D  # Incident wavelength in Frame Mn
      
    Q1_call, Q2_call, PD_Q1_delta_call, PD_Q2_delta_call, PD_Q1_lambda_call, PD_Q2_lambda_call = interpolation_funcs
    try:  
        # TODO: incorporate these interpolation calculations into a function
        # TODO: user script shouldn't need to manually catch exceptions, that should all be handled by cmvint. 
        #       Moving the interpolation calculators into a function might help with that.
        Q1R = Q1_call(lam);     Q2R =  Q2_call(lam);    
        Q1L = Q1R;              Q2L = -Q2R;   

        dQ1ddeltaR  =  PD_Q1_delta_call(lam); dQ2ddeltaR  = PD_Q2_delta_call(lam)
        dQ1ddeltaL  = -dQ1ddeltaR;            dQ2ddeltaL  = dQ2ddeltaR

        dQ1dlambdaR = PD_Q1_lambda_call(lam); dQ2dlambdaR =  PD_Q2_lambda_call(lam)
        dQ1dlambdaL = dQ1dlambdaR;            dQ2dlambdaL = -dQ2dlambdaR
    except ValueError as ve:  
        # Should be caught when interpolator tries to extrapolate outside the lookup table bounds
        print(f"Failed on delta' = {delta}, lambda' = {lam}")
        print(f"Failed on i = {i}, t = {t}, v = {vL}")
        print(f"\nOriginal error: {ve}")
        raise InterpolateError("Interpolator moved out of bounds")
    
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

    # Linearised forces
    fx = (1/m)*(D**2*I/c) * ( (Q1R + delta*dQ1ddeltaR - theta*Q2R)*I0R + (Q1L + delta*dQ1ddeltaL - theta*Q2L)*I0L
                              + (vphiM/c)*((2*Q1R - lam*dQ1dlambdaR)*I1R - (2*Q1L - lam*dQ1dlambdaL)*I1L) 
                            )
    fy = (1/m)*(D**2*I/c) * ( (Q2R + delta*dQ2ddeltaR + theta*Q1R)*I0R + (Q2L + delta*dQ2ddeltaL + theta*Q1L)*I0L
                              + (vphiM/c)*((2*Q2R - lam*dQ2dlambdaR)*I1R - (2*Q2L - lam*dQ2dlambdaL)*I1L) 
                            )
    fphi = -12/(m*L**2)*(D**2*I/c) * ( (Q1R + delta*(dQ1ddeltaR - Q2R))*I1R - (Q1L + delta*(dQ1ddeltaL - Q2L))*I1L
                              + (vphiM/c)*((2*Q1R - lam*dQ1dlambdaR)*I2R + (2*Q1L - lam*dQ1dlambdaL)*I2L) 
                            )

    F = np.array([vxM,vyM,vphiM,fx,fy,fphi])
    return F