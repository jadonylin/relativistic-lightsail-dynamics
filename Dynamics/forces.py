"""
A module containing the force functions to be integrated by odecmvint.

Add your own custom force functions in here and call them in Dynamics_integrator.py or your own dynamics-solving script.
"""

import numpy as np

import pickle

from scipy.interpolate import RegularGridInterpolator

import sys
sys.path.append("../")

from cmvint import InterpolateError
from Optimisation.opt import extract_opt
from parameters import Parameters
from specrel import Gamma, Dv, SinCosTheta, ABSC, E_eps, erf


I, L, m, c = Parameters()
wavelength = 1


def load_essential_data(opt_gratings_data_fname: str, output_opt_idx: int, lookup_data_fname: str):
    """
    Load optimised grating and Qpr lookup table data.

    The Qpr vs efficiency and wavelength lookup data should be called in your main dynamics script and passed to create_interpolation_funcs.

    Parameters
    ----------
    opt_gratings_data_fname :   Optimisation data .pkl filename(s). Passed to extract_opt.
    output_opt_idx          :   Index for the optimised grating twobox object to extract directly. Index 0 
                                is the best grating out of those stored in opt_gratings_data_fname.
    lookup_data_fname       :   Lookup table data filename.

    Returns
    -------
    gaussian_width :   Gaussian-beam width
    lookup_data    :   Qpr lookup table data
    """
    _, _, opt_grating = extract_opt(opt_gratings_data_fname, output_opt_idx)
    gaussian_width = opt_grating.params[-3]  # TODO: replace with direct self.gaussian_width extraction once this branch is merged with jl
    with open(lookup_data_fname, 'rb') as lookup_file: 
        lookup_data = pickle.load(lookup_file)
    return gaussian_width, lookup_data
    

def create_interpolation_funcs(data):
    """Take Qpr vs efficiency and wavelength lookup data and return interpolation functions for Qpr and their derivatives"""
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
    
    return Q1_call, Q2_call, PD_Q1_delta_call, PD_Q2_delta_call, PD_Q1_lambda_call, PD_Q2_lambda_call


def aM(t,yvec,vL,i,w,interpolation_funcs):
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