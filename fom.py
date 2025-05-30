"""
A module to store figure of merit (FoM) functions and helper functions that deal
with linear stability analysis (LSA) of the twobox. 

User figure of merit functions should be defined here.

TODO: need better separation between opt and fom, otherwise figures of merit are mixed between them
"""

import adaptive as adp
import numpy as np
from parameters import Parameters, D1_ND
I0, L, m, c = Parameters()


def FoM(grating, I: float=1e9, grad_method: str="finite") -> float:
    """
    Choose the grating single-wavelength figure of merit F_lam.

    Parameters
    ----------
    grating     :   Calculate figure of merit for this grating
    grad_method :   Method to calculate gradient ("finite", "grad")
    
    Returns
    -------
    F_lam :   Figure of merit
    """
    # return FoM_asymp(grating,I,grad_method)
    return FoM_damp(grating,I,grad_method)


def FoM_damp(grating, I: float=1e9, grad_method: str="finite") -> float:
    """
    Damping FOM: For translation-only motion. Minimise the ratio of the damping-force coefficient 
                 to the longitudinal-force coefficient.
    
    This FOM relies on calculating radiation-pressure efficiency factors for a single grating and then 
    using symmetry to calculate the efficiency factors for the mirror-reflected grating. In this
    implementation, the optimised grating recorded via the twobox instance is the right-half grating,
    i.e. the grating lying on the positive x-axis at equilibrium. Hence, the twobox instance's parameters,
    efficiencies, etc. are all for the right-half grating, with the left-half grating obtained by inverting
    the unit cell along the x-axis about the unit-cell centre.

    Parameters
    ----------
    grating     :   Calculate figure of merit for this grating
    grad_method :   Method to calculate gradient ("finite", "grad")
    
    Returns
    -------
    F_lam :   Figure of merit
    """
    if grad_method != "grad":
        raise ValueError("grad_method must be 'grad' for efficient F_damp calculation. Use TORCWA engine.")
    l = grating.wavelength/grating.grating_pitch # must be normalised to pitch!
    Q1,Q2 = grating.Q()
    angle_zero = 0.
    # print(grating.PDrNeg1(angle_zero))
    damp = l*(grating.PDrNeg1(angle_zero) + grating.PDtNeg1(angle_zero) 
              - grating.PDr1(angle_zero) - grating.PDt1(angle_zero))
    F_lam = damp/Q1
    return F_lam

def FoM_asymp(grating, I: float=1e9, grad_method: str="finite") -> float:
    """
    Asymptotic stability FOM: Minimise the eigenvalue of the linear stability Jacobian with the 
    largest real part. Equivalent to maximising the negative eigenvalue with the smallest real part. 
    
    This FOM relies on calculating radiation-pressure efficiency factors for a single grating and then 
    using symmetry to calculate the efficiency factors for the mirror-reflected grating. In this
    implementation, the optimised grating recorded via the twobox instance is the right-half grating,
    i.e. the grating lying on the positive x-axis at equilibrium. Hence, the twobox instance's parameters,
    efficiencies, etc. are all for the right-half grating, with the left-half grating obtained by inverting
    the unit cell along the x-axis about the unit-cell centre.

    Parameters
    ----------
    grating     :   Calculate figure of merit for this grating
    I           :   Laser intensity
    grad_method :   Method to calculate gradient ("finite","grad"). Must be "finite" for optimisation
    
    Returns
    -------
    F_lam :   Figure of merit
    """
    
    eigReal, eigImag = Eigs(grating, I=I, m=m, c1=c, grad_method=grad_method, return_vec=False)

    # MdS FoM: Minimise the eigenvalue with the largest real part. Equivalent to maximising the 
    #          negative eigenvalue with the smallest real part. 
    F_lam = grating.npa.min(-eigReal)  # standard minimum
    # F_lam = grating.npa.sum(-eigReal*grating.npa.softmin(-eigReal,1.))  # softened minimum
    # F_lam = grating.npa.min(-eigReal) + grating.npa.max(-eigReal)
    
    return F_lam

def FoM_quality_factor(grating, I: float=1e9, grad_method: str="finite") -> float:
    """
    Quality factor FoM: Maximise the magnitude of the quality factor (Re(xi)/Im(xi)) 
                        for the eigenvalue with the smallest quality factor. Issue:
                        Im(xi) --> 0 will blow this up, and we need to track the sign.

    Parameters
    ----------
    grating     :   Calculate figure of merit for this grating
    I           :   Laser intensity
    grad_method :   Method to calculate gradient ("finite","grad"). Must be "finite" for optimisation
    
    Returns
    -------
    F_lam :   Figure of merit
    """
    
    raise NotImplementedError("Must determine how to handle signs and avoid Im(xi) = 0.")

def FoM_LvR(grating, I: float=1e9, grad_method: str="finite") -> float:
    """
    Last FoM implemented by Liam - not working with TORCWA
    Calculate the grating single-wavelength figure of merit F_lam using LvR's most updated method.

    Parameters
    ----------
    grating     :   Calculate figure of merit for this grating
    I           :   Laser intensity
    grad_method :   Method to calculate gradient ("finite","grad"). Must be "finite" for optimisation
    
    Returns
    -------
    F_lam :   Figure of merit
    """
    
    eigReal, eigImag = Eigs(grating, I=I, m=m, c1=c, grad_method=grad_method, return_vec=False)

    def unique_filled(x, filled_value):
        """
        Finds unique values in x and fills remaining entries with filled_value.
        The resultant array is sorted by unique values first.

        Parameters
        ----------
        x            :   4d array
        filled_value :   Float to fill remaining entries in unique_values

        Returns
        -------
        unique_values :   Unique contents of x, with remaining entries filled by filled_value
        """
        
        # Sort array to ensure differentiability
        sorted_x = grating.npa.sort(x.flatten())
        unique_values = sorted_x[grating.npa.concatenate(([True], grating.npa.diff(sorted_x) != 0))]

        # Append filled_value as needed
        k = len(unique_values)
        for i in range(4-k):
            unique_values = grating.npa.append(unique_values,filled_value)

        return unique_values

    # NOTE: In the following penalty and reward terms, all operations must be done element-wise to avoid 
    #       "RuntimeWarning: invalid value encountered in divide" during optimisation
    # TODO: Determine why we can't use npa functions here

    # LvR FoM: Reward all Re(eig) being negative
    # Fill repeated entries in eigReal with -1 so that, after squaring, they don't influence the product
    eig_real_unique     =   unique_filled(eigReal, -1)
    eig_real_neg_unique =   grating.npa.minimum(0., eig_real_unique)
    func_real_neg_array =   grating.npa.power(eig_real_neg_unique, 2)
    func_real_neg       =   func_real_neg_array[0] * func_real_neg_array[1] * func_real_neg_array[2] * func_real_neg_array[3]
    # func_real_neg       =   npa.prod(func_real_neg_array) 

    # Remove Re(eig)<0 contribution if no restoring behaviour
    # log(1+x^2) chosen as a smooth function that moves away from zero
    # NOTE: This function has zero gradient at x=0, which is bad for stepping away from zero imaginary 
    #       part. Also, the gradient saturates at large x, which doesn't matter in the sense of 
    #       needng the imaginary part to be nonzero.
    func_imag_array     =   grating.npa.log(1 + grating.npa.power(eigImag,2))
    func_imag           =   func_imag_array[0] * func_imag_array[1] * func_imag_array[2] * func_imag_array[3]
    # func_imag           =   npa.prod(func_imag_array)

    # Penalise mixed positive and negative Re(eig)
    # Fill repeated entries in eigReal with 0 so that they don't influence the sum
    real_unique_0       =   unique_filled(eigReal, 0.)
    neg_array           =   grating.npa.power(grating.npa.minimum(0.,real_unique_0), 2)
    pos_array           =   grating.npa.power(grating.npa.maximum(0.,real_unique_0), 2)
    # penalty             =   npa.sum(neg_array) * npa.sum(pos_array)
    neg_sum             =   neg_array[0] + neg_array[1] + neg_array[2] + neg_array[3]
    pos_sum             =   pos_array[0] + pos_array[1] + pos_array[2] + pos_array[3]
    penalty             =   neg_sum * pos_sum

    # Penalise all positive Re(eig)
    # Fill repeated entries in eigReal with 1 so that they don't influence the product
    real_unique_1       =   unique_filled(eigReal, 1)
    all_pos_array       =   grating.npa.power(grating.npa.maximum(0.,real_unique_1), 2)
    penalty2            =   all_pos_array[0] * all_pos_array[1] * all_pos_array[2] * all_pos_array[3]
    # penalty2            =   npa.prod(all_pos_array)


    F_lam = func_real_neg * func_imag - penalty - penalty2
    return F_lam



def _F_lam(grating) -> float:
    if grating.RCWA_engine=="TORCWA":
        return FoM(grating, I0, grad_method="grad")
    else:
        return FoM(grating, I0, grad_method="finite")

def F_lam(grating, params):
    """
    Calculate the grating single-wavelength figure of merit.
    lam is short for lambda (wavelength)

    Parameters
    ----------
    grating :   TwoBox instance containing the grating parameters
    params  :   List of parameters to be passed to the grating object. 
                The return value is calculated for a grating with these parameters.
    """
    grating.params = params
    return _F_lam(grating)


def FOM_uniform(grating, final_speed: float=20., goal: float=0.1, return_grad: bool=True) -> float:
    """
    Calculate the figure of merit (FOM) for the given grating over a fixed wavelength range determined by the final speed.
    
    The figure of merit we defined is the expectation value of F_lam over wavelength. Assumes a uniform probability 
    density over wavelength for weighting F_lam.

    Parameters
    ----------
    grating     :   TwoBox instance containing the grating parameters
    final_speed :   Final sail speed as percentage of light speed
    goal        :   Stopping goal for wavelength integration passed to adaptive runner. If int, use npoints_goal; if float, use loss_goal.
    return_grad :   Return [FOM, FOM gradient] instead of just FOM
    """

    # Starting wavelength is copied into laser_wavelength just in case grating.wavelength is unexpectedly modified
    laser_wavelength = grating.wavelength 
    Doppler = D1_ND([final_speed/100,0])
    l_min = 1  # l = grating frame wavelength normalised to laser frame wavelength
    l_max = l_min/Doppler    
    l_range = (l_min, l_max)

    PDF_unif = 1/(l_max-l_min)  # Perturbation probability density function (PDF)
    
    # Define a single-argument function, needed when passing to learner
    def weighted_F_lam(l):
        grating.wavelength = l*laser_wavelength 
        return PDF_unif*grating.to_numpy(_F_lam(grating)) # losing autograd here by calling to_numpy, but torch tensors are not compatible with adaptive
    
    F_lam_learner = adp.Learner1D(weighted_F_lam, bounds=l_range)
    if isinstance(goal, int):
        F_lam_runner = adp.runner.simple(F_lam_learner, npoints_goal=goal)
    elif isinstance(goal, float):
        F_lam_runner = adp.runner.simple(F_lam_learner, loss_goal=goal)
    else: 
        raise ValueError("Sampling goal type not recognised. Must be int for npoints_goal or float for loss_goal.")
    
    F_lam_data = F_lam_learner.to_numpy()
    l_vals = F_lam_data[:,0]
    weighted_F_lams = F_lam_data[:,1]
    FOM = np.trapezoid(weighted_F_lams,l_vals)

    if return_grad:
        """
        Calculate FOM gradient by calculating the gradient of F_lam for the given grating at all wavelengths then averaging 
        the gradient over wavelength
        """
        F_lam_grad = grating.npa.grad(F_lam, argnum=1)
        params = grating.params
        # Define a single-argument function, needed when passing to learner
        def weighted_F_lam_grad(l):
            grating.wavelength = l*laser_wavelength            
            return PDF_unif*grating.to_numpy(F_lam_grad(grating, params))

        # Adaptive sample F_lam_grad
        F_lam_grad_learner = adp.Learner1D(weighted_F_lam_grad, bounds=l_range)

        if isinstance(goal, int):
            F_lam_grad_runner = adp.runner.simple(F_lam_grad_learner, npoints_goal=goal)
        elif isinstance(goal, float):
            F_lam_grad_runner = adp.runner.simple(F_lam_grad_learner, loss_goal=goal)
        
        F_lam_grad_data = F_lam_grad_learner.to_numpy()
        l_vals = F_lam_grad_data[:,0]
        weighted_F_lam_grads = F_lam_grad_data[:,1:]
        
        FOM_grad = np.trapezoid(weighted_F_lam_grads,l_vals, axis=0)

        grating.wavelength = laser_wavelength  # Restore user-initialised wavelength
        return [FOM,FOM_grad] 
    else:
        grating.wavelength = laser_wavelength  # Restore user-initialised wavelength
        return FOM



def sail_stiffness(grating, I: float=10e9, m: float=1/1000, c1:float=299792458, 
                   grad_method: str='finite', out: str="tr", normalise: bool=False):
    """
    Calculate stiffness coefficients/Jacobian coefficients for a symmetric lightsail at equilibrium. Here, symmetric 
    means symmetric with respect to reflections about the laser-beam axis (when the CoM lies on the laser-beam axis).

    Parameters
    ----------
    grating     :   Calculate stiffnesses for this grating
    I           :   Laser intensity
    m           :   Spacecraft mass (sail membrane + payload)
    c1          :   speed of light  # TODO: why is this a parameter?
    grad_method :   Method to calculate gradient ("finite","grad"). Must be "finite" for optimisation
    out         :   Output format 
                    "tr" for translation coefficients first, then rotation coefficients. Use when outputting to Jacobian.
                    "rd" for restoring coefficients first, then damping coefficients
    normalise   :   Normalise all Jacobian coefficients by their individual dimensional factors
    
    Returns
    -------
    The eight stiffness coefficients for the lightsail at equilibrium.
    """
    
    match grad_method:
        case "finite":
            # For optimisation, need to use finite differences
            # Approximately optimal step size is 10^-6.5 for both angle and wavelength
            h_angle = 10**(-6.5)
            h_wavelength = 10**(-6.5)
            Q1R, Q2R, dQ1ddeltaR, dQ2ddeltaR, dQ1dlambdaR, dQ2dlambdaR = grating.return_Qs(h_angle, h_wavelength)
        case "grad":
            Q1R, Q2R, dQ1ddeltaR, dQ2ddeltaR, dQ1dlambdaR, dQ2dlambdaR = grating.return_Qs_auto(return_Q=True)
        case _:
            raise ValueError("grad_method not recognised. Must be 'finite' or 'grad'.")

    w = grating.gaussian_width
    w_bar = w/L  # width normalised to total grating length
    lam = grating.wavelength 

    # Convert velocity factors to wavelength factors
    # TODO: may need to change these factors to account for non-unity starting wavelengths
    D = 1/lam  # Doppler factor assuming starting wavelength is 1
    g = (grating.npa.power(lam,2) + 1)/(2*lam)  # Lorentz factor

    # Lightsail reflection-symmetry conditions
    Q1L = Q1R                ; Q2L = -Q2R;   
    dQ1ddeltaL  = -dQ1ddeltaR; dQ2ddeltaL  = dQ2ddeltaR
    dQ1dlambdaL = dQ1dlambdaR; dQ2dlambdaL = -dQ2dlambdaR        
    
    # y acceleration terms
    # NOTE: derivatives with respect to lambda differ from derivatives with respect to frequency offset, the latter
    # being presented in Liam's thesis
    fy_y      = - D**2 * I/(m*c1) * (Q2R - Q2L) * (1 - grating.npa.exp(-1/(2*w_bar**2)))
    fy_phi    = - D**2 * I/(m*c1) * (dQ2ddeltaR + dQ2ddeltaL) * w/2 * np.sqrt(np.pi/2) * grating.npa.erf(1/(w_bar*np.sqrt(2)))
    fy_vy     = - D**2 * I/(m*c1) * 1/c1 * (D+1)/(D*(g+1)) * (Q1R + Q1L + dQ2ddeltaR + dQ2ddeltaL) * w/2 * np.sqrt(np.pi/2) * grating.npa.erf(1/(w_bar*np.sqrt(2)))
    fy_phidot =   D**2 * I/(m*c1) * 1/c1 * (2*(Q2R - Q2L) - lam*(dQ2dlambdaR - dQ2dlambdaL)) * (w/2)**2 * (1 - grating.npa.exp(-1/(2*w_bar**2)))

    # phi acceleration terms
    J = m*L**2/12  # moment of inertia about the CoM
    fphi_y      =  D**2 * I/(J*c1) * (Q1R + Q1L) * (w/2*np.sqrt(np.pi/2) * grating.npa.erf(1/(w_bar*np.sqrt(2))) - L/2*grating.npa.exp(-1/(2*w_bar**2))) 
    fphi_phi    =  D**2 * I/(J*c1) * (dQ1ddeltaR - dQ1ddeltaL - (Q2R - Q2L)) * (w/2)**2 * (1 - grating.npa.exp(-1/(2*w_bar**2)))
    fphi_vy     =  D**2 * I/(J*c1) * 1/c1 * (D+1)/(D*(g+1)) * (dQ1ddeltaR - dQ1ddeltaL - (Q2R - Q2L)) * (w/2)**2 * (1 - grating.npa.exp(-1/(2*w_bar**2)))
    fphi_phidot = -D**2 * I/(J*c1) * 1/c1 * (2*(Q1R + Q1L) - lam*(dQ1dlambdaR + dQ1dlambdaL)) * (w/2)**2 * (w/2*np.sqrt(np.pi/2) * grating.npa.erf(1/(w_bar*np.sqrt(2))) - L/2*grating.npa.exp(-1/(2*w_bar**2))) 

    if normalise:
        fy_y        /= I/(m*c1)
        fy_phi      /= I*L/(m*c1)
        fy_vy       /= I*L/(m*c1**2)
        fy_phidot   /= I*L**2/(m*c1**2)
        fphi_y      /= I*L/(J*c1)
        fphi_phi    /= I*L**2/(J*c1)
        fphi_vy     /= I*L**2/(J*c1**2)
        fphi_phidot /= I*L**3/(J*c1**2)

    match out:
        case "tr":
            return grating.npa.stack((fy_y, fy_phi, fy_vy, fy_phidot, fphi_y, fphi_phi, fphi_vy, fphi_phidot))
        case "rd":
            return grating.npa.stack((fy_y, fy_phi, fphi_y, fphi_phi, fy_vy, fy_phidot, fphi_vy, fphi_phidot))
        case "mat":
            row1 = grating.npa.stack((fy_y, fy_phi,fy_vy, fy_phidot))
            row2 = grating.npa.stack((fphi_y, fphi_phi, fphi_vy, fphi_phidot))
            mat = grating.npa.stack((row1,row2))
            return mat
        case _:
            raise ValueError("Invalid output format. Must be 'tr' or 'rd'.")
        
def Eigs(grating, I: float=10e9, m: float=1/1000, c1:float=299792458, 
         grad_method: str='finite', return_vec: bool = False):
    """
    Calculate eigendecomposition of Jacobian matrix at equilibrium

    Parameters
    ----------
    grating     :   Calculate eigenvalues for this grating
    I           :   Laser intensity
    m           :   Spacecraft mass (sail membrane + payload)
    c1          :   speed of light  # TODO: why is this a parameter?
    grad_method :   Method to calculate gradient ("finite","grad"). Must be "finite" for optimisation
    return_vec  :   If true, return eigenvectors as well as eigenvalues
    
    Returns
    -------
    eigReal :   Real part of Jacobian eigenvalues
    eigImag :   Imaginary part of Jacobian eigenvalues
    eigvecs :   Eigenvectors of Jacobian matrix, normalised to unit length
    """
    stiffnesses = sail_stiffness(grating,I,m,c1,grad_method,out="mat")
    J = grating.npa.concatenate((grating.npa.array([[0,0,1,0],[0,0,0,1]]), stiffnesses))  # Jacobian matrix
    if return_vec:
        eigvals, eigvecs = grating.npa.eig(J)
        eigReal = grating.npa.real(eigvals)
        eigImag = grating.npa.imag(eigvals)
        return eigReal, eigImag, eigvecs
    else:
        if grating.RCWA_engine == "TORCWA":
            eigvals = grating.npa.eigvals(J)
        else:  # eigvals is not differentiable using HIPS/autograd in GRCWA
            eigvals, _ = grating.npa.eig(J)
        eigReal = grating.npa.real(eigvals)
        eigImag = grating.npa.imag(eigvals)
        return eigReal, eigImag


def lsa_info(grating, I: float=0.5e9, normalise: bool=False):
    """
    Calculate quantities relevant to linear stability analysis (LSA) of the twobox dynamics. Also calculates
    the radiation pressure cross sections and their derivatives.

    Parameters
    ----------
    grating   :   Calculate linear-stability info for this grating
    I         :   Incident light intensity
    normalise :   Normalise all Jacobian coefficients by their individual dimensional factors
    
    Returns
    -------
    efficiencies :   Radiation pressure cross sections and their derivatives  
    rest_coeffs  :   Restoring force/torque coefficients
    damp_coeffs  :   Damping force/torque coefficients
    eigReal      :   Real component of eigenvalues
    eigImag      :   Imaginary component of eigenvalues
    """
    efficiencies = tuple(grating.return_Qs_auto(return_Q=True))
    stiffnesses = sail_stiffness(grating,I,m,c,grad_method="grad",out="rd",normalise=normalise)
    rest_coeffs = tuple([*stiffnesses[:4]])
    damp_coeffs = tuple([*stiffnesses[4:]])
    eigReal, eigImag, eigvecs = Eigs(grating,I,m,c,grad_method="grad",return_vec=True)
    return efficiencies, rest_coeffs, damp_coeffs, eigReal, eigImag, eigvecs