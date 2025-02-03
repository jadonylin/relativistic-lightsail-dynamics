"""
Contains the radiation pressure cross-sections and user-defined figure of merit functions. Also contains the functions that optimise
a structure with given search-space bounds.
"""


# IMPORTS ########################################################################################################################
import adaptive as adp

import autograd.numpy as npa
from autograd import grad
from autograd.scipy.special import erf as autograd_erf
from autograd.numpy import linalg as npaLA



import numpy as np
import nlopt

import os
os.environ["OMP_NUM_THREADS"] = "1" 
os.environ["OPENBLAS_NUM_THREADS"] = "1" 
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 

import random

import scipy

from typing import Callable

from parameters import Parameters, Initial_bigrating, D1_ND
from twobox_updated import TwoBox

I0, L, m, c = Parameters()



# FUNCTIONS ########################################################################################################################

def FD(grating: TwoBox) -> float:
    """
    Calculate the grating single-wavelength figure of merit FD.

    Parameters
    ----------
    grating :           TwoBox instance containing the grating parameters
    gaussian_width :    Width of gaussian beam
    """
    
    Q1,Q2,PD_Q1_angle,PD_Q2_angle,PD_Q1_wavelength,PD_Q2_wavelength=grating.return_Qs()
    w=grating.gaussian_width
    w_bar=w/L

    # Starting wavelength set to 1
    lam=grating.wavelength  # needs to be lambda'

    D=1/lam 
    g=(npa.power(lam,2) + 1)/(2*lam) 

    # Set-up (not sure about whether left or right makes sense - constraints)
    Q1R=Q1; Q2R=Q2; PD_Q1R_angle=PD_Q1_angle;   PD_Q2R_angle=PD_Q2_angle
    PD_Q1R_omega=(lam/D)*PD_Q1_wavelength;   PD_Q2R_omega=(lam/D)*PD_Q2_wavelength

    # Symmetry
    Q1L=Q1R 
    Q2L= - Q2R

    PD_Q1L_angle= - PD_Q1R_angle
    PD_Q2L_angle=PD_Q2R_angle

    PD_Q1L_omega=PD_Q1R_omega
    PD_Q2L_omega= - PD_Q2R_omega


    ####################################
    # y acc
    fy_y= -     D**2 * (I0/(m*c)) * ( Q2R - Q2L) * ( 1 - npa.exp( -1/(2*w_bar**2) ))
    fy_phi= -   D**2 * (I0/(m*c)) * ( PD_Q2R_angle + PD_Q2L_angle) * (w/2) * npa.sqrt( npa.pi/2 ) * autograd_erf( 1/(w_bar*npa.sqrt(2)) )
    fy_vy= -    D**2 * (I0/(m*c)) * (D+1)/(D* (g+1)) * ( Q1R + Q1L + PD_Q1R_angle + PD_Q1L_angle ) * (w/2) * npa.sqrt( np.pi/2 ) * autograd_erf( 1/(w_bar*npa.sqrt(2)) )
    fy_vphi=    D**2 * (I0/(m*c)) * ( 2*( Q2R - Q2L ) - D*( PD_Q2R_omega - PD_Q2L_omega ) ) * (w/2)**2 * ( 1 - npa.exp( -1/(2*w_bar**2) ))

    ####################################
    # phi acc
    fphi_y=     D**2 * (12*I0/( m*c*L**2)) * ( Q1R + Q1L ) * (  (w/2)*npa.sqrt( npa.pi/2 )  * autograd_erf( 1/(w_bar*npa.sqrt(2)))  - (L/2)* npa.exp( -1/(2*w_bar**2) )  ) 
    fphi_phi=   D**2 * (12*I0/( m*c*L**2)) * ( PD_Q1R_angle - PD_Q1L_angle - ( Q2R - Q2L ) ) * (w/2)**2 * ( 1 - npa.exp( -1/(2*w_bar**2) ))
    fphi_vy=    D**2 * (12*I0/( m*c*L**2)) * ( PD_Q1R_angle - PD_Q1L_angle - ( Q2R - Q2L ) ) * (w/2)**2 * ( 1 - npa.exp( -1/(2*w_bar**2) )) * (D+1)/(D* (g+1))
    fphi_vphi= -D**2 * (12*I0/( m*c*L**2)) * ( 2*( Q1R + Q1L ) - D*( PD_Q1R_omega + PD_Q1L_omega ) ) * (w/2)**2 * (  (w/2)*npa.sqrt( np.pi/2 )  * autograd_erf( 1/(w_bar*npa.sqrt(2)))  - (L/2)* npa.exp( -1/(2*w_bar**2) )  ) 

    # Build the Jacobian matrix
    J00=fy_y;   J01=fy_phi;     J02=fy_vy/c;    J03=fy_vphi/c
    J10=fphi_y; J11=fphi_phi;   J12=fphi_vy/c;  J13=fphi_vphi/c
    J=npa.array([[0,0,1,0],[0,0,0,1],[J00,J01,J02,J03],[J10,J11,J12,J13]])

    # Find the real part of eigenvalues    
    EIGVALVEC=npaLA.eig(J)
    eig=EIGVALVEC[0]
    EIGreal=npa.real(eig)
    EIGimag=npa.imag(eig)
    
    def unique_filled(x, filled_value):
        """
        Returns a 4-dimensional array with unique values from `x` and the remaining
        filled by `filled_value`.

        Parameters:
        x (np.ndarray): 4-dimensional input array.
        filled_value (float): Value to fill the remaining positions.

        Returns:
        np.ndarray: A 4-dimensional array of the same shape as `x`.
        """
        
        # Sorting ensures differentiability of np.unique
        sorted_x = npa.sort(x.flatten())
        unique_values = sorted_x[np.concatenate(([True], npa.diff(sorted_x) != 0))]

        k=len(unique_values)
        for i in range(4-k):
            unique_values=npa.append(unique_values,filled_value)

        return unique_values

    FD=npa.prod(unique_filled(EIGreal,1))
    
    return FD

def FD_params_func(grating, params):    # , symboxes: bool=False, onebox: bool=False
    grating_pitch, grating_depth, box1_width, box2_width, box_centre_dist, box1_eps, box2_eps, gaussian_width, substrate_depth, substrate_eps = params
    
    grating.grating_pitch = grating_pitch
    grating.grating_depth = grating_depth
    grating.box1_width = box1_width
    grating.box2_width = box2_width
    grating.box_centre_dist = box_centre_dist
    grating.box1_eps = box1_eps
    grating.box2_eps = box2_eps
    
    grating.gaussian_width=gaussian_width
    grating.substrate_depth = substrate_depth
    grating.substrate_eps = substrate_eps

    return FD(grating)

    # Does this take the gradient w.r.t. gaussian-width as well ?

FD_grad = grad(FD_params_func, argnums=1)


def FOM_uniform(grating: TwoBox, final_speed: float=20., goal: float=0.1, return_grad: bool=True) -> float:
    """
    Calculate wavelength expectation of FD FOM (figure of merit) for the given grating over a fixed wavelength range.
    Assumes a uniform probability distribution for wavelength.

    Parameters
    ----------
    grating     :   TwoBox instance containing the grating parameters
    final_speed :   Final sail speed as percentage of light speed
    goal        :   Stopping goal for wavelength integration passed to adaptive runner. If int, use npoints_goal; if float, use loss_goal.
    return_grad :   Return [FOM, FOM gradient]
    """
    laser_wavelength = grating.wavelength # copy the starting wavelength
    Doppler = D1_ND([final_speed/100,0])
    l_min = 1 # l = grating frame wavelength normalised to laser frame wavelength
    l_max = l_min/Doppler    
    l_range = (l_min, l_max)

    # Perturbation probability density function (PDF)
    PDF_unif = 1/(l_max-l_min)
    
    # Define a one argument function to pass to learner
    def weighted_FD(l):
        grating.wavelength = l*laser_wavelength
        return PDF_unif*FD(grating)
    
    # Adaptive sample FD
    FD_learner = adp.Learner1D(weighted_FD, bounds=l_range)
    if isinstance(goal, int):
        FD_runner = adp.runner.simple(FD_learner, npoints_goal=goal)
    elif isinstance(goal, float):
        FD_runner = adp.runner.simple(FD_learner, loss_goal=goal)
    else: 
        raise ValueError("Sampling goal type not recognised. Must be int for npoints_goal or float for loss_goal.")
    
    FD_data = FD_learner.to_numpy()
    l_vals = FD_data[:,0]
    weighted_FDs = FD_data[:,1]
    
    FOM = np.trapz(weighted_FDs,l_vals)

    if return_grad:
        """
        Should return FOM (average FD over wavelength) gradient at the given grating parameters.

        Implemented by first calculating the gradient at the grating parameters then averaging the gradient over wavelength
        """
        
        # Need to copy the following immutable parameters to pass to FD_grad, otherwise get UFuncTypeError
        grating_pitch = grating.grating_pitch
        grating_depth = grating.grating_depth
        box1_width = grating.box1_width
        box2_width = grating.box2_width
        box_centre_dist = grating.box_centre_dist
        box1_eps = grating.box1_eps
        box2_eps = grating.box2_eps
        gaussian_width=grating.gaussian_width
        substrate_depth=grating.substrate_depth
        substrate_eps=grating.substrate_eps        

        params = [grating_pitch, grating_depth, box1_width, box2_width, box_centre_dist, box1_eps, box2_eps,
                  gaussian_width, substrate_depth, substrate_eps] 
        
        # Define a one argument function to pass to learner
        def weighted_FD_grad(l):
            grating.wavelength = l*laser_wavelength
            return PDF_unif*np.array(FD_grad(grating, params))

        # Adaptive sample FD_grad
        FD_grad_learner = adp.Learner1D(weighted_FD_grad, bounds=l_range)

        if isinstance(goal, int):
            FD_grad_runner = adp.runner.simple(FD_grad_learner, npoints_goal=goal)
        elif isinstance(goal, float):
            FD_grad_runner = adp.runner.simple(FD_grad_learner, loss_goal=goal)
        
        FD_grad_data = FD_grad_learner.to_numpy()
        l_vals = FD_grad_data[:,0]
        weighted_FD_grads = FD_grad_data[:,1:]
        
        FOM_grad = np.trapz(weighted_FD_grads,l_vals, axis=0)

        grating.wavelength = laser_wavelength # restore user-initialised wavelength
        return [FOM,FOM_grad]
    else:
        grating.wavelength = laser_wavelength # restore user-initialised wavelength
        return FOM


def global_optimise(objective, 
                    sampling_method: str="sobol", seed: int=0, n_sample: int=8, maxfev: int=32000,
                    xtol_rel: float=1e-4, ftol_rel: float=1e-8, param_bounds: list=[]):
    """
    Global optimise the twobox on a single CPU core using MLSL global with MMA local.

    Parameters
    ----------
    objective       :   Objective function to optimise. Objective must return (value, gradient)
    sampling_method :   "sobol" or "random" initial point sampling
    seed            :   Seed for initial random parameter space sample and grating_depth samples
    n_sample        :   Number of points for initial sample (per dimension?)
    maxfev          :   Maximum function evaluations per core
    xtol_rel        :   Relative position tolerance for MMA
    ftol_rel        :   Relative objective tolerance for MMA
    param_bounds    :   List of tuples, each containing lower and upper bounds on a parameter
    """
    grating_pitch, _ , box1_width, box2_width, box_centre_dist, box1_eps, box2_eps, gaussian_width, substrate_depth, substrate_eps =Initial_bigrating()
    
    h1_min, h1_max = param_bounds[1]
    ndof = 10    # Now 8 optimisation parameters (inc. gaussian_width)
    init = [grating_pitch, random.uniform(h1_min,h1_max), box1_width, box2_width, box_centre_dist, box1_eps, box2_eps,
            gaussian_width, substrate_depth, substrate_eps]    
    bcd_constraint = True 
    
    # Obey nlopt syntax for objective and constraint functions 
    def fun_nlopt(params,gradn):
        if gradn.size > 0:
            # Even for gradient methods, in some calls gradn will be empty []
            gradn[:] = objective(params)[1]
        y = objective(params)[0]
        return y

    # Constraints to add - must be of form h(x)<=0 
    def bcd_not_redundant(params,gradn): 
        """
        Unit cell periodicity means boxes with separation >0.5*Lam are
        equivalent to boxes with separation <0.5*Lam
        """
        Lam, _, _, _, bcd, _, _, _ , _ , _ = params
        # condition = bcd - 0.5*Lam
        condition = np.abs(bcd - 0.25*Lam) - 0.25*Lam
        return condition

    def box_gaps_non_zero(params,gradn):
        """
        Want a bigrating, so make the distance between two unit cells bigger than zero
        """

        _, _, w1, w2, bcd, _, _, _ , _ , _ = params
        condition= (w1+w2)/2 - bcd
        return condition

    def box_clips_cell_edge(params,gradn):
        """
        Gradients become expensive to compute if the boxes are cutoff at the edge of the unit cell.
        """
        Lam, _, w1, w2, bcd, _, _, _ , _ , _= params
        condition = (w1+w2)/2 + bcd - 0.98*Lam
        return condition
    
    # Choose GO and LO
    if sampling_method == 'sobol':
        global_opt = nlopt.opt(nlopt.G_MLSL_LDS, ndof)
    elif sampling_method == 'random':
        global_opt = nlopt.opt(nlopt.G_MLSL, ndof)
    else:
        global_opt = nlopt.opt(nlopt.G_MLSL_LDS, ndof)
    local_opt = nlopt.opt(nlopt.LD_MMA, ndof)

    # Set LDS and initial grating_depth seed
    nlopt.srand(seed) 
    random.seed(seed) 

    # Set options for optimiser
    global_opt.set_population(n_sample) # initial sampling points

    # Set options for local optimiser
    if bcd_constraint:
        local_opt.add_inequality_constraint(bcd_not_redundant)
    local_opt.add_inequality_constraint(box_clips_cell_edge)
    local_opt.add_inequality_constraint(box_gaps_non_zero)

    local_opt.set_xtol_rel(xtol_rel)
    local_opt.set_ftol_rel(ftol_rel)
    global_opt.set_local_optimizer(local_opt)

    # Set objective function
    global_opt.set_max_objective(fun_nlopt)
    global_opt.set_maxeval(maxfev)
    
    lb = [bounds[0] for bounds in param_bounds]
    ub = [bounds[1] for bounds in param_bounds]
    global_opt.set_lower_bounds(lb)
    global_opt.set_upper_bounds(ub)

    opt_params = global_opt.optimize(init)
    optimum = objective(opt_params)[0]
    
    return (optimum,opt_params)


# def global_optimise(objective, 
#                     sampling_method: str="sobol", seed: int=0, n_sample: int=8, maxfev: int=32000,
#                     xtol_rel: float=1e-4, ftol_rel: float=1e-8, param_bounds: list=[], 
#                     symboxes: bool=False, onebox: bool=False):
#     """
#     Global optimise the twobox on a single CPU core using MLSL global with MMA local.

#     Parameters
#     ----------
#     objective       :   Objective function to optimise. Objective must return (value, gradient)
#     sampling_method :   "sobol" or "random" initial point sampling
#     seed            :   Seed for initial random parameter space sample and grating_depth samples
#     n_sample        :   Number of points for initial sample (per dimension?)
#     maxfev          :   Maximum function evaluations per core
#     xtol_rel        :   Relative position tolerance for MMA
#     ftol_rel        :   Relative objective tolerance for MMA
#     param_bounds    :   List of tuples, each containing lower and upper bounds on a parameter
#     symboxes        :   Symmetric boxes (same permittivity and width)
#     onebox          :   Single box in unit cell
#     """
#     h1_min, h1_max = param_bounds[1]
#     if symboxes:
#         ndof = 5 # number of optimisation parameters
#         init = [1.1, random.uniform(h1_min,h1_max), 0.2, 0.15, 9] # Initial optimum guess

#         bcd_constraint = True # constrain box centre distance
#     elif onebox:
#         ndof = 4 
#         init = [1.1, random.uniform(h1_min,h1_max), 0.2, 9]

#         bcd_constraint = False
#     else:
#         ndof = 7
#         init = [1.1, random.uniform(h1_min,h1_max), 0.2, 0.2, 0.4, 9, 9]

#         bcd_constraint = True 
    
#     # Obey nlopt syntax for objective and constraint functions 
#     def fun_nlopt(params,gradn):
#         if gradn.size > 0:
#             # Even for gradient methods, in some calls gradn will be empty []
#             gradn[:] = objective(params)[1]
#         y = objective(params)[0]
#         return y
    
#     def bcd_not_redundant(params,gradn): 
#         """
#         Two conditions to avoid redundant space:
#             Symmetry wrt swapping box1 and box2, avoid by taking box centre distance > 0
#         """
#         if symboxes:
#             Lam, _, _, bcd, _ = params
#         else:
#             Lam, _, _, _, bcd, _, _ = params
#         condition = np.abs(bcd - 0.25*Lam) - 0.25*Lam
#         return condition
    
#     def box_clips_cell_edge(params,gradn):
#         if symboxes:
#             Lam, _, w1, bcd, _ = params
#             condition = w1 + bcd - 0.98*Lam
#         elif onebox:
#             Lam, _, w1, _ = params
#             condition = w1 - Lam
#         else:
#             Lam, _, w1, w2, bcd, _, _ = params
#             condition = (w1+w2)/2 + bcd - 0.98*Lam
#         return condition
    
#     # Choose GO and LO
#     global_constrained_opt = nlopt.opt(nlopt.AUGLAG, ndof)
#     if sampling_method == 'sobol':
#         global_opt = nlopt.opt(nlopt.G_MLSL_LDS, ndof)
#     elif sampling_method == 'random':
#         global_opt = nlopt.opt(nlopt.G_MLSL, ndof)
#     else:
#         global_opt = nlopt.opt(nlopt.G_MLSL_LDS, ndof)
#     local_opt = nlopt.opt(nlopt.LD_MMA, ndof)

#     # Set LDS and initial grating_depth seed
#     nlopt.srand(seed) 
#     random.seed(seed) 

#     # Set options for constrained optimiser
#     global_constrained_opt.set_local_optimizer(global_opt)
#     global_constrained_opt.set_max_objective(fun_nlopt)
#     global_constrained_opt.set_maxeval(maxfev)

#     if bcd_constraint:
#         global_constrained_opt.add_inequality_constraint(bcd_not_redundant)
#     global_constrained_opt.add_inequality_constraint(box_clips_cell_edge)

#     # Set options for optimiser
#     global_opt.set_population(n_sample) # initial sampling points

#     # Set options for local optimiser
#     # local_opt.set_xtol_rel(xtol_rel)
#     # local_opt.set_ftol_rel(ftol_rel)
#     # global_opt.set_local_optimizer(local_opt)

#     # Set objective function
#     # global_opt.set_max_objective(fun_nlopt)
#     # global_opt.set_maxeval(maxfev)
    
#     lb = [bounds[0] for bounds in param_bounds]
#     ub = [bounds[1] for bounds in param_bounds]
#     # global_opt.set_lower_bounds(lb)
#     # global_opt.set_upper_bounds(ub)

#     global_constrained_opt.set_lower_bounds(lb)
#     global_constrained_opt.set_upper_bounds(ub)

#     # opt_params = global_opt.optimize(init)
#     # optimum = objective(opt_params)[0]

#     opt_params = global_constrained_opt.optimize(init)
#     optimum = objective(opt_params)[0]
    
#     return (optimum,opt_params)