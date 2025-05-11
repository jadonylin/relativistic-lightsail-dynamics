"""
A module for storing the user-defined figure of merit function and global optimisation function. 

You should import your figure-of-merit functions from opt.py into your main optimisation script. 
"""

# IMPORTS ########################################################################################################################
import adaptive as adp

import numpy as np
import nlopt

import os
os.environ["OMP_NUM_THREADS"] = "1" 
os.environ["OPENBLAS_NUM_THREADS"] = "1" 
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 

import random

import sys
sys.path.append("../")

import fom
from parameters import Parameters, D1_ND
I0, L, m, c = Parameters()
from twobox import TwoBox



# FUNCTIONS ########################################################################################################################
def FD(grating: TwoBox) -> float:
    """
    Calculate the grating single-wavelength figure of merit FD.

    Parameters
    ----------
    grating :           TwoBox instance containing the grating parameters
    """
    if grating.RCWA_engine=="TORCWA":
        return fom.FoM(grating, I0, grad_method="grad")
    else:
        return fom.FoM(grating, I0, grad_method="finite")

def FD_params_func(grating, params):
    grating.params = params
    return FD(grating)


def FOM_uniform(grating: TwoBox, final_speed: float=20., goal: float=0.1, return_grad: bool=True) -> float:
    """
    Calculate the figure of merit (FOM) for the given grating over a fixed wavelength range determined by the final speed.
    
    The figure of merit we defined is the expectation value of FD over wavelength. Assumes a uniform probability 
    density over wavelength for weighting FD.

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
    def weighted_FD(l):
        grating.wavelength = l*laser_wavelength 
        return PDF_unif*grating.to_numpy(FD(grating)) # losing autograd here by calling to_numpy, but torch tensors are not compatible with adaptive
    
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
    FOM = np.trapezoid(weighted_FDs,l_vals)

    if return_grad:
        """
        Calculate FOM gradient by calculating the gradient of FD for the given grating at all wavelengths then averaging 
        the gradient over wavelength
        """
        FD_grad = grating.npa.grad(FD_params_func, argnum=1)
        params = grating.params
        # Define a single-argument function, needed when passing to learner
        def weighted_FD_grad(l):
            grating.wavelength = l*laser_wavelength            
            return PDF_unif*grating.to_numpy(FD_grad(grating, params))

        # Adaptive sample FD_grad
        FD_grad_learner = adp.Learner1D(weighted_FD_grad, bounds=l_range)

        if isinstance(goal, int):
            FD_grad_runner = adp.runner.simple(FD_grad_learner, npoints_goal=goal)
        elif isinstance(goal, float):
            FD_grad_runner = adp.runner.simple(FD_grad_learner, loss_goal=goal)
        
        FD_grad_data = FD_grad_learner.to_numpy()
        l_vals = FD_grad_data[:,0]
        weighted_FD_grads = FD_grad_data[:,1:]
        
        FOM_grad = np.trapezoid(weighted_FD_grads,l_vals, axis=0)

        grating.wavelength = laser_wavelength  # Restore user-initialised wavelength
        return [FOM,FOM_grad] 
    else:
        grating.wavelength = laser_wavelength  # Restore user-initialised wavelength
        return FOM

# single wavelength FOM fo rdebuging

def FOM_single(grating: TwoBox, final_speed: float=20., goal: float=0.1, return_grad: bool=True) -> float:
    """
    Calculate the figure of merit (FOM) for the given grating over a fixed wavelength range determined by the final speed.
    
    The figure of merit we defined is the expectation value of FD over wavelength. Assumes a uniform probability 
    density over wavelength for weighting FD.

    Parameters
    ----------
    grating     :   TwoBox instance containing the grating parameters
    final_speed :   Final sail speed as percentage of light speed
    goal        :   Stopping goal for wavelength integration passed to adaptive runner. If int, use npoints_goal; if float, use loss_goal.
    return_grad :   Return [FOM, FOM gradient] instead of just FOM
    """

    # Starting wavelength is copied into laser_wavelength just in case grating.wavelength is unexpectedly modified
    laser_wavelength = grating.wavelength 
    # 
    # Define a single-argument function, needed when passing to learner
    
    FOM = grating.to_numpy(FD(grating))

    if return_grad:
        """
        Calculate FOM gradient by calculating the gradient of FD for the given grating at all wavelengths then averaging 
        the gradient over wavelength
        """
        FD_grad = grating.npa.grad(FD_params_func, argnum=1)
        # Need to copy the following immutable parameters to pass to FD_grad, otherwise get UFuncTypeError       


        params = grating.params
        # Define a single-argument function, needed when passing to learner
        def weighted_FD_grad(l):
            grating.wavelength = l*laser_wavelength

            return grating.to_numpy(FD_grad(grating, params))

        FOM_grad=weighted_FD_grad(1.0)
        return [FOM,FOM_grad] 
    else:
        grating.wavelength = laser_wavelength  # Restore user-initialised wavelength
        return FOM



def average_real_eigs(grating, final_speed, goal, return_eigs: bool=False, I: float=10e9):
    """
    Calculates the average of each Re(eig) over the wavelength range. 
    
    Assumes starting wavelength = 1.

    Parameters
    ----------
    grating     :   TwoBox instance 
    final_speed :   percentage speed of light
    goal        :   integer (number of points) or float (loss goal)
    return_eigs :   If true, return normalised eigenvalues. If false, return averaged eigenvalues
    I           :   Laser intensity

    Returns
    -------
    avg_Reig :   Array containing each eigenvalue's real components averaged over wavelength
    """

    Doppler = D1_ND([final_speed/100,0])
    l_min = 1  # l = grating-frame wavelength normalised to laser-frame wavelength
    l_max = l_min/Doppler    
    l_range = (l_min, l_max)
    
    PDF_unif = 1/(l_max-l_min)  # Probability density function (PDF) for averaging

    def weighted_eig_real(l):
        grating.wavelength = l
        return PDF_unif*fom.Eigs(grating, I=I, m=m, c1=c, return_vec=False)[0]

    # Adaptive sample eig_real
    eig_real_learner = adp.Learner1D(weighted_eig_real, bounds=l_range)
    if isinstance(goal, int):
        eig_real_runner = adp.runner.simple(eig_real_learner, npoints_goal=goal)
    elif isinstance(goal, float):
        eig_real_runner = adp.runner.simple(eig_real_learner, loss_goal=goal)
    else: 
        raise ValueError("Sampling goal type not recognised. Must be int for npoints_goal or float for loss_goal.")

    eig_real_data = eig_real_learner.to_numpy()
    l_vals = eig_real_data[:,0]
    eigvals = eig_real_data[:,1:]

    avg_Reig = np.trapezoid(eigvals, l_vals, axis=0)

    if return_eigs:
        return avg_Reig, l_vals, eigvals[:,0], eigvals[:,1], eigvals[:,2], eigvals[:,3]
    if not return_eigs:
        return avg_Reig
    if not isinstance(return_eigs, bool):
        raise ValueError("input return_eigs must be a bool")


## CONSTRAINT FUNCTIONS ##
# Constraints have the form h(x) <= 0, i.e. the constraint function should return a positive 
# value if the constraint is violated. Additionally, MMA takes the gradients of the constraints,
# so the constraint functions should be differentiable with respect to the optimisation parameters. 
# However, I think the gradients of constrain functions are obtained by MMA internally (likely using
# finite differences), so autogradability is not needed.
def box1_too_wide(params,gradn): 
    """
    Constraint function to prevent box1 from being wider than the unit cell.
    """
    Lam, _, w1, _, _, _, _, _, _, _ = params
    condition = w1 - Lam 
    return condition

def box2_too_wide(params,gradn): 
    """
    Constraint function to prevent box2 from being wider than the unit cell.
    """
    Lam, _, _, w2, _, _, _, _, _, _ = params
    condition = w2 - Lam 
    return condition

def bcd_redundant(params,gradn): 
    """
    Constraint function containing two conditions to avoid redundant parameter space:
        Symmetry wrt swapping box1 and box2, avoid by taking box centre distance > 0
        Unit cell periodicity means boxes with separation >0.5*Lam are equivalent to boxes with separation <0.5*Lam
    """
    Lam, _, _, _, bcd, _, _, _, _, _ = params
    condition = np.abs(bcd - 0.25*Lam) - 0.25*Lam 
    return condition

def boxes_overlap(params,gradn):
    """
    Constraint function to guarantee an asymmetric unit cell by ensuring the distance between two unit cells is larger than zero
    TODO: The boxes overlapping can still be asymmetric, so this constraint is slightly too restrictive
            However, would need to find a way to handle gradients in the overlap regime.
    """
    _, _, w1, w2, bcd, _, _, _, _, _ = params
    condition= (w1+w2)/2 - bcd 
    return condition

def boxes_clip_unit_cell(params,gradn):
    """
    Constraint function that avoids the boxes clipping the edge of the unit cell.
    
    Boxes clipping the unit cell edges can lead to unexpected objective values (user input box widths may not correspond to the 
    box width that RCWA receives). Additionally, gradients become expensive to compute in this case.
    """
    Lam, _, w1, w2, bcd, _, _, _, _, _ = params
    condition = (w1+w2)/2 + bcd - 0.98*Lam
    return condition


def global_optimise(init_params, opt_hyperparams, 
                    sampling_method: str="sobol", seed: int=0, n_sample: int=8, maxstop: dict={'maxfev': 1000, 'maxtime': 600},
                    xtol_rel: float=1e-4, ftol_rel: float=1e-8, param_bounds: list=[], return_settings: bool=False):
    """
    Global optimise the twobox on a single CPU core using MLSL global optimiser with internal MMA local optimiser.

    Parameters
    ----------
    init_params     :   Initial objective-function parameters. Must be passed to local optimiser, but is not used/not important.
    opt_hyperparams :   Hyperparameters for the optimisation. 
    sampling_method :   "sobol" or "random" initial point sampling
    seed            :   Seed for initial random parameter space sample and grating_depth samples
    n_sample        :   Number of points for initial sample (per dimension?)
    maxstop         :   Set maximum function evaluations and/or maximum walltime (minutes) per core
    xtol_rel        :   Relative position tolerance for MMA
    ftol_rel        :   Relative objective tolerance for MMA
    param_bounds    :   Ordered list of tuples, one tuple per parameter bound, each tuple containing one lower and one upper bound
    return_settings :   If true, return the optimisation settings. If false, return only the FOM and the grating object.
    """
    
    random.seed(seed)
    ndof = len(init_params)  # number of optimisation parameters  
    h1_min, h1_max = param_bounds[1]    
    h1_start = random.uniform(h1_min,h1_max)
    init_params[1] = h1_start  # Use the random h1 start value
    bcd_constraint = True 
    
    wavelength, angle, Nx, nG, Qabs, goal, final_speed, return_grad, RCWA_engine, torcwa_sharpness = opt_hyperparams
    grating = TwoBox(*init_params, wavelength, angle, Nx, nG, Qabs, RCWA_engine, torcwa_sharpness)

    def objective(params):
        grating.params = params
        return FOM_uniform(grating, final_speed, goal, return_grad)

    def fun_nlopt(params,gradn):
        """
        nlopt objective function

        Arguments: the list of parameters and the gradient of the objective with respect to the parameters at the given params.

        Returns: value of the objective function at the given params
        """
        y, dy = objective(params)
        if gradn.size > 0:  # Even for gradient methods, in some calls gradn will be empty []
            gradn[:] = dy
        
        # # Debugging: Print constraint values to ensure optimiser moves to negative regions
        # bcd_red = bcd_redundant(params,gradn)
        # boxes_overl = boxes_overlap(params,gradn)
        # boxes_clip = boxes_clip_unit_cell(params,gradn)
        # print(f"Param: {params}")
        # print(f"Gradn: {gradn}")
        # print(bcd_red)
        # print(boxes_overl)
        # print(boxes_clip)
        # print("\n")
        
        return y

    if sampling_method == 'sobol':
        global_opt = nlopt.opt(nlopt.G_MLSL_LDS, ndof)
    elif sampling_method == 'random':
        global_opt = nlopt.opt(nlopt.G_MLSL, ndof)
    else:
        global_opt = nlopt.opt(nlopt.G_MLSL_LDS, ndof)
    local_opt = nlopt.opt(nlopt.LD_MMA, ndof)

    nlopt.srand(seed) 
    n_sample = int(n_sample)
    global_opt.set_population(n_sample)  # set initial sampling points

    if bcd_constraint:
        local_opt.add_inequality_constraint(bcd_redundant)
    local_opt.add_inequality_constraint(box1_too_wide)
    local_opt.add_inequality_constraint(box2_too_wide)
    local_opt.add_inequality_constraint(boxes_clip_unit_cell)
    local_opt.add_inequality_constraint(boxes_overlap)

    local_opt.set_xtol_rel(xtol_rel)
    local_opt.set_ftol_rel(ftol_rel)
    global_opt.set_local_optimizer(local_opt)

    global_opt.set_max_objective(fun_nlopt)
    if "maxfev" in maxstop:
        maxfev = maxstop["maxfev"]
    else:
        maxfev = -1  # -1 means no limit
        print("Warning: maxfev not provided in maxstop dictionary. Ignore if this is intentional.")
    if "maxtime" in maxstop:
        maxtime = 60*maxstop["maxtime"]  # Seconds
    else:
        maxtime = -1  # -1 means no limit
        print("Warning: maxfev not provided in maxstop dictionary. Ignore if this is intentional.")
    global_opt.set_maxeval(maxfev)
    global_opt.set_maxtime(maxtime)

    lb = [bounds[0] for bounds in param_bounds]
    ub = [bounds[1] for bounds in param_bounds]
    global_opt.set_lower_bounds(lb)
    global_opt.set_upper_bounds(ub)


    opt_params = global_opt.optimize(init_params)
    optimum = objective(opt_params)[0]
    is_optimum = True
    num_fev = global_opt.get_numevals()
    print("Success on starting bounds: ", param_bounds[1])
    

    if return_settings:
        settings = {"sampling_method": sampling_method, "seed": seed, "n_sample": n_sample,
                    "maxstop": maxstop, "xtol_rel": xtol_rel, "ftol_rel": ftol_rel,
                    "param_bounds": param_bounds}
        return (optimum, grating, opt_params, is_optimum, num_fev, settings)
    else:
        return (optimum, grating, opt_params, is_optimum, num_fev)