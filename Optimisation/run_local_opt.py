"""
Script for running local optimisation on a given starting twobox grating.

In particular, the local optimisation is best suited to run on a chosen grating that comes from the global optimisation 
in run_parallel.py. It is good practice to run the local optimiser on the results produced by the global optimiser
to polish/hone the FOM to higher precision and eek out slightly better FOM values.

Every step of the local optimisation (FOM, optimisation parameters and constraint values) is saved
in a text file. 
NOTE: the first step/objective-function evaluation is always the initial grating input, which should
correspond to the optimum grating chosen by the user and extracted from the global optimisation .pkl file.
Therefore, the optimisation parameters of this first-step grating are known and can be extracted using
opt_grating.params. However, for unknown reasons, the optimisation parameters for this first step that 
are saved in the text file are incorrect, giving weird values. Just ignore!

TODO: Update to account for variable fixed parameters.
"""

# IMPORTS ####################################################################################################################################################################################
import numpy as np

from datetime import datetime

import nlopt

import sys
sys.path.append('../')

import fom
import parameters
import opt
from twobox import TwoBox


# EXTRACT OPTIMISATION RESULT ####################################################################################################################################################################################
# Initial twobox grating for local optimisation drawn from candidate optima in global optimisation
runID = "MdSnpmin1_torcwa"
num_cores = 90
# maxfev = 1
maxtime = 2820

pkl_fname = f'./Data/{runID}_FOM_optimisation_maxtime{maxtime}'
txt_fname = f'./Data/{runID}_FOM_optimisation_maxtime{maxtime}_curated.txt'
_, _, opt_grating = opt.extract_opt(pkl_fname, num_processes=num_cores, output_opt_idx=2)


# PARAMETERS ####################################################################################################################################################################################
# In the LO honing, can set Qabs to infinity since we are in the vicinity of a true optimum, 
# so it shouldn't get caught on unphysical optima (this was a problem for non-rotation optimisation,
# but I haven't seen it since then)
try:
    op = opt_grating.all_params
except AttributeError:
    op = opt_grating.params
init_grating = TwoBox(*op, wavelength=1., angle=0., Nx=100, nG=12, Qabs=np.inf, 
                      RCWA_engine=opt_grating.RCWA_engine, torcwa_edge_sharpness=opt_grating.torcwa_edge_sharpness,
                      fixed_parameters=)

goal = 0.1  # Stopping criteria for adaptive sampling in the FOM (set float for loss_goal, set int for npoints_goal)
final_speed = 1.  # percentage of c

xtol_rel = 1e-7
ftol_rel = 1e-14
maxfev = 200
txt_fname = f'./Data/{runID}_LO_maxfev{maxfev}.txt'  # save results to text file


# LOCAL OPTIMISATION ####################################################################################################################################################################################
ndof = 10
param_bounds = parameters.Bounds()[2]
# Must convert arrays to np when passing to nlopt optimiser
match init_grating.RCWA_engine:
    case "GRCWA":  # Acquire ArrayBox values
        init = [p._value for p in init_grating.params]  # UNTESTED
    case "TORCWA":  # Acquire tensor values
        init = [p.detach().numpy() for p in init_grating.params]

def objective(params):
    init_grating.params = params
    return fom.FOM_uniform(init_grating, final_speed, goal, return_grad=True)

init_objective = objective(init)[0]


# Set up constraints and objective function (see opt.py for documentation)
ctrl = 0  # count number of optimisation steps
steps = []  # record strings for each step 

def fun_nlopt(params,gradn):
    global ctrl, steps
    gradn[:] = objective(params)[1]
    y = objective(params)[0]

    box1_wide = opt.box1_too_wide(params,gradn)
    box2_wide = opt.box2_too_wide(params,gradn)
    bcd_redundancy = opt.bcd_redundant(params,gradn)
    box_overlap = opt.boxes_overlap(params,gradn)
    box_clipping = opt.boxes_clip_unit_cell(params,gradn) 
    
    step = (ctrl, y, params, box1_wide, box2_wide, bcd_redundancy, box_overlap, box_clipping)
    steps.append(step)
    step_result = f"Step: {ctrl}, FOM: {y}, params: {params} \
                    \n\tBox wideness: ({box1_wide},{box2_wide}), BCD redundancy: {bcd_redundancy}, \
                    \n\tBoxes overlapping: {box_overlap}, Boxes clipping: {box_clipping}"
    print(step_result)

    ctrl += 1
    return y

lb = [bounds[0] for bounds in param_bounds]
ub = [bounds[1] for bounds in param_bounds]

local_opt = nlopt.opt(nlopt.LD_MMA, ndof)
local_opt.set_lower_bounds(lb)
local_opt.set_upper_bounds(ub)

local_opt.set_xtol_rel(xtol_rel)
local_opt.set_ftol_rel(ftol_rel)
local_opt.set_maxeval(maxfev)

local_opt.add_inequality_constraint(opt.box1_too_wide)
local_opt.add_inequality_constraint(opt.box2_too_wide)
local_opt.add_inequality_constraint(opt.bcd_redundant)
local_opt.add_inequality_constraint(opt.boxes_overlap)
local_opt.add_inequality_constraint(opt.boxes_clip_unit_cell)

local_opt.set_max_objective(fun_nlopt)
opt_params = local_opt.optimize(init)
optimum = local_opt.last_optimum_value()


init_line = repr(init)
opt_params_line = repr(opt_params)

hyperparams_dict = {'wavelength': init_grating.wavelength,
                    'angle': init_grating.angle,
                    'Nx': init_grating.Nx, 'nG': init_grating.nG, 'Qabs': init_grating.Qabs,
                    'RCWA engine': init_grating.RCWA_engine, 'TORCWA box sharpness': init_grating.torcwa_edge_sharpness,
                    'Fixed parameters': init_grating.fixed_parameters}
hyperparams_line = str(hyperparams_dict)
FOM_params_dict = {'final_speed': final_speed, 'goal': goal}
FOM_params_line = str(FOM_params_dict)

bounds_dict = {'bounds': param_bounds}
bounds_line = str(bounds_dict)

LO_dict = {'xtol_rel': f"{xtol_rel:.1E}", 'ftol_rel': f"{ftol_rel:.1E}", 'maxfev': maxfev}
LO_line = str(LO_dict)

time_at_execution = str(datetime.now())

init_lines = ["\n\n------------------------------------------------------------------------------------------------------------------------------------\n"
                , f"LO Honing \n"
                , f"Date & time         | {time_at_execution}\n"
                ,  "\n"
                , f"Hyperparameters     | {hyperparams_line}\n"
                , f"FOM parameters      | {FOM_params_line}\n"
                , f"Bounds              | {bounds_line}\n"
                ,  "\n"
                , f"LO options          | {LO_line}\n"
                , f"Initial guess       | ({init_objective}, {init_line})\n"
                , "------------------------------------------------------------------------------------------------------------------------------------\n"]

step_lines = []
for step in steps:
    ctrl, y, params, box1_wide, box2_wide, bcd_redundancy, box_overlap, box_clipping = step
    step_line = f"\nStep: {ctrl}, FOM: {y}, params: {params} \
                    \n\tBox wideness: ({box1_wide},{box2_wide}), BCD redundancy: {bcd_redundancy}, \
                    \n\tBoxes overlapping: {box_overlap}, Boxes clipping: {box_clipping}"
    step_lines.append(step_line)

result_lines = ["------------------------------------------------------------------------------------------------------------------------------------\n"
                , f"LO (Max, maximiser) | ({optimum}, {opt_params_line})\n"
                , "------------------------------------------------------------------------------------------------------------------------------------\n"]

with open(txt_fname, "a") as result_file:
    result_file.writelines(init_lines)
    result_file.writelines(step_lines)
    result_file.writelines(result_lines)