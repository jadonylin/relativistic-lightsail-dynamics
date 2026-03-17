# Relativistic lightsail dynamics optimization and dynamics simulation.

## Overview 
The main directory contains modules that perform electromagnetic simulations (RCWA) of the grating.
The "Optimization" folder contains scripts and modules for running the optimization.
The "Dynamics" folder contains scripts and modules for running the dynamics simulation.


### Main directory
The `twobox.py` module contains the `TwoBox` class, which simulates a grating with given parameters
using rigorous coupled-wave analysis (RCWA). The grating unit cell has two resonators (two ''boxes'')
and a substrate. The modules `qprbox.py` and `plotbox.py` are superclasses defined for modularity. The
`qprbox.py` module computes the grating power efficiencies and radiation pressure efficient factors (Qpr).
The `plotbox.py` module plots some features of the grating such as unit-cell permittivity profile.

The `parameters.py` file sets the optimization search space parameters and optimization hyperparameters.

The `autolib.py` module stores the automatic-differentiation-compatible versions of numpy functions that
should be used in the optimization figure of merit to ensure gradient descent works efficiently. There 
are two options: TORCWA with PyTorch and GRCWA with autograd. The latter is kept as a legacy from previous
optimization strategies; in basically every case, TORCWA with PyTorch should be used.

Finally, `fom.py` contains the user-defined figure of merit functions that are passed to the optimizer.

### Optimization
To run the optimization, check the `run_parallel.py` docstring. Parameters for the grating and 
hyperparameters for the optimization should be set in parameters.py before executing the 
`run_parallel.py` script. The script runs optimization in parallel over a user-defined number of 
cores `num_cores`. The parameter space is divided into `num_cores` disjoint sets whose union is the
full grating parameter space defined by the user. The parameter for grating thickness (h) is the
parameter that is divided into disjoint sets. The output of the optimization is `num_cores` `.pkl`
files, each containing the best grating found by the optimizer on a single core. The output
optimization files are then analyzed by ordering them from best to worst in terms of the figure
of merit and the top results are examined to see if the grating has a physical unit cell.

The `opt.py` file contains the wrapped global optimization function and local optimization
constraint functions unique to the twobox grating. It also contains a function `extract_opt()`, which
sorts the optima from the `num_core` `.pkl` files by largest-to-smallest figure of merit and allows
you to select one by index.

The `Data` folder houses the optimization data we obtained for two separate runs: narrow band 
(Fasympmonochrome) and broad band (Fasymp). The figures of merit with these respective names are
defined in `fom.py` and the manuscript. We used 200 cores with a maximum optimization runtime of 1440 minutes 
to obtain these results. The best single-wavelength and broadband gratings we found after curating the 400 
optima have parameters stored in the `optimised_parameters.txt` file.

### Dynamics
Once an optimized grating has been found, the dynamics simulations are performed from this folder. The first
step is to precompute Qpr (and Qpr wavelength, angle derivatives) over a given wavelength range using the
`generate_lookup_table.py` and `generate_lsa_lookup_table.py` modules. The latter is the same as the former,
but only generates the efficiencies in the linear stability analysis (lsa) regime, where the incident light 
angle (delta') is zero. With the lookup tables, run `Dynamics_integrator.py` to integrate the equations of 
motion for the chosen grating. The nonlinear lookup table is used in the nonlinear dynamics run and the linear
lookup table is used in the linear dynamics run.

`specrel.py` and `forces.py` define special-
relativity and frame-M force functions, respectively, in the particular case of the planar sail geometry
described in the manuscript. 

The `dynplot.py` module contains functions for extracting the dynamics results and plotting the sail 
coordinates in a subplots grid. These functions are used in `Dynamics_results.ipynb`.

The `Data` folder has the lookup table data for the broad band (Fasymp) optimized grating. The raw dynamics
data was not stored here due to the large file size, but can be computed using the saved optimization data
files and lookup table data files.