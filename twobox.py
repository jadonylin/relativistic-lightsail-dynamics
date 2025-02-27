"""
A class to create and simulate a TwoBox grating, storing all grating parameters and hyperparameters.

Contains plotting methods to show: grating permittivity profile, spectra, fields and angle dependence.

TODO: Move functions containing intensity, speed of light and mass to a separate module. These functions
rely on parameters that are not relevant to the grating simulation and should be kept separate. This module
should only contain bigrating simulation functions, enabling the user to easily implement their own figures
of merit in a separate module without worrying about the grating simulation.
"""

# IMPORTS ###########################################################################################################################################################################
import adaptive as adp  # TODO: remove once wavelength-range-dependent functions are moved out of twobox.py

import autograd.numpy as npa
from autograd import grad, jacobian
from autograd.scipy.special import erf as autograd_erf
from autograd.numpy import linalg as npaLA

from parameters import Parameters
from parameters import D1_ND
I0, L, m, c = Parameters()

import grcwa
grcwa.set_backend('autograd')

import matplotlib.pyplot as plt
from matplotlib.ticker import Locator
from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.rcParams['figure.figsize'] = [15, 7.5] # change inline figure size
# plt.rcParams["font.family"] = "Helvetica"
LINE_WIDTH = 2.2
SMALL_SIZE = 16
MEDIUM_SIZE = 18
BIGGER_SIZE = 20
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

import numpy as np
from numpy import *

import os
os.environ["OMP_NUM_THREADS"] = "1" 
os.environ["OPENBLAS_NUM_THREADS"] = "1" 
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 


## Minor ticks on symlog plots ##
# From: https://stackoverflow.com/questions/20470892/how-to-place-minor-ticks-on-symlog-scale
class MinorSymLogLocator(Locator):
    """
    Dynamically find minor tick positions based on the positions of
    major ticks for a symlog scaling.
    """
    def __init__(self, linthresh):
        """
        Ticks will be placed between the major ticks.
        The placement is linear for x between -linthresh and linthresh,
        otherwise its logarithmically
        """
        self.linthresh = linthresh

    def __call__(self):
        'Return the locations of the ticks'
        majorlocs = self.axis.get_majorticklocs()

        # iterate through minor locs
        minorlocs = []

        # handle the lowest part
        for i in range(1, len(majorlocs)):
            majorstep = majorlocs[i] - majorlocs[i-1]
            if abs(majorlocs[i-1] + majorstep/2) < self.linthresh:
                ndivs = 10
            else:
                ndivs = 9
            minorstep = majorstep / ndivs
            locs = np.arange(majorlocs[i-1], majorlocs[i], minorstep)[1:]
            minorlocs.extend(locs)

        return self.raise_if_exceeds(np.array(minorlocs))

    def tick_values(self, vmin, vmax):
        raise NotImplementedError('Cannot get tick locations for a '
                                  '%s type.' % type(self))



# Softmax needed for backpropagation by smoothing out the grating unit cell construction
def softmax(sigma,p):
    e_x = npa.exp(sigma*(p - npa.max(p)))
    return e_x/npa.sum(e_x)


class TwoBox:
    """
    A TwoBox grating is a grating with two "boxes" (dielectric squares/resonators) in the unit cell. 

    Uses GRCWA library to simulate the grating.
    Simulation is re-run if you change instance variables. 
    All physical lengths pertaining to the grating are normalised by the excitation/laser wavelength.

    Attributes
    ----------
    grating_pitch   :   A float for the grating pitch/period 
    grating_depth   :   A float for the grating layer depth/height/thickness 
    box1_width      :   A float for the left box/resonator width
    box2_width      :   A float for the right box/resonator width
    box_centre_dist :   A float for the distance between the box centres
    box1_eps        :   A float for the left box permittivity
    box2_eps        :   A float for the right box permittivity
    gaussian_width  :   A float for the Gaussian beam width
    substrate_depth :   A float for the substrate layer depth/height/thickness
    substrate_eps   :   A float for the substrate permittiivty
    wavelength      :   A float for the excitation-plane-wave wavelength (laser-frame wavelength)
    angle           :   A float for the excitation-plane-wave angle
    Nx              :   An integer for the number of grid points in the unit cell
    nG              :   An integer for the number of Fourier components used in the RCWA simulation
    Qabs            :   A float for the relaxation parameter, determining the strength of the imaginary frequency and thus smoothness of resonances
    """

    def __init__(self, grating_pitch, grating_depth, box1_width, box2_width, box_centre_dist, box1_eps, box2_eps, 
                 gaussian_width, substrate_depth, substrate_eps, 
                 wavelength: float=1., angle: float=0.,
                 Nx: float=1000, nG: int=25, Qabs: float=np.inf) -> None:

        self.grating_pitch = grating_pitch 
        self.grating_depth = grating_depth 
        self.box1_width = box1_width
        self.box2_width = box2_width
        self.box_centre_dist = box_centre_dist
        self.box1_eps = box1_eps
        self.box2_eps = box2_eps
        
        self.gaussian_width = gaussian_width
        self.substrate_depth = substrate_depth
        self.substrate_eps = substrate_eps
        
        self.wavelength = wavelength
        self.angle = angle

        self.Nx = Nx
        self.Ny = 1  # 1D grating simulation, only one grid in the y-direction (transverse to the 1D periodicity)
        self.nG = nG
        self.Qabs = Qabs


        self.invert_unit_cell = False

        self.box_params = [self.box1_width, self.box2_width, self.box_centre_dist, self.box1_eps, self.box2_eps, self.gaussian_width, self.substrate_depth, self.substrate_eps]
        self.params = [self.grating_pitch, self.grating_depth] + self.box_params
        
        self.init_RCWA()
  
    def build_grating(self):
        """
        Build the grating permittivity grid as an array of permittivities based on initialised box parameters. 

        Does not account for boundary permittivities in the finite grid, so is not correctly differentiable by autograd.
        You can still take the gradient of build_grating, but the results may not be consistent with finite difference
        for a variety of grating cases.
        """

        Lam = self.grating_pitch
        w1 = self.box1_width
        w2 = self.box2_width
        bcd = self.box_centre_dist
        x1 = w1/2 + 0.02*Lam  # box1 centre location (offset to avoid left box left edge clipping)    
        x2 = x1 + bcd  # box2 centre location
        
        x = npa.linspace(0,Lam,self.Nx)
        idx_in_box1 = abs(x - x1) <= w1/2
        idx_in_box2 = abs(x - x2) <= w2/2
        
        # Build grating by looping across the unit cell grids instead of using index assignment to make build_grating 
        # autograd differentiable.
        grating = [] 
        for grid_idx in range(0,self.Nx):
            if (idx_in_box1[grid_idx] and idx_in_box2[grid_idx]) or idx_in_box1[grid_idx]:  # Overrides box 2 with box 1 if overlapping
                grating.append(self.box1_eps) 
            elif idx_in_box2[grid_idx]:
                grating.append(self.box2_eps)
            else:  # assumes vacuum permittivity outside the boxes
                grating.append(1)
        
        if self.invert_unit_cell:
            self.grating_grid = npa.array(grating)[::-1]
        else:
            self.grating_grid = npa.array(grating)

        return npa.array(grating)

    def build_grating_gradable(self, sigma: float=100.):
        """
        Build the grating permittivity grid as an array of permittivities based on initialised box parameters. 
        
        Since GRCWA is grid-based, continuous changes in box widths or positions must be handled carefully. 
        Here, permittivities are chosen continuously using a softmax probability weighting depending on how 
        far away each grid is from the centre of the boxes. Softmax ensures this array of permittivities is 
        autograd differentiable. A consequence of the softmax is that the boxes are smoother than they should 
        be, with "smoothness" increasing with the temperature parameter 1/sigma.

        Builds box1 as far to the left in the unit cell as possible then fits box2 afterwards. This ensures 
        that large boxes (relative to the grating pitch) can fit inside the unit cell.

        TODO: handle the case where the boxes are too large to fit in the unit cell. Shouldn't necessarily
        throw an error because the optimiser may sometimes step into this region before stepping out.

        Parameters
        sigma :   Softmax inverse temperature, i.e. inverse smoothing factor. Smaller means smoother grating.
        """
        
        Lam = self.grating_pitch
        w1 = self.box1_width
        w2 = self.box2_width
        bcd = self.box_centre_dist
        x1 = w1/2 + 0.02*Lam  # box1 centre location (offset to avoid left box left edge clipping)    
        x2 = x1 + bcd  # box2 centre location    
        eb1 = self.box1_eps
        eb2 = self.box2_eps

        box_separation = bcd - (w1 + w2)/2
        boxes_midpoint = (x1 + w1/2 + x2 - w2/2)/2

        dx = Lam/self.Nx  # grid spacing
        grid_left_boundaries = npa.linspace(0,Lam-dx,self.Nx)  # does not include x = Lam boundary
        # In this formulation, grid numbers 0, 1, ..., Nx-1 refer to the left boundaries (consistent with x position from 0 to 1*pitch)
        box1_left_boundary = x1-w1/2
        box2_right_boundary = x2+w2/2

        # Build grating by looping across the unit cell grids instead of using index assignment to make build_grating 
        # autograd differentiable.
        grating = [] 
        for grid_left_boundary in grid_left_boundaries:
            # These floats measure how much the current grid fits each condition
            grid_in_box1 = w1/2 - npa.abs(grid_left_boundary-x1)
            grid_left_of_box1 = box1_left_boundary - grid_left_boundary
            grid_in_box2 = w2/2 - npa.abs(grid_left_boundary-x2)
            grid_between_boxes = box_separation/2 - npa.abs(grid_left_boundary - boxes_midpoint)
            grid_right_of_box2 = grid_left_boundary - box2_right_boundary 

            conditions = npa.array([grid_in_box1, grid_in_box2, grid_left_of_box1, grid_between_boxes, grid_right_of_box2])
            returns = npa.array([eb1, eb2, 1, 1, 1])
            
            probs = softmax(sigma,conditions)
            eps = npa.sum(probs*returns)

            grating.append(eps)

        if self.invert_unit_cell:
            self.grating_grid = npa.array(grating)[::-1]
        else:
            self.grating_grid = npa.array(grating)

        return npa.array(grating)


    def init_RCWA(self):
        """
        Create GRWCA object for the twobox with the initialised parameters.
        """

        # To simulate a 1D grating rather than 2D PhC, take a small periodicity in the y-direction (L2). 
        # Note: As mentioned in GRCWA documentation (https://github.com/weiliangjinca/grcwa), can only differentiate 
        # wrt photonic crystal period if the ratio of periodicities in the two in-plane directions (x and y) is fixed. 
        # GRCWA encodes this condition by scaling both (reciprocal) lattice vectors after they've been created in the 
        # kbloch.py module. Hence, set unity grating vector here and use Pscale kwarg in Init_Setup() to scale the period 
        # accordingly.

        dy = 1e-4 
        L1 = [1.,0]
        L2 = [0,dy] 

        freq = 1/self.wavelength  # frequency is 1/wavelength when c = 1
        freqcmp = freq*(1+1j/2/self.Qabs)

        theta = self.angle # radians
        phi = 0.

        obj = grcwa.obj(self.nG,L1,L2,freqcmp,theta,phi,verbose=0)


        eps_vacuum = 1
        vacuum_depth = self.wavelength

        obj.Add_LayerUniform(vacuum_depth,eps_vacuum)
        obj.Add_LayerGrid(self.grating_depth,self.Nx,self.Ny)
        if self.substrate_eps != 0:
            obj.Add_LayerUniform(self.substrate_depth,self.substrate_eps)
        obj.Add_LayerUniform(vacuum_depth,eps_vacuum)
        obj.Init_Setup(Pscale=self.grating_pitch)

        # TODO: re-building the grating every time we calculate diffraction efficiencies is inefficient 
        # because changes to parameters such as wavelength do not change the grating parameters.
        self.build_grating_gradable()  # update twobox whenever user changes box parameters
        obj.GridLayer_geteps(self.grating_grid)


        planewave={'p_amp':0,'s_amp':1,'p_phase':0,'s_phase':0}
        obj.MakeExcitationPlanewave(planewave['p_amp'],planewave['p_phase'],planewave['s_amp'],planewave['s_phase'],order = 0)

        self.RCWA = obj
        return obj


    def eff(self):
        """
        Calculates -1 <= m <= 1 reflection/transmission efficiencies for the twobox.
        """
        self.init_RCWA()
        R_byorder,T_byorder = self.RCWA.RT_Solve(normalize=1, byorder=1)
        Fourier_orders = self.RCWA.G

        Rs = []
        Ts = []
        RT_orders = [-1,0,1]
        # Have to use append method on lists rather than index assignment to make this autograd differentiable
        for order in RT_orders:
            Rs.append(npa.sum(R_byorder[Fourier_orders[:,0]==order]))
            Ts.append(npa.sum(T_byorder[Fourier_orders[:,0]==order]))

        return Rs,Ts
    
    def Q(self):
        """
        Calculate efficiency factors Q_{pr,j}'(delta', lambda') for j = 1, 2
        """
        r,t = self.eff()
        
        def diffraction_angle(m):
            """
            Calculate the diffraction angle for a given diffraction order m, if it exists.
            """
            sin_delta_m = npa.sin(self.angle) + m*self.wavelength/self.grating_pitch
            if abs(sin_delta_m)>=1:
                delta_m = "no_diffraction_order"
            else:
                delta_m = npa.arcsin(sin_delta_m)
            return delta_m
        Q1 = 0
        Q2 = 0
        orders = [-1,0,1]
        for m in range(len(orders)):
            delta_m = diffraction_angle(orders[m])
            if isinstance(delta_m,str):  # Q_{pr,j}' is unchanged by evanescent orders
                Q1 = Q1 + 0
                Q2 = Q2 + 0
            else:
                Q1 = Q1 + r[m]*(1+npa.cos(self.angle+delta_m)) + t[m]*(1-npa.cos(delta_m-self.angle))
                Q2 = Q2 + r[m]*npa.sin(self.angle+delta_m) + t[m]*npa.sin(delta_m-self.angle)
        Q1 =  npa.cos(self.angle)*Q1
        Q2 = -npa.cos(self.angle)*Q2
        
        return npa.array([Q1, Q2])


    def return_Qs(self, h_angle, h_wavelength):
        """
        Calculate efficiency factors, and their derivatives using finite differences.
        """
        
        # Save user-initialised twobox variables
        input_angle = self.angle
        input_wavelength = self.wavelength
        
        Q1,Q2 = self.Q()

        
        self.angle = input_angle - h_angle
        Q1_back_angle,Q2_back_angle = self.Q()
        self.angle = input_angle + h_angle
        Q1_forwards_angle,Q2_forwards_angle = self.Q()
        self.angle = input_angle


        self.wavelength = input_wavelength - h_wavelength
        Q1_back_wavelength,Q2_back_wavelength = self.Q()
        self.wavelength = input_wavelength + h_wavelength
        Q1_forwards_wavelength,Q2_forwards_wavelength = self.Q()


        PD_Q1_angle = (Q1_forwards_angle - Q1_back_angle) / (2*h_angle)
        PD_Q2_angle = (Q2_forwards_angle - Q2_back_angle) / (2*h_angle)
        
        PD_Q1_wavelength = (Q1_forwards_wavelength - Q1_back_wavelength) / (2*h_wavelength)
        PD_Q2_wavelength = (Q2_forwards_wavelength - Q2_back_wavelength) / (2*h_wavelength)


        # Restore user-initialised twobox variables
        self.angle = input_angle
        self.wavelength = input_wavelength

        return Q1, Q2, PD_Q1_angle, PD_Q2_angle, PD_Q1_wavelength, PD_Q2_wavelength


    def return_Qs_auto(self, return_Q: bool=True):
        """
        Calculate efficiency factors, and their derivatives using automatic differentiation.
        """

        # Save user-initialised twobox variables
        input_angle = self.angle
        input_wavelength = self.wavelength
        
        Q1,Q2 = self.Q()


        def Q_both(params):
            angle, wavelength = params
            self.angle = angle
            self.wavelength = wavelength
            return self.Q()

        params = npa.array([input_angle, input_wavelength])
        Q_jacobian = jacobian(Q_both)(params)

        PD_Q1_angle = Q_jacobian[0][0]
        PD_Q2_angle = Q_jacobian[1][0]

        PD_Q1_wavelength = Q_jacobian[0][1]
        PD_Q2_wavelength = Q_jacobian[1][1]


        # Restore user-initialised twobox variables
        self.angle = input_angle
        self.wavelength = input_wavelength

        if return_Q:
            return Q1, Q2, PD_Q1_angle, PD_Q2_angle, PD_Q1_wavelength, PD_Q2_wavelength
        else:
            return PD_Q1_angle, PD_Q2_angle, PD_Q1_wavelength, PD_Q2_wavelength


    def grad2_Q(self, method_one, method_two, param_one, param_two, h_angle, h_wavelength):
        """
        TODO: update function documentation
        
        Parameters
        ----------
        method_one :   derivative method applied first - "grad" or "finite"
        method_two :   derivative method applied last - "grad" or "finite"
        param_one  :   variable for derivative method one - "angle" or "wavelength
        param_two  :   variable for derivative method two - "angle" or "wavelength
        h_angle    :   angular step size 
        h_angle    :   wavelength step size 
        
        Returns
        -------
        d^2 Q1/ d(), d^2 Q2/ d()
        """
        allowed_methods = ("grad", "finite")
        if (method_one or method_two) not in allowed_methods:
            invalid_quantity_message = f"Invalid derivative method. Allowed quantities are: {allowed_methods}"
            raise ValueError(invalid_quantity_message)
        allowed_variables = ("angle", "wavelength")
        if (param_one or param_two) not in allowed_variables:
            invalid_quantity_message = f"Invalid derivative method. Allowed quantities are: {allowed_variables}"
            raise ValueError(invalid_quantity_message)

        input_angle = self.angle
        input_wavelength = self.wavelength
        
        def restore():
            self.angle = input_angle
            self.wavelength = input_wavelength

        ## First call
        def first(angle, wavelength, method_one):
            
            if param_one == "angle":
                var = 0
                input = angle
                h = h_angle
                def backwards():
                    self.angle = input - h
                    self.wavelength = wavelength
                def forwards():
                    self.angle = input + h
                    self.wavelength = wavelength
            if param_one == "wavelength":
                var = 1
                input = wavelength
                h = h_wavelength
                def backwards():
                    self.angle = angle
                    self.wavelength = input - h
                def forwards():
                    self.angle = angle
                    self.wavelength = input + h
            
            if method_one=="finite":    
                
                ## Backwards
                backwards()
                Q_ = self.Q()
                Q1_back = Q_[0]; Q2_back = Q_[1]
    
                ## Forwards
                forwards()
                Q_ = self.Q()
                Q1_forwards = Q_[0]; Q2_forwards = Q_[1]

                ########### Derivatives
                PD_Q1 = (Q1_forwards - Q1_back) / (2 * h)
                PD_Q2 = (Q2_forwards - Q2_back) / (2 * h)

                self.angle = angle
                self.wavelength = wavelength

                return npa.array( [PD_Q1, PD_Q2] )

            if method_one=="grad":
                def Q_params(angle, wavelength):
                    self.angle = angle
                    self.wavelength = wavelength
                    return self.Q()
                
                PD = jacobian(Q_params, argnum = var)
                PD_value = PD(angle, wavelength)

                self.angle = angle
                self.wavelength = wavelength

                return PD_value

        ## Second call (+ composition)
        def second(method_two):
            if param_two == "angle":
                var = 0
                input = input_angle
                h = h_angle
                def backwards():
                    angle = input - h
                    wavelength = input_wavelength
                    return angle, wavelength
                def forwards():
                    angle = input + h
                    wavelength = input_wavelength
                    return angle, wavelength
            
            if param_two == "wavelength":
                var = 1
                input = input_wavelength
                h = h_wavelength
                def backwards():
                    angle = input_angle
                    wavelength = input - h
                    return angle, wavelength
                def forwards():
                    angle = input_angle
                    wavelength = input + h
                    return angle, wavelength

            if method_two == "finite":

                ## Backwards
                angle, wavelength = backwards()
                Q_ = first(angle, wavelength, method_one)
                Q1_back = Q_[0]; Q2_back = Q_[1]

                ## Forwards
                angle, wavelength = forwards()
                Q_ = first(angle, wavelength, method_one)
                Q1_forwards = Q_[0]; Q2_forwards = Q_[1]

                ########### Derivatives
                restore()
                PD_Q1 = (Q1_forwards - Q1_back) / (2 * h)
                PD_Q2 = (Q2_forwards - Q2_back) / (2 * h)

                return npa.array( [PD_Q1, PD_Q2] )

            if method_two == "grad":
                restore()

                def fun(angle, wavelength):
                    return first(angle, wavelength, method_one)
                
                PD2 = jacobian(fun, argnum = var)
                PD2_value = PD2(input_angle, input_wavelength)
                restore()
                return PD2_value

        return second(method_two)

    ### 3rd derivative
    def grad3_Q(self, method_one, method_two, method_three,
                param_one, param_two, param_three, 
                h_one, h_two, h_three):
        """
        TODO: update function documentation
        
        ## Inputs
        method_one: derivative method applied first - "grad" or "finite"
        method_two: derivative method applied last - "grad" or "finite"
        param_one: variable for derivative method one - "angle" or "wavelength
        param_two: variable for derivative method two - "angle" or "wavelength
        h_one: method_one step size 
        h_two: method_two step size 
        h_three: method_three step size 
        ## Outputs
        d^2 Q1/ d(), d^2 Q2/ d()
        """
        allowed_methods = ("grad", "finite")
        if (method_one or method_two) not in allowed_methods:
            invalid_quantity_message = f"Invalid derivative method. Allowed quantities are: {allowed_methods}"
            raise ValueError(invalid_quantity_message)
        allowed_variables = ("angle", "wavelength")
        if (param_one or param_two) not in allowed_variables:
            invalid_quantity_message = f"Invalid derivative method. Allowed quantities are: {allowed_variables}"
            raise ValueError(invalid_quantity_message)

        input_angle = self.angle
        input_wavelength = self.wavelength

        def restore( angle, wavelength ):
            self.angle = angle
            self.wavelength = wavelength

        ## First call
        def first(angle, wavelength, method_one):
            
            if param_one == "angle":
                var = 0
                input = angle
                h = h_one
                def backwards():
                    self.angle = input - h
                    self.wavelength = wavelength
                def forwards():
                    self.angle = input + h
                    self.wavelength = wavelength
            if param_one == "wavelength":
                var = 1
                input = wavelength
                h = h_one
                def backwards():
                    self.angle = angle
                    self.wavelength = input - h
                def forwards():
                    self.angle = angle
                    self.wavelength = input + h
            
            if method_one=="finite":    
                
                ## Backwards
                backwards()
                Q_ = self.Q()
                Q1_back = Q_[0]; Q2_back = Q_[1]
    
                ## Forwards
                forwards()
                Q_ = self.Q()
                Q1_forwards = Q_[0]; Q2_forwards = Q_[1]

                ########### Derivatives
                restore( angle, wavelength )
                PD_Q1 = (Q1_forwards - Q1_back) / (2 * h)
                PD_Q2 = (Q2_forwards - Q2_back) / (2 * h)

                return npa.array( [PD_Q1, PD_Q2] )

            if method_one=="grad":
                restore( angle, wavelength )
                def Q_params(angle, wavelength):
                    self.angle = angle
                    self.wavelength = wavelength
                    return self.Q()
                
                PD = jacobian(Q_params, argnum = var)
                PD_value = PD(angle, wavelength)

                restore( angle, wavelength )

                return PD_value

        ## Second call (+ composition)
        def second(angle2, wavelength2, method_two):

            if param_two == "angle":
                var = 0
                input = angle2
                h = h_two
                def backwards():
                    angle = input - h
                    wavelength = wavelength2
                    return angle, wavelength
                def forwards():
                    angle = input + h
                    wavelength = wavelength2
                    return angle, wavelength          
            if param_two == "wavelength":
                var = 1
                input = wavelength2
                h = h_two
                def backwards():
                    angle = angle2
                    wavelength = input - h
                    return angle, wavelength
                def forwards():
                    angle = angle2
                    wavelength = input + h
                    return angle, wavelength

            if method_two == "finite":

                ## Backwards
                angle, wavelength = backwards()
                Q_ = first(angle, wavelength, method_one)
                Q1_back = Q_[0]; Q2_back = Q_[1]

                restore( angle2, wavelength2 )

                ## Forwards
                angle, wavelength = forwards()
                Q_ = first(angle, wavelength, method_one)
                Q1_forwards = Q_[0]; Q2_forwards = Q_[1]

                ########### Derivatives
                restore( angle2, wavelength2 )
                PD_Q1 = (Q1_forwards - Q1_back) / (2 * h)
                PD_Q2 = (Q2_forwards - Q2_back) / (2 * h)

                return npa.array( [PD_Q1, PD_Q2] )

            if method_two == "grad":
                restore( angle2, wavelength2 )

                def fun(angle2, wavelength2):
                    return first(angle2, wavelength2, method_one)
                
                PD2 = jacobian(fun, argnum = var)
                PD2_value = PD2(angle2, wavelength2)
                
                restore( angle2, wavelength2 )
                
                return PD2_value

        ## Third call (+ composition)
        def third(method_three):
            if param_three == "angle":
                var = 0
                input = input_angle
                h = h_three
                def backwards():
                    angle = input - h
                    wavelength = input_wavelength
                    return angle, wavelength
                def forwards():
                    angle = input + h
                    wavelength = input_wavelength
                    return angle, wavelength           
            if param_three == "wavelength":
                var = 1
                input = input_wavelength
                h = h_three
                def backwards():
                    angle = input_angle
                    wavelength = input - h
                    return angle, wavelength
                def forwards():
                    angle = input_angle
                    wavelength = input + h
                    return angle, wavelength

            if method_three == "finite":

                ## Backwards
                angle, wavelength = backwards()
                Q_ = second(angle, wavelength, method_two)
                Q1_back = Q_[0]; Q2_back = Q_[1]

                restore( input_angle, input_wavelength)

                ## Forwards
                angle, wavelength = forwards()
                Q_ = second(angle, wavelength, method_two)
                Q1_forwards = Q_[0]; Q2_forwards = Q_[1]

                ########### Derivatives
                restore( input_angle, input_wavelength)
                PD_Q1 = (Q1_forwards - Q1_back) / (2 * h)
                PD_Q2 = (Q2_forwards - Q2_back) / (2 * h)

                return npa.array( [PD_Q1, PD_Q2] )

            if method_three == "grad":
                restore( input_angle, input_wavelength)

                def fun(angle, wavelength):
                    return second(angle, wavelength, method_two)
                
                PD3 = jacobian(fun, argnum = var)
                PD3_value = PD3(input_angle, input_wavelength)
                
                restore( input_angle, input_wavelength)
                
                return PD3_value

        return third(method_three)


    def FoM(self, I:float=1e9, grad_method: str="finite") -> float:
        """
        Calculate the grating single-wavelength figure of merit FD.

        Parameters
        ----------
        I           :   Laser intensity
        grad_method :   Method to calculate gradient ("finite","grad"). Must be "finite" for optimisation
        
        Returns
        -------
        FD :   Figure of merit
        """
        eigReal, eigImag = self.Eigs(I=I, m=m, c1=c, grad_method=grad_method, check_det=True, return_vec=False)

        def unique_filled(x, filled_value):
            """
            Parameters
            ----------
            x            :   4d array
            filled_value :   Float to fill remaining entries in unique_values

            Returns
            -------
            unique_values :   Unique contents of x, with remaining entries filled by filled_value
            """
            
            # Sort array to ensure differentiability
            sorted_x = npa.sort(x.flatten())
            unique_values = sorted_x[npa.concatenate(([True], npa.diff(sorted_x) != 0))]

            # Append filled_value as needed
            k = len(unique_values)
            for i in range(4-k):
                unique_values = npa.append(unique_values,filled_value)

            return unique_values
    
        # Reward all Re(eig) being negative
        # NOTE: In the following penalty and reward terms, all operations must be done element-wise to avoid 
        #       "RuntimeWarning: invalid value encountered in divide" during optimisation
        # TODO: Determine why we can't use npa functions here

        # Reward all Re(eig) being negative
        eig_real_unique     =   unique_filled(eigReal, -1)
        eig_real_neg_unique =   npa.minimum(0., eig_real_unique)
        func_real_neg_array =   npa.power(eig_real_neg_unique, 2)
        func_real_neg       =   func_real_neg_array[0] * func_real_neg_array[1] * func_real_neg_array[2] * func_real_neg_array[3]
        # func_real_neg       =   npa.prod(func_real_neg_array)  

        # Remove Re(eig)<0 contribution if no restoring behaviour
        # log(1+x^2) chosen as a smooth approximation to the Heaviside step function
        func_imag_array     =   npa.log(1 + npa.power(eigImag,2))
        func_imag           =   func_imag_array[0] * func_imag_array[1] * func_imag_array[2] * func_imag_array[3]
        # func_imag           =   npa.prod(func_imag_array)

        # Penalise mixed positive and negative Re(eig)
        real_unique_0       =   unique_filled(eigReal, 0.)
        neg_array           =   npa.power(npa.minimum(0.,real_unique_0), 2)
        pos_array           =   npa.power(npa.maximum(0.,real_unique_0), 2)
        # penalty             =   npa.sum(neg_array) * npa.sum(pos_array)
        neg_sum             =   neg_array[0] + neg_array[1] + neg_array[2] + neg_array[3]
        pos_sum             =   pos_array[0] + pos_array[1] + pos_array[2] + pos_array[3]
        penalty             =   neg_sum * pos_sum

        # Penalise all positive Re(eig)
        real_unique_1       =   unique_filled(eigReal, 1)
        all_pos_array       =   npa.power(npa.maximum(0.,real_unique_1), 2)
        penalty2            =   all_pos_array[0] * all_pos_array[1] * all_pos_array[2] * all_pos_array[3]
        # penalty2            =   npa.prod(all_pos_array)


        FD = func_real_neg * func_imag - penalty - penalty2
        return FD


    def Eigs(self, I: float=10e9, m: float=1/1000, c1:float=299792458, grad_method: str='finite', check_det: bool = False, return_vec: bool = False):
        """
        Calculate eigenvalues of Jacobian matrix at equilibrium

        Parameters
        ----------
        I           :   Laser intensity
        m           :   Spacecraft mass (sail membrane + payload)
        c1          :   speed of light  # TODO: why is this a parameter?
        grad_method :   Method to calculate gradient ("finite","grad"). Must be "finite" for optimisation
        check_det   :   If true, check if Jacobian determinant is zero. If Jacobian is zero, print the twobox parameters
        return_vec  :   If true, return eigenvectors as well as eigenvalues
        
        
        Returns
        -------
        FD :   Figure of merit
        """
        
        if grad_method == 'finite':
            # For optimisation, need to use finite differences
            # Approximately optimal step size is 10^-6.5 for both angle and wavelength
            h_angle = 10**(-6.5)
            h_wavelength = 10**(-6.5)
            Q1R, Q2R, dQ1ddeltaR, dQ2ddeltaR, dQ1dlambdaR, dQ2dlambdaR = self.return_Qs(h_angle, h_wavelength)
        if grad_method == "grad":
            Q1R, Q2R, dQ1ddeltaR, dQ2ddeltaR, dQ1dlambdaR, dQ2dlambdaR = self.return_Qs_auto(return_Q=True)

        w = self.gaussian_width
        w_bar = w/L  # width normalised to total grating length
        lam = self.wavelength 

        # Convert velocity factors to wavelength factors
        # TODO: may need to change these factors to account for non-unity starting wavelengths
        D = 1/lam  # Doppler factor assuming starting wavelength is 1
        g = (npa.power(lam,2) + 1)/(2*lam)  # Lorentz factor
   
        # Lightsail reflection-symmetry conditions
        Q1L = Q1R;   Q2L = -Q2R;   
        dQ1ddeltaL  = -dQ1ddeltaR;    dQ2ddeltaL  = dQ2ddeltaR
        dQ1dlambdaL = dQ1dlambdaR;    dQ2dlambdaL = -dQ2dlambdaR

        # y acceleration terms
        # NOTE: derivatives with respect to lambda differ from derivatives with respect to frequency offset, the latter
        # being presented in Liam's thesis
        fy_y    = - D**2 * I/(m*c1) * (Q2R - Q2L) * (1 - npa.exp(-1/(2*w_bar**2)))
        fy_phi  = - D**2 * I/(m*c1) * (dQ2ddeltaR + dQ2ddeltaL) * w/2 * npa.sqrt(np.pi/2) * autograd_erf(1/(w_bar*npa.sqrt(2)))
        fy_vy   = - D**2 * I/(m*c1) * 1/c1 * (D+1)/(D*(g+1)) * (Q1R + Q1L + dQ2ddeltaR + dQ2ddeltaL) * w/2 * npa.sqrt(np.pi/2) * autograd_erf(1/(w_bar*npa.sqrt(2)))
        fy_vphi =   D**2 * I/(m*c1) * 1/c1 * (2*(Q2R - Q2L) - lam*(dQ2dlambdaR - dQ2dlambdaL)) * (w/2)**2 * (1 - npa.exp(-1/(2*w_bar**2)))

        # phi acceleration terms
        # TODO: generalise for non-flat-geometry moments of inertia
        fphi_y    =  D**2 * 12*I/(m*c1*L**2) * (Q1R + Q1L) * (w/2*npa.sqrt(np.pi/2) * autograd_erf(1/(w_bar*npa.sqrt(2))) - L/2*npa.exp(-1/(2*w_bar**2))) 
        fphi_phi  =  D**2 * 12*I/(m*c1*L**2) * (dQ1ddeltaR - dQ1ddeltaL - (Q2R - Q2L)) * (w/2)**2 * (1 - npa.exp(-1/(2*w_bar**2)))
        fphi_vy   =  D**2 * 12*I/(m*c1*L**2) * 1/c1 * (D+1)/(D*(g+1)) * (dQ1ddeltaR - dQ1ddeltaL - (Q2R - Q2L)) * (w/2)**2 * (1 - npa.exp(-1/(2*w_bar**2)))
        fphi_vphi = -D**2 * 12*I/(m*c1*L**2) * 1/c1 * (2*(Q1R + Q1L) - lam*(dQ1dlambdaR + dQ1dlambdaL)) * (w/2)**2 * (w/2*npa.sqrt(np.pi/2) * autograd_erf(1/(w_bar*npa.sqrt(2))) - L/2*npa.exp(-1/(2*w_bar**2))) 

        # Build the Jacobian matrix
        J00 = fy_y;   J01 = fy_phi;   J02 = fy_vy;   J03 = fy_vphi
        J10 = fphi_y; J11 = fphi_phi; J12 = fphi_vy; J13 = fphi_vphi
        J = npa.array([[0,0,1,0],[0,0,0,1],[J00,J01,J02,J03],[J10,J11,J12,J13]])

        # For debugging non-differentiable gratings during optimisation
        if check_det:
            if npaLA.det(J)==0:
                print("Grating parameters:")
                print(self.grating_pitch)
                print(self.grating_depth)
                print(self.box1_width)
                print(self.box2_width)
                print(self.box_centre_dist)
                print(self.box1_eps)
                print(self.box2_eps)
                print(self.gaussian_width)
                print(self.substrate_depth)
                print(self.substrate_eps)

                print("lam: ", lam)
                print("\n")

        # Find the real part of eigenvalues    
        eigvalvec   = npaLA.eig(J)
        eigvals     = eigvalvec[0]
        eigReal     = npa.real(eigvals)
        eigImag     = npa.imag(eigvals)

        if return_vec:
            eigvecs = eigvalvec[1]
            return eigReal, eigImag, eigvecs
        else:
            return eigReal, eigImag

    def lsa_info(self, wavelength, I: float=0.5e9):
        """
        TODO: update function documentation

        ## Inputs
        wavelength
        I - intensity
        ## Outputs
        eff_array, restoring_array, damping_array, Re(eig)_array, Im(eig)_array
        """
        input_wavelength = self.wavelength
        self.wavelength = wavelength

        ####################################
        ## Call efficiency factors
        Q1R, Q2R, dQ1ddeltaR, dQ2ddeltaR, dQ1dlambdaR, dQ2dlambdaR = self.return_Qs_auto(return_Q=True)
        eff_array = (Q1R, Q2R, dQ1ddeltaR, dQ2ddeltaR, dQ1dlambdaR, dQ2dlambdaR)
        
        w = self.gaussian_width
        w_bar = w/L

        lam = self.wavelength 

        ## Convert velocity dependence to wavelength dependence
        D = 1/lam 
        g = (npa.power(lam,2) + 1)/(2*lam) 
   
        ## Symmetry
        Q1L = Q1R;   Q2L = -Q2R;   
        dQ1ddeltaL  = -dQ1ddeltaR;    dQ2ddeltaL  = dQ2ddeltaR
        dQ1dlambdaL = dQ1dlambdaR;    dQ2dlambdaL = -dQ2dlambdaR

        w_bar = w / L

        # y acceleration
        fy_y= -     D**2 * (I/(m*c)) *  ( Q2R - Q2L ) * ( 1 - np.exp(-1/(2*w_bar**2) ) )
        fy_phi= -   D**2 * (I/(m*c)) * ( dQ2ddeltaR + dQ2ddeltaL ) * (w/2) * np.sqrt( np.pi/2 ) * autograd_erf( 1/(w_bar*np.sqrt(2)) )
        fy_vy= -    D**2 * (I/(m*c)) * (1/c) * ( (D+1)/(D*(g+1)) ) * ( Q1R + Q1L  + dQ2ddeltaR + dQ2ddeltaL ) * (w/2) * np.sqrt( np.pi/2 ) * autograd_erf( 1/(w_bar*np.sqrt(2)) )
        fy_vphi=    D**2 * (I/(m*c)) * (1/c) * ( 2*( Q2R - Q2L ) - lam*( dQ2dlambdaR - dQ2dlambdaL ) ) * (w/2)**2 * ( 1 - np.exp( -1/(2*w_bar**2) ))

        # phi acceleration
        fphi_y=     D**2 * (12*I/( m*c*L**2)) * ( Q1R + Q1L ) * (  (w/2)*np.sqrt( np.pi/2 )  * autograd_erf( 1/(w_bar*np.sqrt(2)))  - (L/2)* np.exp( -1/(2*w_bar**2) )  ) 
        fphi_phi=   D**2 * (12*I/( m*c*L**2)) * ( dQ1ddeltaR - dQ1ddeltaL - ( Q2R - Q2L ) ) * (w/2)**2 * ( 1 - np.exp( -1/(2*w_bar**2) ))
        fphi_vy=    D**2 * (12*I/( m*c*L**2)) * (1/c) * ( (D+1)/(D*(g+1)) ) * ( dQ1ddeltaR - dQ1ddeltaL - ( Q2R - Q2L ) ) * (w/2)**2 * ( 1 - np.exp( -1/(2*w_bar**2) ))
        fphi_vphi= -D**2 * (12*I/( m*c*L**2)) * (1/c) * ( 2*( Q1R + Q1L ) - lam*( dQ1dlambdaR + dQ1dlambdaL ) ) * (w/2)**2 * (  (w/2)*np.sqrt( np.pi/2 )  * autograd_erf( 1/(w_bar*np.sqrt(2)))  - (L/2)* np.exp( -1/(2*w_bar**2) )  ) 


        ## array
        rest_array = ( fy_y,fy_phi,  fphi_y,fphi_phi )
        damp_array = ( fy_vy,fy_vphi,  fphi_vy,fphi_vphi )

        # Build the Jacobian matrix
        J00=fy_y;   J01=fy_phi;     J02=fy_vy;    J03=fy_vphi
        J10=fphi_y; J11=fphi_phi;   J12=fphi_vy;  J13=fphi_vphi
        J=npa.array([[0,0,1,0],[0,0,0,1],[J00,J01,J02,J03],[J10,J11,J12,J13]])

        # Find the real part of eigenvalues    
        eigvalvec   = npaLA.eig(J)
        eig         = eigvalvec[0]
        eigReal     = npa.real(eig)
        eigImag     = npa.imag(eig)

        # Eigenvectors
        vec = eigvalvec[1]
        vec1 = vec[:,0]
        vec2 = vec[:,1]
        vec3 = vec[:,2]
        vec4 = vec[:,3]
        vec_array = (vec1, vec2, vec3, vec4)

        ## Restore wavelength
        self.wavelength = input_wavelength
        return eff_array, rest_array, damp_array, eigReal, eigImag , vec


    def average_real_eigs(self, final_speed, goal, return_eigs:bool=False, I:float=10e9):
        """
        TODO: this function should be moved to opt.py because it depends on final sail velocity

        Calculates the average of each Re(eig) over the wavelength range. 
        
        Assumes starting wavelength = 1.

        Parameters
        ----------
        final_speed :   percentage speed of light
        goal        :   integer (number of points) or float (loss goal)
        return_eigs :   If true, return normalised eigenvalues. If false, return averaged eigenvalues
        I           :   Laser intensity
        """

        Doppler = D1_ND([final_speed/100,0])
        l_min = 1  # l = grating-frame wavelength normalised to laser-frame wavelength
        l_max = l_min/Doppler    
        l_range = (l_min, l_max)
        
        PDF_unif = 1/(l_max-l_min)  # Probability density function (PDF) for averaging

        def weighted_eig_real(l):
            self.wavelength = l
            return PDF_unif*self.Eigs(I=I, m=m, c1=c, check_det=False, return_vec=False)[0]

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

    
    def show_permittivity(self, show_analytic_box: bool=False):
        """
        Show permittivity profile for the twobox.

        Displays two panels:
            Panel 1 is the user-input permittivity profile (the boxes smoothed by build_grating_gradable())
            Panel 2 is the GRCWA-resolved permittivity profile (with precision limited by nG)

        Note: GRCWA does not always align the returned permittivity with the grid numbers, it may be displaced by several grids.
        This is not an issue because the unit cell can be arbitrarily shifted provided you maintain periodic boundary conditions.

        Parameters
        ----------
        show_analytic_box :   Show analytic (perfect rectangular) boxes on top of softmax-smoothed boxes

        Returns
        -------
        x0             :   Coordinates along the unit cell grid (0 to grating_pitch) 
        eps_array_real :   Permittivity values at x0 coordintes
        fig            :   Permittivity profile figure object
        axs            :   Permittivity profile axs object
        """ 

        self.init_RCWA()

        x0 = np.linspace(0,self.grating_pitch,self.Nx, endpoint=False)
    
        eps_array = self.RCWA.Return_eps(which_layer=1,Nx=self.Nx,Ny=self.Ny,component='xx')
        eps_array_real = eps_array.real

        # Flip displayed GRCWA permittivity profile to match ordering of input permittivity profile
        grids = np.arange(0, self.Nx, 1)
        eps_array_real = np.flip(eps_array_real)


        fig, axs = plt.subplots(2, 1, figsize=(6,10), sharex=True)

        axs[0].plot(grids,self.grating_grid)
        axs[0].set(xlabel="Grid no.", ylabel=r"$\varepsilon$")  
        axs[0].set_title(f"Input permittivity profile. nG = {self.nG}, grid points = {self.Nx}")
        axs[0].set_xlim([0-0.01*self.Nx, self.Nx-1+0.01*self.Nx])

        axs[1].plot(grids,eps_array_real)
        axs[1].set(xlabel="Grid no.", ylabel=r"$\varepsilon$")
        axs[1].set_title(f"GRCWA permittivity profile. nG = {self.nG}, grid points = {self.Nx}")

        if show_analytic_box:
            init_Nx = self.Nx
            self.Nx = 1000*self.Nx
            fine_grids = np.arange(0, self.Nx, 1)
            analytic_boxes = self.build_grating()
            axs[0].plot(fine_grids/1000,analytic_boxes)
            self.Nx = init_Nx
        
        plt.show()
        return x0, eps_array_real, fig, axs
    
    def show_angular_efficiency(self, theta_max: float=20., num_plot_points: int=100):
        """
        Show grating efficiencies as a function of excitation angle for the twobox.

        Parameters
        ----------
        theta_max       :   Plot angles up to theta_max (degrees)
        num_plot_points :   Number of points to plot

        Returns
        -------
        fig :   Angle plot figure object
        ax  :   Angle plot axs object
        """

        init_angle = self.angle  # record user-initialised angle
        inc_angles = np.pi/180*np.linspace(-theta_max, theta_max, num_plot_points) 


        efficiencies = np.zeros((6,num_plot_points), dtype=float)
        for idx, theta in enumerate(inc_angles):
            self.angle = theta
            Rs,Ts = self.eff()
            efficiencies[:3,idx] = Rs
            efficiencies[3:,idx] = Ts
        self.angle = init_angle  # reset user-initialised angle


        fig, ax = plt.subplots(1)           

        inc_angles = inc_angles*180/np.pi        

        # -1 order
        ax.plot(inc_angles, efficiencies[0], color=(0.7, 0, 0), linestyle='-', label="$r_{-1}$", lw = LINE_WIDTH)
        ax.plot(inc_angles, efficiencies[3], color=(0.7, 0, 0), linestyle='-.', label="$t_{-1}$", lw = LINE_WIDTH) 

        # 0 order
        ax.plot(inc_angles, efficiencies[1], color='0.4', linestyle='-', label="$r_0$", lw = LINE_WIDTH)
        ax.plot(inc_angles, efficiencies[4], color='0.4', linestyle='-.', label="$t_0$", lw = LINE_WIDTH) 

        # 1 order
        ax.plot(inc_angles, efficiencies[2], color=(0, 0, 0.7), linestyle='-', label="$r_1$", lw = LINE_WIDTH)
        ax.plot(inc_angles, efficiencies[5], color=(0, 0, 0.7), linestyle='-.', label="$t_1$", lw = LINE_WIDTH) 

        # Verify energy conservation by showing sum of efficiencies to unity
        eff_sum = np.sum(efficiencies[:6,:],axis=0)
        ax.plot(inc_angles, eff_sum, color=(0, 0.7, 0), linestyle='-', label=r"$\Sigma (r_i + t_i)$", lw = LINE_WIDTH) 

        ax.set(title=rf"$\Lambda' = {self.grating_pitch/self.wavelength:.3f}\lambda$, $h_1' = {self.grating_depth/self.wavelength:.3f}\lambda$"
            , xlabel="Incident angle (°)"
            , ylabel="Efficiency")

        leg = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        frame = leg.get_frame()
        frame.set_edgecolor('black')

        ax.set_xlim([-theta_max, theta_max])
        ax.set_ylim([-0.01, 1.01])  # displace lower bound from zero to highlight the transition to evanescence
        cm_to_inch = 0.393701
        fig_width = 20.85*cm_to_inch
        fig_height = 17.6*cm_to_inch
        fig.set_size_inches(fig_width/1.2, fig_height/1.2)
        
        return fig, ax

    def show_spectrum(self, angle: float=0., efficiency_quantity: str="PDr", wavelength_range: list=[1., 1.5], num_plot_points: int=200, I: float=10e9):
        """
        Show spectrum of various efficiency quantities for the twobox.

        Parameters
        ----------
        angle               :   Angle of incident plane wave excitation (radians)
        efficiency_quantity :   The efficiency quantity you want spectrum for
                                "r" - reflection, "PDr" - reflection angular derivative, 
                                "t" - transmission, "PDr" - transmission angular derivative),
                                "FoM" - single-wavelength figure of merit
        wavelength_range    :   Wavelength range to plot spectrum (same units as grating pitch)
        num_plot_points     :   Number of points to plot
        I                   :   Laser intensity

        Returns
        -------
        fig :   Spectrum figure object
        ax  :   Spectrum axs object
        """

        allowed_quantities = ("r", "t", "PDr", "PDt", "FoM")
        if efficiency_quantity not in allowed_quantities:
            invalid_quantity_message = f"Invalid efficiency quantity. Allowed quantities are: {allowed_quantities}"
            raise ValueError(invalid_quantity_message)
        

        wavelengths = np.linspace(*wavelength_range, num_plot_points)
        init_wavelength = self.wavelength  # record user-initialised wavelength
        inc_angle_deg = angle*180/np.pi


        RT_orders = [-1,0,1]
        n_orders = len(RT_orders)
        efficiencies = np.zeros((2*n_orders,num_plot_points), dtype=float)
        
        for idx, lam in enumerate(wavelengths):
            self.wavelength = lam
            
            if efficiency_quantity == "r" or efficiency_quantity == "t":
                Rs,Ts = self.eff()
                efficiencies[:n_orders,idx] = Rs
                efficiencies[n_orders:,idx] = Ts
            elif efficiency_quantity == "PDr":
                efficiencies[0,idx] = self.PDrNeg1(angle)
            elif efficiency_quantity == "PDt":
                efficiencies[0,idx] = self.PDtNeg1(angle)
            elif efficiency_quantity == "FoM":
                efficiencies[0,idx] = self.FoM(I=I, grad_method="grad")


        self.wavelength = init_wavelength


        fig, ax = plt.subplots(1)         
        p = self.grating_pitch
        ax.set_xlim(np.array(wavelength_range)/p)  # normalise wavelength to grating pitch
        legend_needed = ("r", "t")
        symlog_needed = ("PDr", "PDt")


        if efficiency_quantity == "r":
            # -1 order
            ax.plot(wavelengths/p, efficiencies[0], color=(0.7, 0, 0), linestyle='-', label="$r_{-1}'$", lw = LINE_WIDTH) 
            # 0 order
            ax.plot(wavelengths/p, efficiencies[1], color='0.4', linestyle='-', label="$r_0'$", lw = LINE_WIDTH) 
            # 1 order
            ax.plot(wavelengths/p, efficiencies[2], color=(0, 0, 0.7), linestyle='-', label="$r_1'$", lw = LINE_WIDTH) 
            ax.set_ylim([-0.01, 1.01]) 
            ylabel = rf"Reflection at $\theta' = {inc_angle_deg}°$"
        elif efficiency_quantity == "t":
            # -1 order
            ax.plot(wavelengths/p, efficiencies[0], color=(0.7, 0, 0), linestyle='-', label="$r_{-1}'$", lw = LINE_WIDTH)
            ax.plot(wavelengths/p, efficiencies[3], color=(0.7, 0, 0), linestyle='-.', label="$t_{-1}'$", lw = LINE_WIDTH)  
            # 0 order
            ax.plot(wavelengths/p, efficiencies[1], color='0.4', linestyle='-', label="$r_0'$", lw = LINE_WIDTH) 
            ax.plot(wavelengths/p, efficiencies[4], color='0.4', linestyle='-.', label="$t_0'$", lw = LINE_WIDTH) 
            # 1 order
            ax.plot(wavelengths/p, efficiencies[2], color=(0, 0, 0.7), linestyle='-', label="$r_1'$", lw = LINE_WIDTH) 
            ax.plot(wavelengths/p, efficiencies[5], color=(0, 0, 0.7), linestyle='-.', label="$t_1'$", lw = LINE_WIDTH) 
            ax.set_ylim([-0.01, 1.01]) 
            ylabel = rf"Efficiency at $\theta' = {inc_angle_deg}°$"
        elif efficiency_quantity == "PDr":
            ax.plot(wavelengths/p, efficiencies[0], color=(0.7, 0, 0), linestyle='-', lw = LINE_WIDTH) 
            ylabel = rf"$\frac{{\partial r_{{-1}}'}}{{\partial\theta'}}({inc_angle_deg}°)$"
        elif efficiency_quantity == "PDt":
            ax.plot(wavelengths/p, efficiencies[0], color=(0.7, 0, 0), linestyle='-', lw = LINE_WIDTH) 
            ylabel = rf"$\frac{{\partial t_{{-1}}'}}{{\partial\theta'}}({inc_angle_deg}°)$"

        elif efficiency_quantity=="FoM":
            ax.plot(wavelengths/p, efficiencies[0], color=(0.7, 0, 0), linestyle='-', lw = LINE_WIDTH) 
            ylabel = rf"FoM"

      
        if efficiency_quantity in legend_needed:
            leg = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            frame = leg.get_frame()
            frame.set_edgecolor('black')
        
        if efficiency_quantity in symlog_needed:
            linthr = 0.1
            ax.set_yscale("symlog", linthresh=linthr, linscale=0.4)
            ax.yaxis.set_minor_locator(MinorSymLogLocator(linthr))
        

        ax.axhline(y=0, color='black', linestyle='-', lw = '1')
        ax.tick_params(axis='both', which='both', direction='in') # ticks inside box
        ax.set(title=rf"$h_1' = {self.grating_depth/self.wavelength:.3f}\lambda_0$, $\Lambda' = {self.grating_pitch/self.wavelength:.3f}\lambda_0$", xlabel=r"$\lambda'/\Lambda'$", ylabel=ylabel)


        cm_to_inch = 0.393701
        fig_width = 20.85*cm_to_inch
        fig_height = 17.6*cm_to_inch
        fig.set_size_inches(fig_width/1.2, fig_height/1.2)
        
        return fig, ax


    def show_Eigs(self, marker: str='o', eig_real_log_axis: bool=True, eig_imag_log_axis: bool=True, wavelength_range: list=[1., 1.5], num_plot_points: int=200, I: float=10e9):
        """
        Show spectrum of various efficiency quantities for the twobox.

        Parameters
        ----------
        marker              :   Marker style passed to plt.plot()
        eig_real_log_axis   :   If true, logarithmic scale for real part of eigenvalues
        eig_imag_log_axis   :   If true, logarithmic scale for imaginary part of eigenvalues
        efficiency_quantity :   The efficiency quantity you want spectrum for
                                "r" - reflection, "PDr" - reflection angular derivative, 
                                "t" - transmission, "PDr" - transmission angular derivative),
                                "FoM" - single-wavelength figure of merit
        wavelength_range    :   Wavelength range to plot spectrum (same units as grating pitch)
        num_plot_points     :   Number of points to plot
        I                   :   Laser intensity

        Returns
        -------
        fig :   Spectrum figure object
        ax  :   Spectrum axs object
        """

        wavelengths = np.linspace(*wavelength_range, num_plot_points)
        init_wavelength = self.wavelength  # record user-initialised wavelength

        ## CALCULATE EIGS ##
        eigvals = np.zeros((4,num_plot_points), dtype=np.complex128)
        
        for idx, lam in enumerate(wavelengths):
            # Calculate eigs for each order
            self.wavelength = lam
            real, imag = self.Eigs(I=I,m=m,c1=c, grad_method="grad", check_det=False, return_vec=False)
            eigvals[:,idx] = real + 1j*imag
            
        self.wavelength = init_wavelength # restore user-initialised wavelength


        # I'm assuming the dummy subplot creates spacing between the two other subplots
        fig, (ax1, dummy, ax2) = plt.subplots(nrows=1, ncols=3, width_ratios=(1,0.1,1))
        dummy.axis('off')
        p = self.grating_pitch
        
        ax1.set_xlim(np.array(wavelength_range)/p) 
        ax2.set_xlim(np.array(wavelength_range)/p) 
        ax2.yaxis.tick_right()
        ax2.yaxis.set_label_position("right")

        colorReal = (0.7, 0, 0)
        colorImag = 'blue'
        for i in range(4):            
            ax1.plot(wavelengths/p,np.real(eigvals[i,:]), marker, markersize=0.5, markerfacecolor=colorReal, fillstyle='full',  color=colorReal)
            ax2.plot(wavelengths/p,np.imag(eigvals[i,:]), marker, markersize=0.5, markerfacecolor=colorImag, fillstyle='full',  color=colorImag)
            

        if eig_real_log_axis:
            linthr = 0.1
            ax1.set_yscale("symlog", linthresh=linthr, linscale=0.4)
            ax1.yaxis.set_minor_locator(MinorSymLogLocator(linthr))
        if eig_imag_log_axis:
            linthr = 0.1
            ax2.set_yscale("symlog", linthresh=linthr, linscale=0.4)
            ax2.yaxis.set_minor_locator(MinorSymLogLocator(linthr))

        
        ax1.axhline(y=0, color='black', linestyle='-', lw = '1')
        ax1.tick_params(axis='both', which='both', direction='in')  # ticks inside box
        # ax1.tick_params(axis='y', color=colorReal, labelcolor=colorReal)  # colored ticks
        ax1.set_ylabel(ylabel=rf"$\Re(\lambda)$")  #color=colorReal  # colored y label
        ax1.set(xlabel=r"$\lambda'/\Lambda'$")

        ax2.axhline(y=0, color='black', linestyle='-', lw = '1')
        ax2.tick_params(axis='both', which='both', direction='in')  # ticks inside box
        # ax2.tick_params(axis='y', color = colorImag, labelcolor=colorImag)  # colored ticks
        ax2.set_ylabel(ylabel=rf"$\Im(\lambda)$")  #color=colorImag  # colored y label
        ax2.set(xlabel=r"$\lambda'/\Lambda'$")

        # fig.suptitle(t=rf"$h_1' = {self.grating_depth/self.wavelength:.3f}\lambda_0$, $\Lambda' = {self.grating_pitch/self.wavelength:.3f}\lambda_0$")

        # Modify axes
        cm_to_inch = 0.393701
        fig_width = 30*cm_to_inch
        fig_height = 17.6*cm_to_inch
        fig.set_size_inches(fig_width/1.2, fig_height/1.2)

        return fig, (ax1, ax2)


    def show_depth_dependence(self, angle: float=0., efficiency_quantity: str="PDr", depth_range: list=[0., 1.], num_plot_points: int=200):
        """
        TODO: update function documentation. This function is basically unused.

        Show grating depth dependence for the twobox.

        Parameters
        ----------
        angle               :   Angle of incident plane wave excitation (radians)
        efficiency_quantity :   Which efficiency quantity you want spectrum for ("r" - reflection, "PDr" - reflection angular derivative, "t" - transmission, "PDr" - transmission angular derivative)
        depth_range         :   Depth range to plot spectrum (normalised to wavelength)
        num_plot_points     :   Number of points to plot
        """
        ### Setup ###
        allowed_quantities = ("r", "t", "PDr", "PDt")
        if efficiency_quantity not in allowed_quantities:
            invalid_quantity_message = f"Invalid efficiency quantity. Allowed quantities are: {allowed_quantities}"
            raise ValueError(invalid_quantity_message)
        
        heights = np.linspace(*depth_range, num_plot_points)
        init_depth = self.grating_depth # record user-initialised wavelength
        inc_angle_deg = angle*180/np.pi


        ## CALCULATE EFFICIENCY ##
        RT_orders = [-1,0,1]
        n_orders = len(RT_orders)
        efficiencies = np.zeros((2*n_orders,num_plot_points), dtype=float)
        # Rows of efficiencies correspond to the order of diffraction (row,order): 
        # Reflection: (0,-1), (1,0), (2,1)
        # Transmission: (3,-1), (4,0), (5,1)
        for idx, d in enumerate(heights):
            # Calculate efficiencies for each order
            self.grating_depth = d
            if efficiency_quantity == "r" or efficiency_quantity == "t":
                Rs,Ts = self.eff()
                efficiencies[:n_orders,idx] = Rs
                efficiencies[n_orders:,idx] = Ts
            elif efficiency_quantity == "PDr":
                efficiencies[0,idx] = self.PDrNeg1(angle)
            elif efficiency_quantity == "PDt":
                efficiencies[0,idx] = self.PDtNeg1(angle)
        self.grating_depth = init_depth # restore user-initialised wavelength


        ### PLOTTING ### 
        # Set up figure
        fig, ax = plt.subplots(1)         
        ax.set_xlim(depth_range)
        legend_needed = ("r", "t")
        symlog_needed = ("PDr", "PDt")

        ## Plot efficiency vs height ##
        if efficiency_quantity == "r":
            # -1 order
            ax.plot(heights, efficiencies[0], color=(0.7, 0, 0), linestyle='-', label="$r_{-1}'$", lw = LINE_WIDTH) 
            # 0 order
            ax.plot(heights, efficiencies[1], color='0.4', linestyle='-', label="$r_0'$", lw = LINE_WIDTH) 
            # 1 order
            ax.plot(heights, efficiencies[2], color=(0, 0, 0.7), linestyle='-', label="$r_1'$", lw = LINE_WIDTH) 
            ax.set_ylim([-0.01, 1.01]) # displace from zero to see the transition to evanescence
            ylabel = rf"Reflection at $\theta' = {inc_angle_deg}°$"
        elif efficiency_quantity == "t":
            # -1 order
            ax.plot(heights, efficiencies[0], color=(0.7, 0, 0), linestyle='-', label="$r_{-1}'$", lw = LINE_WIDTH)
            ax.plot(heights, efficiencies[3], color=(0.7, 0, 0), linestyle='-.', label="$t_{-1}'$", lw = LINE_WIDTH)  
            # 0 order
            ax.plot(heights, efficiencies[1], color='0.4', linestyle='-', label="$r_0'$", lw = LINE_WIDTH) 
            ax.plot(heights, efficiencies[4], color='0.4', linestyle='-.', label="$t_0'$", lw = LINE_WIDTH) 
            # 1 order
            ax.plot(heights, efficiencies[2], color=(0, 0, 0.7), linestyle='-', label="$r_1'$", lw = LINE_WIDTH) 
            ax.plot(heights, efficiencies[5], color=(0, 0, 0.7), linestyle='-.', label="$t_1'$", lw = LINE_WIDTH) 
            ax.set_ylim([-0.01, 1.01]) # displace from zero to see the transition to evanescence
            ylabel = rf"Efficiency at $\theta' = {inc_angle_deg}°$"
        elif efficiency_quantity == "PDr":
            ax.plot(heights, efficiencies[0], color=(0.7, 0, 0), linestyle='-', lw = LINE_WIDTH) 
            ylabel = rf"$\frac{{\partial r_{{-1}}'}}{{\partial\theta'}}({inc_angle_deg}°)$"
        elif efficiency_quantity == "PDt":
            ax.plot(heights, efficiencies[0], color=(0.7, 0, 0), linestyle='-', lw = LINE_WIDTH) 
            ylabel = rf"$\frac{{\partial t_{{-1}}'}}{{\partial\theta'}}({inc_angle_deg}°)$"

        # Optional plotting
        if efficiency_quantity in legend_needed:
            leg = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            frame = leg.get_frame()
            frame.set_edgecolor('black')
        
        if efficiency_quantity in symlog_needed:
            linthr = 0.1
            ax.set_yscale("symlog", linthresh=linthr, linscale=0.4)
            ax.yaxis.set_minor_locator(MinorSymLogLocator(linthr))
        
        # Axis labels
        ax.axhline(y=0, color='black', linestyle='-', lw = '1')
        ax.tick_params(axis='both', which='both', direction='in') # ticks inside box
        ax.set(title=rf"$\Lambda' = {self.grating_pitch/self.wavelength:.3f}\lambda$", xlabel=r"$h_1'/\lambda$", ylabel=ylabel)

        # Modify axes
        cm_to_inch = 0.393701
        fig_width = 20.85*cm_to_inch
        fig_height = 17.6*cm_to_inch
        fig.set_size_inches(fig_width/1.2, fig_height/1.2)

        return fig, ax
    

    def calculate_y_fields(self, height):
        """
        Calculate Ey fields on the grid at a fixed z height

        Parameters
        ----------
        height :   Height to calculate field

        Returns
        -------
        Ey :   y-component of electric field 
        """

        self.init_RCWA()
        fields = self.RCWA.Solve_FieldOnGrid(1,height)
        Efield = fields[0]
        Ey = np.transpose(Efield[1])
        return Ey

    def show_fields(self, heights: np.ndarray, field_output: str="real", fill_style: str="gouraud", show_eps_profile: bool=False):
        """
        Show out-of-plane electric field at given z heights for the twobox.

        Parameters
        ----------
        field_output     :   "real" (Ey), "square" (Ey**2) or "abs" (|Ey|) fields
        heights          :   z heights to calculate fields
        fill_style       :   Field plot shading style (passed to pcolormesh)
        show_eps_profile :   Overlay permittivity profile onto fields 

        Returns
        -------
        fig :   Field plot figure object
        axs :   Field plot axs object
        """

        Eys = np.zeros((len(heights), self.Nx), dtype=np.complex128)  # must be complex to assign complex Ey to Eys
        for idx, d in enumerate(heights):
            Eys[idx,:] = self.calculate_y_fields(d)

        if field_output == "real":
            Eys = np.real(Eys)
            cbar_label = r"$\Re(E_y)$"
            max_colour_scale = np.maximum(np.abs(np.min(Eys)), np.abs(np.max(Eys)))
            vmax = max_colour_scale
            vmin = -max_colour_scale
        elif field_output == "square":
            Eys = np.power(np.real(Eys),2)
            cbar_label = r"$\Re(E_y)^2$"
            max_colour_scale = np.max(Eys)
            vmax = max_colour_scale
            vmin = 0
        elif field_output == "abs":
            Eys = np.abs(Eys)
            cbar_label = r"$|E_y|$"
            max_colour_scale = np.max(Eys)
            vmax = max_colour_scale
            vmin = 0
        else:
            return print("Field output form not valid. Must be 'real', 'square' or 'abs'.")

        # The incident light comes from the positive z direction with our convention (can check by giving the boxes negative 
        # permittivity and observing the direction of reflection of the fields relative to the structure).
        Eys = Eys[::-1]  
        x0 = np.linspace(-0.5,0.5,self.Nx)
        

        fig, axs = plt.subplots(nrows=1, ncols=1)
        
        if show_eps_profile:
            axs2 = axs.twinx()
            
            # eps_array = self.RCWA.Return_eps(1,self.Nx,self.Ny,component='xx')
            # eps_array = np.flip(eps_array.real)
            eps_array = self.grating_grid
            
            eps_min = np.min(eps_array)
            eps_max = np.max(eps_array)
            
            eps_color = 'b'
            axs2.plot(x0, eps_array, color=eps_color)
            axs2.set(ylabel=r"$\varepsilon$")
            axs2.set_ylim(bottom=eps_min, top=eps_max)
            axs2.yaxis.label.set_color(eps_color)
        
        E_mesh = axs.pcolormesh(x0, heights/self.grating_pitch, Eys, vmin=vmin, vmax=vmax, shading=fill_style, cmap='hot')


        # Create an axes on the right side of ax. The width of cax will be x%
        # of ax and the padding between cax and ax will be fixed at y inch.
        if show_eps_profile:
            divider = make_axes_locatable(axs2)
            cax = divider.append_axes("right", size="2.5%", pad=0.9)
        else:
            divider = make_axes_locatable(axs)
            cax = divider.append_axes("right", size="2.5%", pad=0.05) 
        fig.colorbar(E_mesh, label=cbar_label, cax=cax)


        axs.set_title(label=rf"$\Lambda'={self.grating_pitch:.4f}\lambda_0$")
        axs.set(xlabel=r"$x'/\Lambda'$", ylabel=r"$z'/\Lambda'$")

        
        fig_mult = 4
        fig_width = 9
        if self.grating_depth/self.grating_pitch < 0.2:
            fig_height = 3*self.grating_depth/self.grating_pitch*fig_mult
        else:
            fig_height = self.grating_depth/self.grating_pitch*fig_mult
        fig.set_size_inches(fig_width, fig_height)
        
        return fig, axs