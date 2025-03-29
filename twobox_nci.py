"""
A class to create and simulate a TwoBox grating, storing all grating parameters and hyperparameters.

Contains plotting methods to show: grating permittivity profile, spectra, fields and angle dependence.

TODO: Move functions containing intensity, speed of light and mass to a separate module. These functions
rely on parameters that are not relevant to the grating simulation and should be kept separate. This module
should only contain bigrating simulation functions, enabling the user to easily implement their own figures
of merit in a separate module without worrying about the grating simulation.

TODO: Move plotting functions out of twobox.py and into a separate module.
"""

# IMPORTS ###########################################################################################################################################################################
import autograd.numpy as npa
from autograd import grad, jacobian
from autograd.scipy.special import erf as autograd_erf
from autograd.numpy import linalg as npaLA

from parameters import D1_ND, Parameters
I0, L, m, c = Parameters()

import grcwa
grcwa.set_backend('autograd')

import numpy as np
from numpy import *

import os
os.environ["OMP_NUM_THREADS"] = "1" 
os.environ["OPENBLAS_NUM_THREADS"] = "1" 
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 


def softmax(p,sigma):
    """
    Softmax needed for backpropagation by smoothing out the grating unit cell construction    
    
    NOTE: The translation by npa.max(p) is to avoid numerical instability when exponentiating 
          large numbers. Adding the max function doesn't ruin differentiability, because the 
          softmax expression with and without the max(p) shift are identical. Hence, the 
          derivative of the max-translated softmax is equivalent to the derivative of the
          unshifted softmax.
    
    Parameters
    ----------
    p     :   Array of floats to be converted to probabilities
    sigma :   Softmax inverse temperature, i.e. inverse smoothing factor. Smaller means smoother grating.

    Returns
    -------
    Probability of each element in array p based on its numerical value
    """

    e_x = npa.exp(sigma*(p - npa.max(p)))
    return e_x/npa.sum(e_x)

def softmin(p,sigma):
    """Used to approximate min via expected value of array p with probability distribution given by softmin(p)"""
    e_x = npa.exp(sigma*(npa.min(p) - p))
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
    box1_eps        :   A float for the left box relative permittivity
    box2_eps        :   A float for the right box relative permittivity
    gaussian_width  :   A float for the Gaussian beam width (metres)
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
        
        self.init_RCWA()

    @property
    def params(self):
        # Manipulate self.params instance variable using getter and setter properties rather than defining
        # in __init__. Also, need to define self.params here instead of in __init__. Both of these are
        # needed in order for user changes to instance variables to update self.params (and vice versa). 
        self._params = [self.grating_pitch, self.grating_depth, 
                       self.box1_width, self.box2_width, self.box_centre_dist, self.box1_eps, self.box2_eps, 
                       self.gaussian_width, self.substrate_depth, self.substrate_eps]
        return self._params
    @params.setter
    def params(self, new_params):
        self._params = new_params
        (self.grating_pitch, self.grating_depth, 
        self.box1_width, self.box2_width, self.box_centre_dist, self.box1_eps, self.box2_eps, 
        self.gaussian_width, self.substrate_depth, self.substrate_eps) = new_params
        self.build_grating_gradable()  # TODO: I think every instance method calls init_RCWA, so this is not needed
  
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
            
            probs = softmax(conditions,sigma)
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
        #       because changes to parameters such as wavelength do not change the grating parameters.
        self.build_grating_gradable()  # update twobox whenever user changes box parameters
        obj.GridLayer_geteps(self.grating_grid)


        planewave = {'p_amp':0,'s_amp':1,'p_phase':0,'s_phase':0}
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

    def diffraction_angle(self, m):
        """
        Calculate the diffraction angle for a given diffraction order m, if it exists.
        """
        sin_delta_m = npa.sin(self.angle) + m*self.wavelength/self.grating_pitch
        if abs(sin_delta_m) >= 1:
            delta_m = "no_diffraction_order"
        else:
            delta_m = npa.arcsin(sin_delta_m)
        return delta_m

    def Q(self):
        """
        Calculate efficiency factors Q_{pr,j}'(delta', lambda') for j = 1, 2
        """
        r,t = self.eff()
        
        Q1 = 0
        Q2 = 0
        orders = [-1,0,1]
        for m in range(len(orders)):  # TODO: get rid of the for loop here and use native np vectorisation
            delta_m = self.diffraction_angle(orders[m])
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


    def FoM(self, I:float=1e9, grad_method: str="finite", sigma: float=1.) -> float:
        """
        Calculate the grating single-wavelength figure of merit FD.

        This FOM relies on calculating radiation-pressure efficiency factors for a single grating and then 
        using symmetry to calculate the efficiency factors for the mirror-reflected grating. In this
        implementation, the optimised grating recorded via the twobox instance is the right-half grating,
        i.e. the grating lying on the positive x-axis at equilibrium. Hence, the twobox instance's parameters,
        efficiencies, etc. are all for the right-half grating, with the left-half grating obtained by inverting
        the unit cell along the x-axis about the unit-cell centre.

        MdS FoM: Minimise the eigenvalue with the largest real part. Equivalent to maximising the 
                 negative eigenvalue with the smallest real part. 

        Parameters
        ----------
        I           :   Laser intensity
        grad_method :   Method to calculate gradient ("finite","grad"). Must be "finite" for optimisation
        
        Returns
        -------
        FD :   Figure of merit
        """
        
        eigReal, eigImag = self.Eigs(I=I, m=m, c1=c, grad_method=grad_method, return_vec=False)

        # MdS FoM: Minimise the eigenvalue with the largest real part. Equivalent to maximising the 
        #          negative eigenvalue with the smallest real part. 
        FD = npa.min(-eigReal)  # standard minimum
        # FD = npa.sum(-eigReal*softmin(-eigReal,1.))  # softened minimum
        
        return FD

    def FoM_quality_factor(self, I:float=1e9, grad_method: str="finite") -> float:
        """
        Calculate the grating single-wavelength figure of merit FD.

        Quality factor FoM: Maximise the magnitude of the quality factor (Re(xi)/Im(xi)) 
                            for the eigenvalue with the smallest quality factor. Issue:
                            Im(xi) --> 0 will blow this up, and we need to track the sign.

        Parameters
        ----------
        I           :   Laser intensity
        grad_method :   Method to calculate gradient ("finite","grad"). Must be "finite" for optimisation
        
        Returns
        -------
        FD :   Figure of merit
        """
        
        raise NotImplementedError("Must determine how to handle signs and avoid Im(xi) = 0.")

    def FoM_LvR(self, I:float=1e9, grad_method: str="finite") -> float:
        """
        Calculate the grating single-wavelength figure of merit FD using LvR's most updated method.

        Parameters
        ----------
        I           :   Laser intensity
        grad_method :   Method to calculate gradient ("finite","grad"). Must be "finite" for optimisation
        
        Returns
        -------
        FD :   Figure of merit
        """
        
        eigReal, eigImag = self.Eigs(I=I, m=m, c1=c, grad_method=grad_method, return_vec=False)

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
            sorted_x = npa.sort(x.flatten())
            unique_values = sorted_x[npa.concatenate(([True], npa.diff(sorted_x) != 0))]

            # Append filled_value as needed
            k = len(unique_values)
            for i in range(4-k):
                unique_values = npa.append(unique_values,filled_value)

            return unique_values
    
        # NOTE: In the following penalty and reward terms, all operations must be done element-wise to avoid 
        #       "RuntimeWarning: invalid value encountered in divide" during optimisation
        # TODO: Determine why we can't use npa functions here

        # LvR FoM: Reward all Re(eig) being negative
        # Fill repeated entries in eigReal with -1 so that, after squaring, they don't influence the product
        eig_real_unique     =   unique_filled(eigReal, -1)
        eig_real_neg_unique =   npa.minimum(0., eig_real_unique)
        func_real_neg_array =   npa.power(eig_real_neg_unique, 2)
        func_real_neg       =   func_real_neg_array[0] * func_real_neg_array[1] * func_real_neg_array[2] * func_real_neg_array[3]
        # func_real_neg       =   npa.prod(func_real_neg_array) 

        # Remove Re(eig)<0 contribution if no restoring behaviour
        # log(1+x^2) chosen as a smooth function that moves away from zero
        # NOTE: This function has zero gradient at x=0, which is bad for stepping away from zero imaginary 
        #       part. Also, the gradient saturates at large x, which doesn't matter in the sense of 
        #       needng the imaginary part to be nonzero.
        func_imag_array     =   npa.log(1 + npa.power(eigImag,2))
        func_imag           =   func_imag_array[0] * func_imag_array[1] * func_imag_array[2] * func_imag_array[3]
        # func_imag           =   npa.prod(func_imag_array)

        # Penalise mixed positive and negative Re(eig)
        # Fill repeated entries in eigReal with 0 so that they don't influence the sum
        real_unique_0       =   unique_filled(eigReal, 0.)
        neg_array           =   npa.power(npa.minimum(0.,real_unique_0), 2)
        pos_array           =   npa.power(npa.maximum(0.,real_unique_0), 2)
        # penalty             =   npa.sum(neg_array) * npa.sum(pos_array)
        neg_sum             =   neg_array[0] + neg_array[1] + neg_array[2] + neg_array[3]
        pos_sum             =   pos_array[0] + pos_array[1] + pos_array[2] + pos_array[3]
        penalty             =   neg_sum * pos_sum

        # Penalise all positive Re(eig)
        # Fill repeated entries in eigReal with 1 so that they don't influence the product
        real_unique_1       =   unique_filled(eigReal, 1)
        all_pos_array       =   npa.power(npa.maximum(0.,real_unique_1), 2)
        penalty2            =   all_pos_array[0] * all_pos_array[1] * all_pos_array[2] * all_pos_array[3]
        # penalty2            =   npa.prod(all_pos_array)


        FD = func_real_neg * func_imag - penalty - penalty2
        return FD

    def sail_stiffness(self, I: float=10e9, m: float=1/1000, c1:float=299792458, grad_method: str='finite', out="tr"):
        """
        Calculate stiffness coefficients/Jacobian coefficients for a symmetric lightsail at equilibrium. Here, symmetric 
        means symmetric with respect to reflections about the laser-beam axis (when the CoM lies on the laser-beam axis).

        Parameters
        ----------
        I           :   Laser intensity
        m           :   Spacecraft mass (sail membrane + payload)
        c1          :   speed of light  # TODO: why is this a parameter?
        grad_method :   Method to calculate gradient ("finite","grad"). Must be "finite" for optimisation
        out         :   Output format 
                        "tr" for translation coefficients first, then rotation coefficients. Use when outputting to Jacobian.
                        "rd" for restoring coefficients first, then damping coefficients
        
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
                Q1R, Q2R, dQ1ddeltaR, dQ2ddeltaR, dQ1dlambdaR, dQ2dlambdaR = self.return_Qs(h_angle, h_wavelength)
            case "grad":
                Q1R, Q2R, dQ1ddeltaR, dQ2ddeltaR, dQ1dlambdaR, dQ2dlambdaR = self.return_Qs_auto(return_Q=True)
            case _:
                raise ValueError("grad_method not recognised. Must be 'finite' or 'grad'.")

        w = self.gaussian_width
        w_bar = w/L  # width normalised to total grating length
        lam = self.wavelength 

        # Convert velocity factors to wavelength factors
        # TODO: may need to change these factors to account for non-unity starting wavelengths
        D = 1/lam  # Doppler factor assuming starting wavelength is 1
        g = (npa.power(lam,2) + 1)/(2*lam)  # Lorentz factor
   
        # Lightsail reflection-symmetry conditions
        Q1L = Q1R                ; Q2L = -Q2R;   
        dQ1ddeltaL  = -dQ1ddeltaR; dQ2ddeltaL  = dQ2ddeltaR
        dQ1dlambdaL = dQ1dlambdaR; dQ2dlambdaL = -dQ2dlambdaR        
        
        # y acceleration terms
        # NOTE: derivatives with respect to lambda differ from derivatives with respect to frequency offset, the latter
        # being presented in Liam's thesis
        fy_y    = - D**2 * I/(m*c1) * (Q2R - Q2L) * (1 - npa.exp(-1/(2*w_bar**2)))
        fy_phi  = - D**2 * I/(m*c1) * (dQ2ddeltaR + dQ2ddeltaL) * w/2 * npa.sqrt(np.pi/2) * autograd_erf(1/(w_bar*npa.sqrt(2)))
        fy_vy   = - D**2 * I/(m*c1) * 1/c1 * (D+1)/(D*(g+1)) * (Q1R + Q1L + dQ2ddeltaR + dQ2ddeltaL) * w/2 * npa.sqrt(np.pi/2) * autograd_erf(1/(w_bar*npa.sqrt(2)))
        fy_vphi =   D**2 * I/(m*c1) * 1/c1 * (2*(Q2R - Q2L) - lam*(dQ2dlambdaR - dQ2dlambdaL)) * (w/2)**2 * (1 - npa.exp(-1/(2*w_bar**2)))

        # phi acceleration terms
        # TODO: generalise for non-flat-geometry moments of inertia
        # TODO: rename vphi to phidot to avoid confusion with vphi = length*phidot
        fphi_y    =  D**2 * 12*I/(m*c1*L**2) * (Q1R + Q1L) * (w/2*npa.sqrt(np.pi/2) * autograd_erf(1/(w_bar*npa.sqrt(2))) - L/2*npa.exp(-1/(2*w_bar**2))) 
        fphi_phi  =  D**2 * 12*I/(m*c1*L**2) * (dQ1ddeltaR - dQ1ddeltaL - (Q2R - Q2L)) * (w/2)**2 * (1 - npa.exp(-1/(2*w_bar**2)))
        fphi_vy   =  D**2 * 12*I/(m*c1*L**2) * 1/c1 * (D+1)/(D*(g+1)) * (dQ1ddeltaR - dQ1ddeltaL - (Q2R - Q2L)) * (w/2)**2 * (1 - npa.exp(-1/(2*w_bar**2)))
        fphi_vphi = -D**2 * 12*I/(m*c1*L**2) * 1/c1 * (2*(Q1R + Q1L) - lam*(dQ1dlambdaR + dQ1dlambdaL)) * (w/2)**2 * (w/2*npa.sqrt(np.pi/2) * autograd_erf(1/(w_bar*npa.sqrt(2))) - L/2*npa.exp(-1/(2*w_bar**2))) 

        match out:
            case "tr":
                return npa.array([fy_y, fy_phi, fy_vy, fy_vphi, fphi_y, fphi_phi, fphi_vy, fphi_vphi])
            case "rd":
                return npa.array([fy_y, fy_phi, fphi_y, fphi_phi, fy_vy, fy_vphi, fphi_vy, fphi_vphi])
            case _:
                raise ValueError("Invalid output format. Must be 'tr' or 'rd'.")
    

    def Eigs(self, I: float=10e9, m: float=1/1000, c1:float=299792458, grad_method: str='finite', return_vec: bool = False):
        """
        Calculate eigendecomposition of Jacobian matrix at equilibrium

        Parameters
        ----------
        I           :   Laser intensity
        m           :   Spacecraft mass (sail membrane + payload)
        c1          :   speed of light  # TODO: why is this a parameter?
        grad_method :   Method to calculate gradient ("finite","grad"). Must be "finite" for optimisation
        return_vec  :   If true, return eigenvectors as well as eigenvalues
        
        Returns
        -------
        eigReal :   Real part of Jacobian eigenvalues
        eigImag :   Imaginary part of Jacobian eigenvalues
        eigvecs :   Eigenvectors of Jacobian matrix
        """

        stiffnesses = self.sail_stiffness(I,m,c1,grad_method,out="tr")

        # Build the Jacobian matrix
        J = npa.array([[0,0,1,0],[0,0,0,1],[*stiffnesses[:4]],[*stiffnesses[4:]]])

        # Find the real part of eigenvalues    
        eigvalvec = npaLA.eig(J)
        eigvals   = eigvalvec[0]
        eigReal   = npa.real(eigvals)
        eigImag   = npa.imag(eigvals)

        if return_vec:
            eigvecs = eigvalvec[1]
            return eigReal, eigImag, eigvecs
        else:
            return eigReal, eigImag


    def lsa_info(self, wavelength, I: float=0.5e9):
        """
        Calculate quantities relevant to linear stability analysis (LSA) of the twobox dynamics. Also calculates
        the radiation pressure cross sections and their derivatives.
        
        TODO: Remove wavelength parameter and use self.wavelength instead
        TODO: why do the returns need to be tuples?

        Parameters
        ----------
        wavelength :   Wavelength of incident light
        I          :   Incident light intensity
        
        Returns
        -------
        efficiencies :   Radiation pressure cross sections and their derivatives  
        rest_coeffs  :   Restoring force/torque coefficients
        damp_coeffs  :   Damping force/torque coefficients
        eigReal      :   Real component of eigenvalues
        eigImag      :   Imaginary component of eigenvalues
        """

        input_wavelength = self.wavelength
        self.wavelength = wavelength

        efficiencies = tuple(self.return_Qs_auto(return_Q=True))
        
        stiffnesses = self.sail_stiffness(I,m,c,grad_method="grad",out="rd")
        rest_coeffs = tuple([*stiffnesses[:4]])
        damp_coeffs = tuple([*stiffnesses[4:]])

        eigReal, eigImag, eigvecs = self.Eigs(I,m,c,grad_method="grad",return_vec=True)

        self.wavelength = input_wavelength
        return efficiencies, rest_coeffs, damp_coeffs, eigReal, eigImag, eigvecs