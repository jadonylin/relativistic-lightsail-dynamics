"""
A class to create and simulate a TwoBox grating, storing all grating parameters and hyperparameters.

Contains plotting methods to show: grating permittivity profile, spectra, fields and angle dependence.

TODO: Move functions containing intensity, speed of light and mass to a separate module. These functions
rely on parameters that are not relevant to the grating simulation and should be kept separate. This module
should only contain bigrating simulation functions, enabling the user to easily implement their own figures
of merit in a separate module without worrying about the grating simulation.
"""

# IMPORTS ###########################################################################################################################################################################
# from torch import erf as torch_erf
# from torch import linalg as torchLA
# from autograd.scipy.special import erf as erf
# from autograd.numpy import linalg as npaLA
from agfunc import agfunc
from parameters import Parameters
try:
    from autograd.numpy.numpy_boxes import ArrayBox
except ImportError:
    ArrayBox = None

I0, L, m, c = Parameters()


import torch
import torcwa
import grcwa
grcwa.set_backend('autograd')

# If GPU support TF32 tensor core, the matmul operation is faster than FP32 but with less precision.
# If you need accurate operation, you have to disable the flag below.
torch.backends.cuda.matmul.allow_tf32 = False
# sim_dtype = torch.complex64
# geo_dtype = torch.float32

sim_dtype = torch.complex128
geo_dtype = torch.float64


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')




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
# from numpy import *

import os
# os.environ["OMP_NUM_THREADS"] = "1" 
# os.environ["OPENBLAS_NUM_THREADS"] = "1" 
# os.environ["MKL_NUM_THREADS"] = "1" 
import grcwa
grcwa.set_backend('autograd')
# os.environ["VECLIB_MAXIMUM_THREADS"] = "1" 
# os.environ["NUMEXPR_NUM_THREADS"] = "1" 

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





class TwoBox:
    """
    A TwoBox grating is a grating with two "boxes" (dielectric squares/resonators) in the unit cell. 

    Uses GRCWA library to simulate the grating.
    Simulation is re-run if you change instance variables. 
    All physical lengths pertaining to the grating are normalised by the excitation/laser wavelength.

    Attributes
    ----------
    grating_pitch         :   A float for the grating pitch/period 
    grating_depth         :   A float for the grating layer depth/height/thickness 
    box1_width            :   A float for the left box/resonator width
    box2_width            :   A float for the right box/resonator width
    box_centre_dist       :   A float for the distance between the box centres
    box1_eps              :   A float for the left box relative permittivity
    box2_eps              :   A float for the right box relative permittivity
    gaussian_width        :   A float for the Gaussian beam width (metres)
    substrate_depth       :   A float for the substrate layer depth/height/thickness
    substrate_eps         :   A float for the substrate permittiivty
    wavelength            :   A float for the excitation-plane-wave wavelength (laser-frame wavelength)
    angle                 :   A float for the excitation-plane-wave angle
    Nx                    :   An integer for the number of grid points in the unit cell
    nG                    :   An integer for the number of Fourier components used in the RCWA simulation
    Qabs                  :   A float for the relaxation parameter, determining the strength of the imaginary frequency and thus smoothness of resonances
    RCWA_engine           :   RCWA engine to use - 'GRCWA' or 'TORCWA'
    torcwa_edge_sharpness :   An integer for the sharpness of the edge of the unit cell in TORCWA
    title                 :   A string for the title of plots
    """

    def __init__(self, grating_pitch: float, grating_depth: float, box1_width: float, box2_width: float, box_centre_dist: float, box1_eps: complex, box2_eps: complex, 
                 gaussian_width: float, substrate_depth: float, substrate_eps: float, 
                 wavelength: float=1., angle: float=0.,
                 Nx: float=1000, nG: int=25, Qabs: float=np.inf,
                 RCWA_engine: float='GRCWA', torcwa_edge_sharpness: int =45, title: str=None) -> None:

        self.RCWA_engine = RCWA_engine
        
        if self.RCWA_engine == 'GRCWA':
            self.npa = agfunc('autograd')
        elif self.RCWA_engine == 'TORCWA':            
            if Nx < nG*2:
                raise ValueError("Nx must be at least 2*nG for TORCWA")
            if geo_dtype == torch.float64:
                self.npa = agfunc('torch', device=device, precision='double')
            elif geo_dtype == torch.float32:
                self.npa = agfunc('torch', device=device, precision='single')
            else:
                raise ValueError("Invalid torch precision. Choose 'double' or 'single'.")
        else:
            raise ValueError("Invalid RCWA engine. Choose 'GRCWA' or 'TORCWA'.")
        self.grating_pitch = self.npa.array(float(grating_pitch))
        self.grating_depth = self.npa.array(float(grating_depth))
        self.box1_width = self.npa.array(float(box1_width))
        self.box2_width = self.npa.array(float(box2_width))
        self.box_centre_dist = self.npa.array(float(box_centre_dist))
        self.box1_eps = self.npa.array(float(box1_eps)) # complex causes problems with FOM (adaptive? gradient? not clear)
        self.box2_eps = self.npa.array(float(box2_eps))  # complex causes problems with FOM (adaptive? gradient? not clear)
        
        self.gaussian_width = self.npa.array(float(gaussian_width))
        self.substrate_depth = self.npa.array(float(substrate_depth))
        self.substrate_eps = self.npa.array(float(substrate_eps))
        
        self.wavelength = self.npa.array(float(wavelength))
        self.angle = self.npa.array(float(angle))

        self.Nx = Nx
        self.Ny = 1  # 1D grating simulation, only one grid in the y-direction (transverse to the 1D periodicity)
        self.nG = nG
        self.Qabs = Qabs
        self.torcwa_edge_sharpness = torcwa_edge_sharpness


        self.invert_unit_cell = False

        if title is None:
            self.title = self.RCWA_engine
        else:
            self.title = title

        if self.RCWA_engine == 'GRCWA':
            self.init_RCWA()

        elif self.RCWA_engine == 'TORCWA':            
            if Nx<nG*2:
                raise ValueError("Nx must be at least 2*nG for TORCWA")
            self.init_TORCWA()
        else:
            raise ValueError("Invalid RCWA engine. Choose 'GRCWA' or 'TORCWA'.")

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
        CHECK: Not suitable for torcwa either ?
        """

        Lam = self.grating_pitch
        w1 = self.box1_width
        w2 = self.box2_width
        bcd = self.box_centre_dist
        x1 = w1/2 + 0.02*Lam  # box1 centre location (offset to avoid left box left edge clipping)    
        x2 = x1 + bcd  # box2 centre location
        
        x = self.npa.linspace(0,Lam,self.Nx)
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
            self.grating_grid = self.npa.array(grating)[::-1]
        else:
            self.grating_grid = self.npa.array(grating)

        return self.npa.array(grating)

    def build_grating_gradable(self, sigma: float=100.):
        if self.RCWA_engine == 'GRCWA':
            self.build_grating_GRCWA(sigma)
        elif self.RCWA_engine == 'TORCWA':            
            self.build_grating_torcwa()
        else:
            raise ValueError("Invalid RCWA engine. Choose 'GRCWA' or 'TORCWA'.")
        
    def build_grating_GRCWA(self, sigma: float=100.):
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
        grid_left_boundaries = self.npa.linspace(0,Lam-dx,self.Nx) # does not include x = Lam boundary
        # In this formulation, grid numbers 0, 1, ..., Nx-1 refer to the left boundaries (consistent with x position from 0 to 1*pitch)
        box1_left_boundary = x1-w1/2
        box2_right_boundary = x2+w2/2

        # Build grating by looping across the unit cell grids instead of using index assignment to make build_grating 
        # autograd differentiable.
        grating = [] 
        for grid_left_boundary in grid_left_boundaries:
            # These floats measure how much the current grid fits each condition
            grid_in_box1 = w1/2 - self.npa.abs(grid_left_boundary-x1)
            grid_left_of_box1 = box1_left_boundary - grid_left_boundary
            grid_in_box2 = w2/2 - self.npa.abs(grid_left_boundary-x2)
            grid_between_boxes = box_separation/2 - self.npa.abs(grid_left_boundary - boxes_midpoint)
            grid_right_of_box2 = grid_left_boundary - box2_right_boundary 

            conditions = self.npa.array([grid_in_box1, grid_in_box2, grid_left_of_box1, grid_between_boxes, grid_right_of_box2])
            returns = self.npa.array([eb1, eb2, 1, 1, 1])
            
            probs = self.npa.softmax(conditions,sigma)
            eps = self.npa.sum(probs*returns)

            grating.append(eps)

        if self.invert_unit_cell:
            self.grating_grid = self.npa.array(grating)[::-1]
        else:
            self.grating_grid = self.npa.array(grating)

        return self.npa.array(grating)


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

        # setup RCWA
        obj = grcwa.obj(self.nG,L1,L2,freqcmp,theta,phi,verbose=0) # verbose=1 for debugging, prints ng actually used 

        # add layers

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
        if self.RCWA_engine == 'GRCWA':
            self.init_RCWA()
            R_byorder,T_byorder = self.RCWA.RT_Solve(normalize=1, byorder=1)
            Fourier_orders = self.RCWA.G

            Rs = []
            Ts = []
            RT_orders = [-1,0,1]
            # IMPORTANT: have to use append method to a list rather than index assignment
            # Else, autograd will throw a TypeError with float() argument being an ArrayBox
            for order in RT_orders:
                Rs.append(self.npa.sum(R_byorder[Fourier_orders[:,0]==order]))
                Ts.append(self.npa.sum(T_byorder[Fourier_orders[:,0]==order]))
        elif self.RCWA_engine == 'TORCWA':
            self.init_TORCWA()
            RT_orders=self.grating_orders()
            Rs=self.npa.zeros(len([-1,0,1]))
            Ts=self.npa.zeros(len([-1,0,1]))
            orders=[[j,0] for j in RT_orders]
            lRs = self.npa.abs(self.npa.power(self.RCWA.S_parameters(orders=orders,direction='forward',port='reflection',polarization='yy',ref_order=[0,0],power_norm=True),2))
            lTs = self.npa.abs(self.npa.power(self.RCWA.S_parameters(orders=orders,direction='forward',port='transmission',polarization='yy',ref_order=[0,0],power_norm=True),2))
            for i,j in enumerate(RT_orders):
                Rs[1+j]=lRs[i]
                Ts[1+j]=lTs[i]
        return Rs,Ts
    def Q_trivial(self):
        """ 
        returns simple funciton of aguments, for testing jacobian
        
        """ 
        # Q1=self.npa.cos(self.angle)
        # Q2=self.wavelength**2
        r,t = self.eff()
        Q1=r[0]
        Q2=t[0]
        print('Q1,Q2',Q1,Q2)
        # Q1=self.tNeg1(self.angle)
        # Q2=self.rNeg1(self.angle)
        
        if self.RCWA_engine == 'TORCWA':
            return torch.stack((Q1,Q2))
        else:
            return self.npa.array( [Q1, Q2] )
    def diffraction_angle(self, m):
        """
        Calculate the diffraction angle for a given diffraction order m, if it exists.
        """
        sin_delta_m = self.npa.sin(self.npa.array(self.angle)) + m*self.wavelength/self.grating_pitch
        delta_m = self.npa.arcsin(sin_delta_m)

        return delta_m

    def Q(self):
        """
        Calculates efficiency factors Q_{pr,j}'(delta', lambda')
        todo: check the torch version works...
        """
        r,t = self.eff()
        
        
        Q1=self.npa.array(0.0)
        Q2=self.npa.array(0.0)
        M=[-1,0,1]
        
        # M=self.grating_orders() # this works in pytorch, but not in autograd
        # begin debugging torch pytorch jacobian returning 0:
        # M=[0]
        # end debug
        if self.RCWA_engine == 'TORCWA':
            M=self.grating_orders() # this works in pytorch, but not in autograd, which throws an error that int isn't differentiable
            # however, doing it this way doesn't help with NaN being by  returned for derivatives when only m=0 orders are propagative in torcwa
        for ord in M:
            m=1+ord # convert grating order to index of array, assumes -1,0,1
            delta_m=self.diffraction_angle(ord)
            # # if isinstance(delta_m,str):
            if self.npa.isnan(delta_m):
                """
                If no diffraction order, Q_{pr,j}' is unchanged
                """
                Q1 = Q1 + 0
                Q2 = Q2 + 0
            else:
            # take back to real as Q are real, complex intermediary only for gradient tracking compatibility
                Q1 = Q1+ r[m]*(1+self.npa.cos(self.angle+delta_m))+t[m]*(1-self.npa.cos(delta_m-self.angle))
                Q2 = Q2+ r[m]*self.npa.sin(self.angle+delta_m)+t[m]*self.npa.sin(delta_m-self.angle)
        Q1 =  self.npa.cos(self.angle)*Q1
        Q2 = -self.npa.cos(self.angle)*Q2
        if self.RCWA_engine == 'TORCWA':
            return torch.stack((Q1,Q2))
            # return self.npa.array( [Q1, Q2] )  
        else:
            return self.npa.array( [Q1, Q2] )
        
    ##################
    #### 1st derivatives


    def return_Qs(self, h_angle, h_wavelength):
        """
        Calculate efficiency factors and their derivatives
        Note: unchanged from pre-torcwa as it doesn't use autograd or grcwa
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
            # for debuging
            # return self.Q_trivial()
            # end debugging

        # Q_jacobian = self.npa.jacobian(Q_both, argnum=1)
        params =self.npa.array([ input_angle, input_wavelength] )

        Q_jacobian = self.npa.jacobian(Q_both)(params).squeeze() # PD_both_Q(self, params)
      

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
        NOTE: with TORCWA works with 'grad' automatic differentiation for both angle and wavelength 
              with GRCWA one of method_one, method_two must be 'finite' or code runs forever        
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

                if self.RCWA_engine == 'TORCWA':
                    return torch.stack((PD_Q1, PD_Q2))
                else:                    
                    return self.npa.array( [PD_Q1, PD_Q2] )

            if method_one=="grad":
                def Q_params(angle, wavelength):
                    self.angle = angle
                    self.wavelength = wavelength
                    return self.Q()
                

                PD = self.npa.jacobian(Q_params, argnum = var)
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

                if self.RCWA_engine == 'TORCWA':
                    return torch.stack((PD_Q1, PD_Q2))
                else:                    
                    return self.npa.array( [PD_Q1, PD_Q2] )
                

            if method_two == "grad":
                restore()

                def fun(angle, wavelength):
                    return first(angle, wavelength, method_one)
                
                
                PD2 = self.npa.jacobian(fun, argnum = var)
                PD2_value = PD2(input_angle, input_wavelength)
                restore()
                return PD2_value

        return second(method_two)

    ##################
    #### Optimisation functions

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
        FD = self.npa.min(-eigReal)  # standard minimum
        # FD = self.npa.sum(-eigReal*self.npa.softmin(-eigReal,1.))  # softened minimum
        # FD = self.npa.min(-eigReal) + self.npa.max(-eigReal)
        
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
        Last FoM implemented by Liam - not working with TORCWA
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
            sorted_x = self.npa.sort(x.flatten())
            unique_values = sorted_x[self.npa.concatenate(([True], self.npa.diff(sorted_x) != 0))]

            # Append filled_value as needed
            k = len(unique_values)
            for i in range(4-k):
                unique_values = self.npa.append(unique_values,filled_value)

            return unique_values
    
        # NOTE: In the following penalty and reward terms, all operations must be done element-wise to avoid 
        #       "RuntimeWarning: invalid value encountered in divide" during optimisation
        # TODO: Determine why we can't use npa functions here

        # LvR FoM: Reward all Re(eig) being negative
        # Fill repeated entries in eigReal with -1 so that, after squaring, they don't influence the product
        eig_real_unique     =   unique_filled(eigReal, -1)
        eig_real_neg_unique =   self.npa.minimum(0., eig_real_unique)
        func_real_neg_array =   self.npa.power(eig_real_neg_unique, 2)
        func_real_neg       =   func_real_neg_array[0] * func_real_neg_array[1] * func_real_neg_array[2] * func_real_neg_array[3]
        # func_real_neg       =   npa.prod(func_real_neg_array) 

        # Remove Re(eig)<0 contribution if no restoring behaviour
        # log(1+x^2) chosen as a smooth function that moves away from zero
        # NOTE: This function has zero gradient at x=0, which is bad for stepping away from zero imaginary 
        #       part. Also, the gradient saturates at large x, which doesn't matter in the sense of 
        #       needng the imaginary part to be nonzero.
        func_imag_array     =   self.npa.log(1 + self.npa.power(eigImag,2))
        func_imag           =   func_imag_array[0] * func_imag_array[1] * func_imag_array[2] * func_imag_array[3]
        # func_imag           =   npa.prod(func_imag_array)

        # Penalise mixed positive and negative Re(eig)
        # Fill repeated entries in eigReal with 0 so that they don't influence the sum
        real_unique_0       =   unique_filled(eigReal, 0.)
        neg_array           =   self.npa.power(self.npa.minimum(0.,real_unique_0), 2)
        pos_array           =   self.npa.power(self.npa.maximum(0.,real_unique_0), 2)
        # penalty             =   npa.sum(neg_array) * npa.sum(pos_array)
        neg_sum             =   neg_array[0] + neg_array[1] + neg_array[2] + neg_array[3]
        pos_sum             =   pos_array[0] + pos_array[1] + pos_array[2] + pos_array[3]
        penalty             =   neg_sum * pos_sum

        # Penalise all positive Re(eig)
        # Fill repeated entries in eigReal with 1 so that they don't influence the product
        real_unique_1       =   unique_filled(eigReal, 1)
        all_pos_array       =   self.npa.power(self.npa.maximum(0.,real_unique_1), 2)
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
        g = (self.npa.power(lam,2) + 1)/(2*lam)  # Lorentz factor
   
        # Lightsail reflection-symmetry conditions
        Q1L = Q1R                ; Q2L = -Q2R;   
        dQ1ddeltaL  = -dQ1ddeltaR; dQ2ddeltaL  = dQ2ddeltaR
        dQ1dlambdaL = dQ1dlambdaR; dQ2dlambdaL = -dQ2dlambdaR        
        
        # y acceleration terms
        # NOTE: derivatives with respect to lambda differ from derivatives with respect to frequency offset, the latter
        # being presented in Liam's thesis
        fy_y    = - D**2 * I/(m*c1) * (Q2R - Q2L) * (1 - self.npa.exp(-1/(2*w_bar**2)))
        fy_phi  = - D**2 * I/(m*c1) * (dQ2ddeltaR + dQ2ddeltaL) * w/2 * np.sqrt(np.pi/2) * self.npa.erf(1/(w_bar*np.sqrt(2)))
        fy_vy   = - D**2 * I/(m*c1) * 1/c1 * (D+1)/(D*(g+1)) * (Q1R + Q1L + dQ2ddeltaR + dQ2ddeltaL) * w/2 * np.sqrt(np.pi/2) * self.npa.erf(1/(w_bar*np.sqrt(2)))
        fy_vphi =   D**2 * I/(m*c1) * 1/c1 * (2*(Q2R - Q2L) - lam*(dQ2dlambdaR - dQ2dlambdaL)) * (w/2)**2 * (1 - self.npa.exp(-1/(2*w_bar**2)))

        # phi acceleration terms
        # TODO: generalise for non-flat-geometry moments of inertia
        # TODO: rename vphi to phidot to avoid confusion with vphi = length*phidot
        fphi_y    =  D**2 * 12*I/(m*c1*L**2) * (Q1R + Q1L) * (w/2*np.sqrt(np.pi/2) * self.npa.erf(1/(w_bar*np.sqrt(2))) - L/2*self.npa.exp(-1/(2*w_bar**2))) 
        fphi_phi  =  D**2 * 12*I/(m*c1*L**2) * (dQ1ddeltaR - dQ1ddeltaL - (Q2R - Q2L)) * (w/2)**2 * (1 - self.npa.exp(-1/(2*w_bar**2)))
        fphi_vy   =  D**2 * 12*I/(m*c1*L**2) * 1/c1 * (D+1)/(D*(g+1)) * (dQ1ddeltaR - dQ1ddeltaL - (Q2R - Q2L)) * (w/2)**2 * (1 - self.npa.exp(-1/(2*w_bar**2)))
        fphi_vphi = -D**2 * 12*I/(m*c1*L**2) * 1/c1 * (2*(Q1R + Q1L) - lam*(dQ1dlambdaR + dQ1dlambdaL)) * (w/2)**2 * (w/2*np.sqrt(np.pi/2) * self.npa.erf(1/(w_bar*np.sqrt(2))) - L/2*self.npa.exp(-1/(2*w_bar**2))) 

        match out:
            case "tr":
                return self.npa.stack((fy_y, fy_phi, fy_vy, fy_vphi, fphi_y, fphi_phi, fphi_vy, fphi_vphi))
            case "rd":
                return self.npa.stack((fy_y, fy_phi, fphi_y, fphi_phi, fy_vy, fy_vphi, fphi_vy, fphi_vphi))
            case "mat":
                row1=self.npa.stack((fy_y, fy_phi,fy_vy, fy_vphi ))
                row2=self.npa.stack(( fphi_y, fphi_phi, fphi_vy, fphi_vphi))                                    
                mat=self.npa.stack((row1,row2))
                return mat
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
        eigvecs :   Eigenvectors of Jacobian matrix, normalised to unit length
        """

        # stiffnesses = self.sail_stiffness(I,m,c1,grad_method,out="tr")
        stiffnesses = self.sail_stiffness(I,m,c1,grad_method,out="mat")

        # Build the Jacobian matrix
        # J = self.npa.array([[0,0,1,0],[0,0,0,1],[*stiffnesses[:4]],[*stiffnesses[4:]]])
        J=self.npa.concatenate((self.npa.array([[0,0,1,0],[0,0,0,1]]),stiffnesses))
        # J = self.npa.concatenate([[0,0,1,0],[0,0,0,1],[*stiffnesses[:4]],[*stiffnesses[4:]]])

        # Find the real part of eigenvalues    
        eigvalvec = self.npa.eig(J)
        eigvals   = eigvalvec[0]
        eigReal   = self.npa.real(eigvals)
        eigImag   = self.npa.imag(eigvals)

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
        self.wavelength = self.npa.array(wavelength)

        efficiencies = tuple(self.return_Qs_auto(return_Q=True))
        
        stiffnesses = self.sail_stiffness(I,m,c,grad_method="grad",out="rd")
        rest_coeffs = tuple([*stiffnesses[:4]])
        damp_coeffs = tuple([*stiffnesses[4:]])

        eigReal, eigImag, eigvecs = self.Eigs(I,m,c,grad_method="grad",return_vec=True)

        self.wavelength = input_wavelength
        return efficiencies, rest_coeffs, damp_coeffs, eigReal, eigImag, eigvecs

    ##################
    #### Plotting
    

    def show_permittivity(self, show_analytic_box: bool=False, show_box_edges: bool=False):
        """
        Show permittivity profile for the twobox.

        Displays two panels:
            Panel 1 is the user-input permittivity profile (the boxes smoothed by build_grating_gradable())
            Panel 2 is the GRCWA-resolved permittivity profile (with precision limited by nG)

        NOTE: GRCWA does not always align the returned permittivity with the grid numbers, it may be displaced by several grids.
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


       
        # if self.RCWA_engine == 'TORCWA':
        #     self.init_TORCWA() 
        #     #Torcwa does not need flipping this array - check x axis conventions?           
        #     eps_array=self.RCWA.return_layer(0,self.Nx,1)[0].cpu().numpy()
                
        # elif self.RCWA_engine == 'GRCWA':
        #     self.init_RCWA()
        #     eps_array = self.RCWA.Return_eps(which_layer=1,Nx=self.Nx,Ny=self.Ny,component='xx')
        #     # flip to match ordering of desired eps vs grid number - 
        #     eps_array = np.flip(eps_array)
        x0,eps_array=self.return_epsilon()
        # eps_array=self.to_numpy(eps_array)
        eps_array_real = eps_array.real

        # Show actual eps vs grid number 
        grids = np.arange(0, self.Nx, 1)
       

        fig, axs = plt.subplots(2, 1, figsize=(6,10), sharex=True)

        axs[0].plot(grids,self.grating_grid)
        axs[0].set(xlabel="Grid no.", ylabel=r"$\varepsilon$")  
        axs[0].set_title(f"Input permittivity profile. nG = {self.nG}, grid points = {self.Nx}")
        axs[0].set_xlim([0-0.01*self.Nx, self.Nx-1+0.01*self.Nx])

        axs[1].plot(grids,eps_array_real)
        axs[1].set(xlabel="Grid no.", ylabel=r"$\varepsilon$")
        axs[1].set_title(f"{self.title} permittivity profile. nG = {self.nG}, grid points = {self.Nx}")
        p=self.to_numpy(self.grating_pitch)
        if show_analytic_box:
            init_Nx = self.Nx
            self.Nx = 2000 # 1000*self.Nx
            fine_grids = np.arange(0, self.Nx, 1)
            if self.RCWA_engine == 'GRCWA':
                analytic_boxes = self.build_grating()
                axs[0].plot(fine_grids/2000.0*init_Nx,analytic_boxes)
                self.Nx = init_Nx
            elif self.RCWA_engine == 'TORCWA':
                torcwa.rcwa_geo.nx = self.Nx
                torcwa.rcwa_geo.grid()
                analytic_boxes = self.build_grating_torcwa()
                axs[0].plot(fine_grids/2000.0*init_Nx,analytic_boxes[:])
                self.Nx = init_Nx
                torcwa.rcwa_geo.nx = self.Nx            
                torcwa.rcwa_geo.grid()

        # TODO: Needs to be adapted to TORCWA with self.to_numpy
        #  if show_box_edges
        #     box1_left_edge = 0.02*p  # TODO: save preset 0.02 value (used here and in build_grating()) somewhere accessible
        #     box1_right_edge = box1_left_edge + self.to_numpy(self.box1_width)
        #     box1_centre = (box1_left_edge + box1_right_edge)/2
        #     box2_left_edge = box1_centre + self.box_centre_dist - self.box2_width/2 
        #     box2_right_edge = box2_left_edge + self.box2_width

        #     vlines = self.Nx/self.grating_pitch*np.array([box1_left_edge, box1_right_edge, box2_left_edge, box2_right_edge])
        #     # colors = ["tab:orange", "tab:orange", "tab:orange", "tab:orange"]
        #     ymax = [self.box1_eps, self.box1_eps, self.box2_eps, self.box2_eps]
        #     for v, y in zip(vlines,ymax):
                # axs[0].axvline(v, ymax=y, color="tab:orange", linestyle="--")  
        plt.show()
        return x0, eps_array_real, fig, axs
    
    def show_angular_efficiency(self, theta_max: float=20., num_plot_points: int=100):
        """
        Show grating efficiencies as a function of excitation angle for the twobox.

        TODO: add flag to show angles for order cutoffs

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
            # Calculate efficiencies for each order
            self.angle = self.npa.array(theta)
            Rs,Ts = self.eff()
            efficiencies[:3,idx] = self.to_numpy(Rs)
            efficiencies[3:,idx] = self.to_numpy(Ts)
            
        self.angle = init_angle # reset user-initialised angle


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

        ax.set(title=rf"{self.title} $\Lambda' = {self.grating_pitch/self.wavelength:.3f}\lambda$, \
               $h_1' = {self.grating_depth/self.wavelength:.3f}\lambda$, $\lambda$ = {self.wavelength:.7f}~$\mu$m"
            , xlabel=r"Incident angle, $\delta$ (°)"
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

# the following functions for plotting and debugging autograd/torch
    def rNeg1(self, angle):
        self.angle = angle
        ra,_ = self.eff()
        # r = ra[1]
        r = ra[0]
        return r


    def tNeg1(self, angle):
        self.angle = angle
        _,ta= self.eff()
        t=ta[0]
        return t

    def PDrNeg1(self, angle):
        a=self.npa.grad(self.rNeg1)(self.npa.array(angle))
        return a


    def PDtNeg1(self, angle):
        a=self.npa.grad(self.tNeg1)(self.npa.array(angle))
        return a
# end debug functions

    def show_spectrum(self, angle: float=0., efficiency_quantity: str="PDr", wavelength_range: list=[1., 1.5], num_plot_points: int=200, I: float=10e9, grad_method: str="grad"):
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
                Rs,Ts = self.RT()
                efficiencies[:n_orders,idx] = self.to_numpy(Rs)
                efficiencies[n_orders:,idx] = self.to_numpy(Ts)
            elif efficiency_quantity == "PDr":
                efficiencies[0,idx] = self.to_numpy(self.PDrNeg1(angle)) # this removes the autograd function -- ok for plotting            
            elif efficiency_quantity == "PDt":
                efficiencies[0,idx] = self.to_numpy(self.PDtNeg1(angle)) # this removes the autograd function -- ok for plotting
                

            elif efficiency_quantity == "FoM":
                efficiencies[0,idx] = self.FoM(I=I, grad_method="grad")


        self.wavelength = init_wavelength


        fig, ax = plt.subplots(1)         
        p = self.to_numpy(self.grating_pitch)
        ax.set_xlim(wavelength_range/p)  # normalise wavelength to grating pitch
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
            ylabel = rf"Reflection at $\theta' = {inc_angle_deg:.2f}°$"
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
            ylabel = rf"Efficiency at $\theta' = {inc_angle_deg:.2f}°$"
        elif efficiency_quantity == "PDr":
            ax.plot(wavelengths/p, efficiencies[0], color=(0.7, 0, 0), linestyle='-', lw = LINE_WIDTH) 
            ylabel = rf"$\frac{{\partial r_{{-1}}'}}{{\partial\theta'}}({inc_angle_deg:.2f}°)$"
        elif efficiency_quantity == "PDt":
            ax.plot(wavelengths/p, efficiencies[0], color=(0.7, 0, 0), linestyle='-', lw = LINE_WIDTH) 
            ylabel = rf"$\frac{{\partial t_{{-1}}'}}{{\partial\theta'}}({inc_angle_deg:.2f}°)$"

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
        ax.set(title=rf"{self.title} $h_1' = {self.grating_depth/self.wavelength:.3f}\lambda_0$, $\Lambda' = {self.grating_pitch/self.wavelength:.3f}\lambda_0$", xlabel=r"$\lambda'/\Lambda'$", ylabel=ylabel)


        cm_to_inch = 0.393701
        fig_width = 20.85*cm_to_inch
        fig_height = 17.6*cm_to_inch
        fig.set_size_inches(fig_width/1.2, fig_height/1.2)
        
        return fig, ax


    def show_Eigs(self, wavelength_range: list=[1., 1.5],  I: float=10e9, num_plot_points: int=200, eig_real_log_axis: bool=True, eig_imag_log_axis: bool=True, marker: str='o'):
        """
        Show eigenvalue spectrum for the twobox.

        Parameters
        ----------
        wavelength_range    :   Wavelength range to plot spectrum (same units as grating pitch)
        I                   :   Laser intensity
        num_plot_points     :   Number of points to plot
        eig_real_log_axis   :   If true, logarithmic scale for real part of eigenvalues
        eig_imag_log_axis   :   If true, logarithmic scale for imaginary part of eigenvalues
        marker              :   Marker style passed to plt.plot()

        Returns
        -------
        fig         :   Spectrum figure object
        (ax1, ax2)  :   Real and imaginary spectrum axis objects
        """

        wavelengths = np.linspace(*wavelength_range, num_plot_points)
        init_wavelength = self.wavelength  # record user-initialised wavelength

        ## CALCULATE EIGS ##
        eigvals = self.npa.zeros((4,num_plot_points), dtype=np.complex128)
        
        for idx, lam in enumerate(wavelengths):
            # Calculate eigs for each order
            self.wavelength = self.npa.array(lam)
            real, imag = self.Eigs(I=I,m=m,c1=c, grad_method="grad", return_vec=False)
            eigvals[:,idx] = real + 1j*imag
            
        self.wavelength = init_wavelength # restore user-initialised wavelength


        # I'm assuming the dummy subplot creates spacing between the two other subplots
        fig, (ax1, dummy, ax2) = plt.subplots(nrows=1, ncols=3, width_ratios=(1,0.1,1))
        dummy.axis('off')
        p = self.to_numpy(self.grating_pitch)
        ax1.set_xlim(np.array(wavelength_range)/p) # normalise to grating pitch
        ax2.set_xlim(np.array(wavelength_range)/p) # normalise to grating pitch
        ax2.yaxis.tick_right()
        ax2.yaxis.set_label_position("right")

        colorReal = (0.7, 0, 0)
        colorImag = 'blue'
        for i in range(4):            
            ax1.plot(wavelengths/p,np.real(self.to_numpy(eigvals[i,:])), marker, markersize=0.5, markerfacecolor=colorReal, fillstyle='full',  color=colorReal)
            ax2.plot(wavelengths/p,np.imag(self.to_numpy(eigvals[i,:])), marker, markersize=0.5, markerfacecolor=colorImag, fillstyle='full',  color=colorImag)
            

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
        NOTE: unchaned for Torcwa as no GRCWA/autograd used
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
                efficiencies[:n_orders,idx] = self.to_numpy(Rs)
                efficiencies[n_orders:,idx] = self.to_numpy(Ts)
            elif efficiency_quantity == "PDr":
                efficiencies[0,idx] =self.to_numpy(self.PDrNeg1(angle))
            elif efficiency_quantity == "PDt":
                efficiencies[0,idx] = self.to_numpy(self.PDtNeg1(angle))
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
            ylabel = rf"Reflection at $\theta' = {inc_angle_deg:.2f}°$"
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
            ylabel = rf"Efficiency at $\theta' = {inc_angle_deg:.2f}°$"
        elif efficiency_quantity == "PDr":
            ax.plot(heights, efficiencies[0], color=(0.7, 0, 0), linestyle='-', lw = LINE_WIDTH) 
            ylabel = rf"$\frac{{\partial r_{{-1}}'}}{{\partial\theta'}}({inc_angle_deg:.2f}°)$"
        elif efficiency_quantity == "PDt":
            ax.plot(heights, efficiencies[0], color=(0.7, 0, 0), linestyle='-', lw = LINE_WIDTH) 
            ylabel = rf"$\frac{{\partial t_{{-1}}'}}{{\partial\theta'}}({inc_angle_deg:.2f}°)$"

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
        ax.set(title=rf"{self.title} Efficiencies at $\theta' = {inc_angle_deg:.2f}°$, $\Lambda' = {self.grating_pitch/self.wavelength:.3f}\lambda$", xlabel=r"$h_1'/\lambda$", ylabel=ylabel)

        # Modify axes
        cm_to_inch = 0.393701
        fig_width = 20.85*cm_to_inch
        fig_height = 17.6*cm_to_inch
        fig.set_size_inches(fig_width/1.2, fig_height/1.2)

        return fig, ax
    

    def calculate_y_fields(self, height):
        """
        Return Ey on the grid at a fixed height

        Parameters
        ----------
        height :   Height to calculate field
        TODO: check TORCWA returns same orientation/order as GRCWA
        """
        if self.RCWA_engine == 'GRCWA':
            self.init_RCWA()
            if(height<=self.grating_depth and height>=0): # in grating layer
                fields = self.RCWA.Solve_FieldOnGrid(1,height) # above grating
                Efield = fields[0]
                Ey = np.transpose(Efield[1])
            elif(height>self.grating_depth): # above  grating
                Ey = np.zeros((self.Nx,1)) # self.RCWA.Solve_FieldOnGrid(2,height) 
            else:            # below grating
                Ey = np.zeros((self.Nx,1)) # self.RCWA.Solve_FieldOnGrid(0,height)
            
        elif self.RCWA_engine == 'TORCWA':
            self.init_TORCWA()
            self.RCWA.source_planewave(amplitude=[0,1.],direction='forward')
            z=self.npa.array([height])
            [Ex, Ey, Ez], [Hx, Hy, Hz] = self.to_numpy(self.RCWA.field_xz(torcwa.rcwa_geo.x,z,0))
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
        NOTE: unchaned for Torcwa as no GRCWA/autograd used

        Returns
        -------
        fig :   Field plot figure object
        axs :   Field plot axs object
        """

        Eys = np.zeros((len(heights), self.Nx), dtype=np.complex128)  # must be complex to assign complex Ey to Eys
        for idx, d in enumerate(heights):
            Eys[idx,:] = self.calculate_y_fields(d).flatten()

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
        
        p=self.to_numpy(self.grating_pitch)
        grating_depth=self.to_numpy(self.grating_depth)
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
        E_mesh = axs.pcolormesh(x0, heights/p, Eys, vmin=vmin, vmax=vmax, shading=fill_style, cmap='hot')


        # Create an axes on the right side of ax. The width of cax will be x%
        # of ax and the padding between cax and ax will be fixed at y inch.
        if show_eps_profile:
            divider = make_axes_locatable(axs2)
            cax = divider.append_axes("right", size="2.5%", pad=0.9)
        else:
            divider = make_axes_locatable(axs)
            cax = divider.append_axes("right", size="2.5%", pad=0.05) 
        fig.colorbar(E_mesh, label=cbar_label, cax=cax)


        axs.set_title(label=rf"{self.title}$\Lambda'={p:.4f}\lambda_0$")
        axs.set(xlabel=r"$x'/\Lambda'$", ylabel=r"$z'/\Lambda'$")
        axs.set_ylim(bottom=np.min(heights), top=np.max(heights))
        axs.set_aspect('equal')
        
        fig_mult = 4
        fig_width = 9
        
        
        # if grating_depth/p < 0.2:
        #     fig_height = 3*grating_depth/p*fig_mult
        # else:
        #     fig_height = grating_depth/p*fig_mult
        # fig.set_size_inches(fig_width, fig_height)
        
        return fig, axs
    
# TORCWA methods
    def init_TORCWA(self):
        """
        Initialise the TORCWA solver
        """
        # empty GPU cache to avoid memory issues
        # torch.cuda.empty_cache()
        # Grating
        # To simulate a 1D grating, take a small periodicity in the y-direction. 
        # The grating is in the x-direction.
        dy = 1e-4 
        L1 = [1.,0]
        L2 = [0,dy] 
        # testing if this helps with jacobian
        # self.wavelength=self.npa.array(self.wavelength) #,dtype=geo_dtype,device=device)
        # end test
        freq = self.npa.array(1/self.wavelength,dtype=geo_dtype,device=device) # freq = 1/wavelength when c = 1
        # freq=1.0/self.wavelength
        freqcmp = freq*(1+1j/2/self.Qabs)

        # Incoming wave
        theta = self.angle # radians
        phi = 0.

        # setup TORCWA
        # geometry
        L = [self.grating_pitch, dy]            # nm / nm size of unit cell
        torcwa.rcwa_geo.dtype = geo_dtype
        torcwa.rcwa_geo.device = device
        torcwa.rcwa_geo.Lx = L[0]
        torcwa.rcwa_geo.Ly = L[1]
        torcwa.rcwa_geo.nx = self.Nx
        torcwa.rcwa_geo.ny = 2 # np.min(self.Ny,2) # 2 minimum for 2d simulation displaying ? 
        torcwa.rcwa_geo.grid()
        torcwa.rcwa_geo.edge_sharpness = self.torcwa_edge_sharpness
        sim = torcwa.rcwa(freq=freq,order=[self.nG,0],L=L,dtype=sim_dtype,device=device,stable_eig_grad=False) # 4/3/25 added stable_eig_grad=False to debug jacobian not working 
            # Without this flag, self.Eig doesn't work, but with it, grad sometmies returns ill defined eigenvector error when calling grad, instead of NaN - both it seems only for orders past cutoff (tbc)
        
        ## CREATE LAYERS ##
        eps_vacuum = 1        
        sim.add_input_layer(eps=eps_vacuum) # input and output layers are eps=mu=1 by default, so this line not needed
        sim.set_incident_angle(inc_ang=theta,azi_ang=phi)     # for some reason throws an error in solve_global_smatrix if this line is before defining input layer   
        self.build_grating_torcwa()
        sim.add_layer(thickness=self.grating_depth,eps=self.grating_grid_torcwa)
        sim.add_layer(thickness=self.substrate_depth,eps=self.substrate_eps)
        sim.solve_global_smatrix()
        self.RCWA = sim
        

    def build_grating_torcwa(self):
        """
        Build the grating for the TORCWA solver using the twobox parameters
        no care taken for autograd, assuming torcwa/torch will handle this
        """
        # layers
        dy=1e-4
        L = [self.grating_pitch, dy]                    
        Lam = self.grating_pitch
        w1 = self.box1_width
        w2 = self.box2_width
        bcd = self.box_centre_dist
        x1 = w1/2 + 0.02*Lam # box1 centre location (offset to avoid left box left edge clipping)    
        x2 = x1 + bcd # box2 centre location    
        eb1 = self.box1_eps
        eb2 = self.box2_eps
        box1_bool = torcwa.rcwa_geo.rectangle(Wx=w1,Wy=L[1],Cx=x1,Cy=L[1]/2.) # width, heigh, centerx, centery
        box2_bool = torcwa.rcwa_geo.rectangle(Wx=w2,Wy=L[1],Cx=x2,Cy=L[1]/2.) # width, heigh, centerx, centery
        layer0_bool=torcwa.rcwa_geo.union(box1_bool,box2_bool)
        layer0_eps =eb1*box1_bool+eb2*box2_bool + (1.-layer0_bool)
        self.grating_grid_torcwa = layer0_eps
        try: # when called to calculate gradient functions rather than values, tensors are virtual - do not copy to grating_grid
            self.grating_grid = self.to_numpy(layer0_eps)
        except:
            self.grating_grid =np.zeros((self.Nx,0))
        return self.grating_grid

    def return_epsilon(self):
        p=self.to_numpy(self.grating_pitch)
        x0 = np.linspace(0,p,self.Nx, endpoint=False)
    
        if self.RCWA_engine == 'TORCWA':
            self.init_TORCWA() 
            #Torcwa does not need flipping this array - check x axis conventions?           
            eps_array=self.to_numpy(self.RCWA.return_layer(0,self.Nx,1)[0])
                
        elif self.RCWA_engine == 'GRCWA':
            self.init_RCWA()
            eps_array = self.RCWA.Return_eps(which_layer=1,Nx=self.Nx,Ny=self.Ny,component='xx')
            # flip to match ordering of desired eps vs grid number - 
            eps_array = np.flip(eps_array)

        return x0,eps_array
    def to_numpy(self,x):
        """ Converts tensors, autograd arrays or numpy arrays, or list or tuples of these
          (including mixed tuples) to numpy arrays (or tuples of these). For scalars, output are native python, not numpy, for 
          easier readability in print statements,
          All results are separated from gradient information.
        """        
        if isinstance(x,(list,tuple)):        
            result = []
            for item in x:
                if isinstance(item, torch.Tensor):
                    # Convert tensor to numpy array.
                    if item.numel()==1:
                        result.append(item.item())
                    else:
                        result.append(item.detach().cpu().numpy())
                elif(isinstance(item, (tuple,list))):
                    # nested tuple -> recurse
                    result.append(self.to_numpy(item)) # recurse for nested tuples
                elif ArrayBox is not None and isinstance(x, ArrayBox):
                    # autograd array
                    if item.size==1:
                        result.append(np.asarray(x).item())
                    else:
                        result.append(np.array(item))
                    
                elif isinstance(item, (np.ndarray)):
                    result.append(np.array(item))
                elif np.isscalar(item):
                    result.append(item)
                else:
                    raise TypeError(f"to_numpy Unsupported type: {type(item)}")
            if isinstance(x,tuple):
                if len(result)==1:
                    return result[0]
                else:
                    return tuple(result)        
            if isinstance(x,list):
                try:
                    return np.array(result)
                except:
                    return result
            else:
                return result
        else:        
            if(isinstance(x, torch.Tensor)):
                return x.detach().cpu().numpy()
            else:
                return np.array(x)
        
    def grating_orders(self):
        """ return list of grating orders given current wavelenth and incident angle """
        # if np.isnan(self.to_numpy(wavelength)): wavelength=self.to_numpy(self.wavelength) 
        # if np.isnan(self.to_numpy(angle)): angle=self.to_numpy(self.angle)
        angle=self.angle
        wavelength=self.wavelength
        p=self.grating_pitch
        # Calculate the maximum possible diffraction order
        m_max = self.npa.int(((p / wavelength) * (1 - self.npa.sin(angle))))
        # Initialize a list to store the valid diffraction orders
        orders = []
        # Iterate over possible diffraction orders from -m_max to m_max
        for m in range(-m_max-1, m_max + 1):
            # Calculate sin(θ_m) using the grating equation
            sin_theta_m = (m * wavelength / p) + self.npa.sin(angle)
            # Check if sin(θ_m) is within the valid range [-1, 1]
            if -1 <= sin_theta_m <= 1:
                orders.append(m)
        return orders
    
    # needed for pickling - removes autograd information, written by chatgpt
    def __getstate__(self):
        state = self.__dict__.copy()
        # remove parts that can't be pickled
        if 'RCWA' in state:
            del state['RCWA']
            del state['npa']
        return self.detach_tensors(state)
    def __setstate__(self, state):
        self.__dict__.update(state)
        # may need to add RCWA/TORCWA init, and redefine npa as these are not pickled.
    
    def detach_tensors(self,obj):
        if isinstance(obj, torch.Tensor):
            return obj.detach()
        elif isinstance(obj, list):
            return [self.detach_tensors(x) for x in obj]
        elif isinstance(obj, tuple):
            return tuple(self.detach_tensors(x) for x in obj)
        elif isinstance(obj, dict):
            return {k: self.detach_tensors(v) for k, v in obj.items()}
        elif hasattr(obj, '__dict__'):
            # If the object is a custom class instance, create a shallow copy
            # and recursively detach tensors in its __dict__
            new_obj = obj.__class__.__new__(obj.__class__)
            new_obj.__dict__ = self.detach_tensors(obj.__dict__)
            return new_obj
        else:
            return obj