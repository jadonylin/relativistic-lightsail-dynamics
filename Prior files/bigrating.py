"""
Create and simulate TwoBox grating to store grating parameters and hyperparameters.

Simulation is re-run if you change instance variables.

Plotting methods to show: grating permittivity profile, spectra, fields, angle dependence
and chosen efficiency data in (box width, grating height) parameter space (for onebox cases).

Note: all grating lengths are normalised to the excitation/laser wavelength.
"""

# IMPORTS ###########################################################################################################################################################################
import autograd.numpy as npa
from autograd import grad
from autograd.scipy.special import erf as autograd_erf
from autograd.numpy import linalg as npaLA
# import jax.numpy as npa
# from jax import grad

from parameters import Parameters, gamma, D1
I0, L, m, c = Parameters()

import grcwa
grcwa.set_backend('autograd')  # important!!

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



# Smoothing if conditionals for backpropagation
def softmax(sigma,p):
    e_x = npa.exp(sigma*(p - npa.max(p)))
    return e_x/npa.sum(e_x)


class Bigrating:
    def __init__(self, grating_pitch, grating_depth, box1_width, box2_width, box_centre_dist, box1_eps, box2_eps, 
                 gaussian_width,
                 wavelength: float=1., substrate_depth: float=1., substrate_eps: float=-1e6, angle: float=0.,
                 Nx: float=1000, nG: int=25, Qabs: float=np.inf,
                 v: float=0) -> None:
        """
        Initialise twobox grating, excitation and hyperparameters.

        Parameters
        ----------
        grating_pitch   :   Grating pitch/period
        grating_depth   :   Grating layer depth/height/thickness
        box1_width      :   Left box width 
        box2_width      :   Right box width
        box_centre_dist         :   Distance between box centres
        box1_eps        :   Left box permittivity
        box2_eps        :   Right box permittivity
        gaussian_width  :   Width of gaussian beam
        wavelength      :   Excitation plane wave wavelength (laser-frame wavelength)
        substrate_depth :   Substrate layer depth/height/thickness
        substrate_eps   :   Substrate permittiivty (set to zero for no substrate)
        angle           :   Excitation plane wave angle in radians
        Nx              :   Number of grid points in the unit cell
        nG              :   Number of Fourier components
        Qabs            :   Relaxation parameter
        """
        self.grating_pitch = grating_pitch 
        self.grating_depth = grating_depth 
        self.box1_width = box1_width
        self.box2_width = box2_width
        self.box_centre_dist = box_centre_dist
        self.box1_eps = box1_eps
        self.box2_eps = box2_eps
        
        self.gaussian_width=gaussian_width

        self.wavelength = wavelength
        self.substrate_depth = substrate_depth
        self.substrate_eps = substrate_eps
        
        self.angle = angle

        self.Nx = Nx
        self.Ny = 1
        self.nG = nG
        self.Qabs = Qabs


        self.invert_unit_cell = False

        self.box_params = [self.box1_width, self.box2_width, self.box_centre_dist, self.box1_eps, self.box2_eps, self.gaussian_width]
        self.params = [self.grating_pitch, self.grating_depth] + self.box_params
        
        self.init_RCWA()
  
    def build_grating(self):
        """
        Build the grating permittivity grid as array of permittivities. 

        Does not account for boundary permittivities in the finite grid, so is not autograd-
        differentiable. 
        """

        Lam = self.grating_pitch
        w1 = self.box1_width
        w2 = self.box2_width
        bcd = self.box_centre_dist
        x1 = w1/2 + 0.02*Lam # box1 centre location (offset to avoid left box left edge clipping)    
        x2 = x1 + bcd # box2 centre location
        
        x = npa.linspace(0,Lam,self.Nx)
        idx_in_box1 = abs(x - x1) <= w1/2
        idx_in_box2 = abs(x - x2) <= w2/2
        
        # Build grating without using index assignment to satisfy autograd
        grating = [] 
        for grid_idx in range(0,self.Nx):
            if (idx_in_box1[grid_idx] and idx_in_box2[grid_idx]) or idx_in_box1[grid_idx]:
                # Overrides box 2 with box 1 if overlapping
                grating.append(self.box1_eps) 
            elif idx_in_box2[grid_idx]:
                grating.append(self.box2_eps)
            else: # vacuum permittivity
                grating.append(1)
        
        if self.invert_unit_cell:
            self.grating_grid = npa.array(grating)[::-1]
        else:
            self.grating_grid = npa.array(grating)

        return npa.array(grating)

    def build_grating_gradable(self, sigma: float=100.):
        """
        Build the grating permittivity grid as array of permittivities. 
        
        Since the RCWA is grid-based, continuous changes in box widths or positions must be
        handled carefully. Here, permittivities are chosen continuously using a softmax 
        probability weighting depending on how far away each grid is from the centre of the 
        boxes. Softmax ensures this array of permittivities is autograd-differentiable. A
        consequence of the softmax is that the boxes are smoother than they should be
        (smoothness increasing with the temperature parameter 1/sigma).

        Builds box1 as far to the left in the unit cell as possible then fits box2 afterwards. 
        This ensures that large boxes can fit inside the unit cell.

        Parameters
        sigma :   Softmax inverse temperature, i.e. inverse smoothing factor. Smaller means smoother grating.
        """
        
        Lam = self.grating_pitch
        w1 = self.box1_width
        w2 = self.box2_width
        bcd = self.box_centre_dist
        x1 = w1/2 + 0.02*Lam # box1 centre location (offset to avoid left box left edge clipping)    
        x2 = x1 + bcd # box2 centre location    
        eb1 = self.box1_eps
        eb2 = self.box2_eps

        box_separation = bcd - (w1 + w2)/2
        boxes_midpoint = (x1 + w1/2 + x2 - w2/2)/2

        dx = Lam/self.Nx # grid spacing
        grid_left_boundaries = npa.linspace(0,Lam-dx,self.Nx) # does not include x = Lam boundary
        # In this formulation, grid numbers 0, 1, ..., Nx-1 refer to the left boundaries (consistent with x position from 0 to 1*pitch)
        box1_left_boundary = x1-w1/2
        box2_right_boundary = x2+w2/2

        # Build grating without using index assignment to ensure gradable
        grating = [] 
        for grid_left_boundary in grid_left_boundaries:
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
        Calculate GRWCA object for the twobox.
        """
        # Grating
        # To simulate a 1D grating, take a small periodicity in the y-direction. 
        # Note: As mentioned in GRCWA documentation, can only differentiate wrt 
        # photonic crystal period if the ratio of periodicities in the two in-plane 
        # directions (x and y) is fixed. GRCWA encodes this by scaling both 
        # (reciprocal) lattice vectors after they've been created in the kbloch.py
        # module. Hence, set unity grating vector here and use Pscale kwarg in
        # Init_Setup() to scale the period accordingly.
        dy = 1e-4 
        L1 = [1.,0]
        L2 = [0,dy] 

        freq = 1/self.wavelength # freq = 1/wavelength when c = 1
        freqcmp = freq*(1+1j/2/self.Qabs)

        # Incoming wave
        theta = self.angle # radians
        phi = 0.

        # setup RCWA
        obj = grcwa.obj(self.nG,L1,L2,freqcmp,theta,phi,verbose=0)


        ## CREATE LAYERS ##
        # Layer depth
        eps_vacuum = 1
        vacuum_depth = self.wavelength

        # Construct layers
        obj.Add_LayerUniform(vacuum_depth,eps_vacuum)
        obj.Add_LayerGrid(self.grating_depth,self.Nx,self.Ny)
        if self.substrate_eps != 0:
            obj.Add_LayerUniform(self.substrate_depth,self.substrate_eps)
        obj.Add_LayerUniform(vacuum_depth,eps_vacuum)
        obj.Init_Setup(Pscale=self.grating_pitch)

        # Construct patterns
        # TODO: inefficient because this re-builds the grating every time we calculate RT
        self.build_grating_gradable() # to update twobox whenever user changes box parameters
        obj.GridLayer_geteps(self.grating_grid)


        ## INCIDENT WAVE ##
        planewave={'p_amp':0,'s_amp':1,'p_phase':0,'s_phase':0}
        obj.MakeExcitationPlanewave(planewave['p_amp'],planewave['p_phase'],planewave['s_amp'],planewave['s_phase'],order = 0)

        self.RCWA = obj
        return obj

    def RT(self):
        """
        Calculate up to -1 <= m <= 1 orders' reflection/transmission for the twobox.
        """
        self.init_RCWA()
        R_byorder,T_byorder = self.RCWA.RT_Solve(normalize=1, byorder=1)
        Fourier_orders = self.RCWA.G

        Rs = []
        Ts = []
        RT_orders = [-1,0,1]
        # IMPORTANT: have to use append method to a list rather than index assignment
        # Else, autograd will throw a TypeError with float() argument being an ArrayBox
        for order in RT_orders:
            Rs.append(npa.sum(R_byorder[Fourier_orders[:,0]==order]))
            Ts.append(npa.sum(T_byorder[Fourier_orders[:,0]==order]))

        return Rs,Ts


    def return_Qs(self):
        """
        Calculate efficiency factors and their derivatives
        """

        def eff(self):
            """
            Calculate reflection and transmission coefficients 
            """
            self.init_RCWA()
            R_byorder,T_byorder = self.RCWA.RT_Solve(normalize=1, byorder=1)
            Fourier_orders = self.RCWA.G

            Rs = []
            Ts = []
            RT_orders = [-1,0,1]
            # IMPORTANT: have to use append method to a list rather than index assignment
            # Else, autograd will throw a TypeError with float() argument being an ArrayBox
            for order in RT_orders:
                Rs.append(npa.sum(R_byorder[Fourier_orders[:,0]==order]))
                Ts.append(npa.sum(T_byorder[Fourier_orders[:,0]==order]))

            return Rs,Ts

        def Q(self):
            """
            Calculates efficiency factors
            """

            r,t=eff(self)
            def beta_m(m,self):
                test=(npa.sin(self.angle)+m*self.wavelength/self.grating_pitch)
                if abs(test)>=1:
                    delta_m="no_diffraction_order"
                else:
                    delta_m=npa.arcsin(test)
                return delta_m
            Q1=0
            Q2=0
            M=[-1,0,1]
            for m in range(len(M)):
                delta_m=beta_m(M[m],self)
                if isinstance(delta_m,str):
                    Q1=Q1+0
                    Q2=Q2+0
                else:
                    Q1=Q1+ r[m]*(1+npa.cos(self.angle+delta_m))+t[m]*(1-npa.cos(delta_m-self.angle))
                    Q2=Q2+ r[m]*npa.sin(self.angle+delta_m)+t[m]*npa.sin(delta_m-self.angle)
            Q1=npa.cos(self.angle)*Q1
            Q2=-npa.cos(self.angle)*Q2
            return Q1,Q2

        ## Saving current angle, wavelength
        input_angle=self.angle
        input_wavelength=self.wavelength

        ## Centred 
        Q1,Q2=Q(self)

        ########### Angle
        h_angle=1e-4
        ## Backwards angle
        self.angle=input_angle - h_angle
        Q1_back_angle,Q2_back_angle=Q(self)

        ## Forwards angle
        self.angle=input_angle + h_angle
        Q1_forwards_angle,Q2_forwards_angle=Q(self)

        ########### Wavelength
        self.angle=input_angle
        h_wavelength=1e-4
        ## Backwards angle
        self.wavelength=input_wavelength - h_wavelength
        Q1_back_wavelength,Q2_back_wavelength=Q(self)

        ## Forwards angle
        self.wavelength=input_wavelength + h_wavelength
        Q1_forwards_wavelength,Q2_forwards_wavelength=Q(self)

        ########### Restore
        self.angle=input_angle
        self.wavelength=input_wavelength

        ########### Derivatives
        PD_Q1_angle=(Q1_forwards_angle - Q1_back_angle) / (2 * h_angle)
        PD_Q2_angle=(Q2_forwards_angle - Q2_back_angle) / (2 * h_angle)
        
        PD_Q1_wavelength=(Q1_forwards_wavelength - Q1_back_wavelength) / (2 * h_wavelength)
        PD_Q2_wavelength=(Q2_forwards_wavelength - Q2_back_wavelength) / (2 * h_wavelength)

        return Q1,Q2,PD_Q1_angle,PD_Q2_angle,PD_Q1_wavelength,PD_Q2_wavelength

    def FoM(self,v):
        Q1,Q2,PD_Q1_angle,PD_Q2_angle,PD_Q1_wavelength,PD_Q2_wavelength=self.return_Qs()
        w=self.gaussian_width
        w_bar=w/L

        
        D=D1(v) # could manage to make this 1/l somehow
        g=gamma(v) # g=(l**2 + 1)/(2*l)
        lam=self.wavelength  # needs to be lambda'

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
        
        ## Sum of eigenvalues
        FD=npa.sum(EIGreal)

        #FD=npa.prod(EIGreal)    # can't use npa.unique

        # Attempting average of product
        # Could still have (negative + ib, negative -ib, c, d) where c,d not complex pairs
        # NEED c,d to be complex conjugate pairs

        return FD


    def Eigs(self,v):
        Q1,Q2,PD_Q1_angle,PD_Q2_angle,PD_Q1_wavelength,PD_Q2_wavelength=self.return_Qs()
        w=self.gaussian_width
        w_bar=w/L

        D=D1(v) # could manage to make this 1/l somehow
        g=gamma(v) # g=(l**2 + 1)/(2*l)
        lam=self.wavelength  # needs to be lambda'

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
        eigReal=npa.real(eig)
        eigimag=npa.imag(eig)

        return eigReal,eigimag




    def rNeg1(self, angle):
        self.angle = angle
        r = self.RT()[0][0]
        return r
    def tNeg1(self, angle):
        self.angle = angle
        t = self.RT()[1][0]
        return t
    
    def PDrNeg1(self, angle):
        return grad(self.rNeg1)(angle)
    def PDtNeg1(self, angle):
        return grad(self.tNeg1)(angle)
            

    def calculate_y_fields(self, height):
        """
        Return Ey on the grid at a fixed height

        Parameters
        ----------
        height :   Height to calculate field
        """
        self.init_RCWA()
        fields = self.RCWA.Solve_FieldOnGrid(1,height)
        Efield = fields[0]
        Ey = np.transpose(Efield[1])
        return Ey
    
    def show_permittivity(self, show_analytic_box: bool=False):
        """
        Show permittivity profile for the twobox.
        Returns the horizontal coordinates and corresponding epsilon values, and figure/axes objects.
        Note: GRCWA does not always align the returned permittivity with the grid numbers, it may be displaced by some grids.
        This is not an issue because the unit cell can be arbitrarily shifted provided you maintain periodic boundary conditions

        Parameters
        ----------
        show_analytic_box :   Show analytic boxes overlaid onto boundary-permittivity-accounted-for boxes
        """
        self.init_RCWA()

        x0 = np.linspace(0,self.grating_pitch,self.Nx, endpoint=False)
    
        eps_array = self.RCWA.Return_eps(which_layer=1,Nx=self.Nx,Ny=self.Ny,component='xx')
        eps_array_real = eps_array.real

        # Show actual eps vs grid number and flip to match ordering of desired eps vs grid number 
        grids = np.arange(0, self.Nx, 1)
        eps_array_real = np.flip(eps_array_real)


        ## Plot ##
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
        Show grating efficiencies vs excitation angle for the twobox.

        Parameters
        ----------
        theta_max       :   Plot angles up to theta_max (degrees)
        num_plot_points :   Number of points to plot
        """
        ### Setup ###
        init_angle = self.angle # record user-initialised angle
        inc_angles = np.pi/180*np.linspace(-theta_max, theta_max, num_plot_points) 

        ## CALCULATE EFFICIENCY ##
        efficiencies = np.zeros((6,num_plot_points), dtype=float)
        # Rows of efficiencies correspond to the order of diffraction (row,order): 
        # Reflection: (0,-1), (1,0), (2,1)
        # Transmission: (3,-1), (4,0), (5,1)
        for idx, theta in enumerate(inc_angles):
            # Calculate efficiencies for each order
            self.angle = theta
            Rs,Ts = self.RT()
            efficiencies[:3,idx] = Rs
            efficiencies[3:,idx] = Ts
        self.angle = init_angle # reset user-initialised angle

        ### PLOTTING ### 
        fig, ax = plt.subplots(1)           

        ## Plot efficiency vs incoming angle ##
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

        # Sum
        eff_sum = np.sum(efficiencies[:6,:],axis=0)
        ax.plot(inc_angles, eff_sum, color=(0, 0.7, 0), linestyle='-', label=r"$\Sigma (r_i + t_i)$", lw = LINE_WIDTH) 

        # Axis labels
        ax.set(title=rf"$\Lambda' = {self.grating_pitch/self.wavelength:.3f}\lambda$, $h_1' = {self.grating_depth/self.wavelength:.3f}\lambda$"
            , xlabel="Incident angle (°)"
            , ylabel="Efficiency")

        leg = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        frame = leg.get_frame()
        frame.set_edgecolor('black')

        # Modify axes
        ax.set_xlim([-theta_max, theta_max])
        ax.set_ylim([-0.01, 1.01]) # displace from zero to see the transition to evanescence
        cm_to_inch = 0.393701
        fig_width = 20.85*cm_to_inch
        fig_height = 17.6*cm_to_inch
        fig.set_size_inches(fig_width/1.2, fig_height/1.2)
        
        return fig, ax

    def show_spectrum(self, angle: float=0., efficiency_quantity: str="PDr", wavelength_range: list=[1., 1.5], num_plot_points: int=200):
        """
        Show grating spectrum for the twobox.

        Parameters
        ----------
        angle               :   Angle of incident plane wave excitation (radians)
        efficiency_quantity :   Which efficiency quantity you want spectrum for ("r" - reflection, "PDr" - reflection angular derivative, "t" - transmission, "PDr" - transmission angular derivative, "FoM" - figure of merit)
        wavelength_range    :   Wavelength range to plot spectrum (same units as grating pitch, chosen by you)
        num_plot_points     :   Number of points to plot
        """
        ### Setup ###
        allowed_quantities = ("r", "t", "PDr", "PDt","FoM","eig")
        if efficiency_quantity not in allowed_quantities:
            invalid_quantity_message = f"Invalid efficiency quantity. Allowed quantities are: {allowed_quantities}"
            raise ValueError(invalid_quantity_message)
        
        wavelengths = np.linspace(*wavelength_range, num_plot_points)
        init_wavelength = self.wavelength # record user-initialised wavelength
        inc_angle_deg = angle*180/np.pi


        ## CALCULATE EFFICIENCY ##
        RT_orders = [-1,0,1]
        n_orders = len(RT_orders)
        efficiencies = np.zeros((2*n_orders,num_plot_points), dtype=float)
        Reig1= np. zeros( (1,num_plot_points) , dtype=float)
        Reig2= np. zeros( (1,num_plot_points) , dtype=float)
        Reig3= np. zeros( (1,num_plot_points) , dtype=float)
        Reig4= np. zeros( (1,num_plot_points) , dtype=float)
        Ieig1= np. zeros( (1,num_plot_points) , dtype=float)
        Ieig2= np. zeros( (1,num_plot_points) , dtype=float)
        Ieig3= np. zeros( (1,num_plot_points) , dtype=float)
        Ieig4= np. zeros( (1,num_plot_points) , dtype=float)
        # Rows of efficiencies correspond to the order of diffraction (row,order): 
        # Reflection: (0,-1), (1,0), (2,1)
        # Transmission: (3,-1), (4,0), (5,1)
        for idx, lam in enumerate(wavelengths):
            # Calculate efficiencies for each order
            self.wavelength = lam
            if efficiency_quantity == "r" or efficiency_quantity == "t":
                Rs,Ts = self.RT()
                efficiencies[:n_orders,idx] = Rs
                efficiencies[n_orders:,idx] = Ts
            elif efficiency_quantity == "PDr":
                efficiencies[0,idx] = self.PDrNeg1(angle)
            elif efficiency_quantity == "PDt":
                efficiencies[0,idx] = self.PDtNeg1(angle)

            elif efficiency_quantity == "FoM":
                efficiencies[0,idx] = self.FoM()
            elif efficiency_quantity == "eig":
                real,imag=self.Eigs()
                Reig1[0,idx] = real[0]
                Reig2[0,idx] = real[1]
                Reig3[0,idx] = real[2]
                Reig3[0,idx] = real[3]

                Ieig1[0,idx] = imag[0]
                Ieig2[0,idx] = imag[1]
                Ieig3[0,idx] = imag[2]
                Ieig4[0,idx] = imag[3]
        self.wavelength = init_wavelength


        ### PLOTTING ### 
        # Set up figure
        fig, ax = plt.subplots(1)         
        p = self.grating_pitch
        ax.set_xlim(np.array(wavelength_range)/p) # normalise to grating pitch
        legend_needed = ("r", "t")
        symlog_needed = ("PDr", "PDt","FoM")

        ## Plot efficiency vs wavelength ##
        if efficiency_quantity == "r":
            # -1 order
            ax.plot(wavelengths/p, efficiencies[0], color=(0.7, 0, 0), linestyle='-', label="$r_{-1}'$", lw = LINE_WIDTH) 
            # 0 order
            ax.plot(wavelengths/p, efficiencies[1], color='0.4', linestyle='-', label="$r_0'$", lw = LINE_WIDTH) 
            # 1 order
            ax.plot(wavelengths/p, efficiencies[2], color=(0, 0, 0.7), linestyle='-', label="$r_1'$", lw = LINE_WIDTH) 
            ax.set_ylim([-0.01, 1.01]) # displace from zero to see the transition to evanescence
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
            ax.set_ylim([-0.01, 1.01]) # displace from zero to see the transition to evanescence
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

        elif efficiency_quantity=="eig":
            colorReal=(0.7, 0, 0)
            ax.plot(wavelengths/p,Reig1[0], 'o', markersize=0.5, markerfacecolor=colorReal, fillstyle='full',  color=colorReal)
            ax.plot(wavelengths/p,Reig2[0], 'o', markersize=0.5, markerfacecolor=colorReal, fillstyle='full',  color=colorReal)
            ax.plot(wavelengths/p,Reig3[0], 'o', markersize=0.5, markerfacecolor=colorReal, fillstyle='full',  color=colorReal)
            ax.plot(wavelengths/p,Reig4[0], 'o', markersize=0.5, markerfacecolor=colorReal, fillstyle='full',  color=colorReal)
            ylabel=rf"Eigenvalues"

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
        ax.set(title=rf"$h_1' = {self.grating_depth/self.wavelength:.3f}\lambda_0$, $\Lambda' = {self.grating_pitch/self.wavelength:.3f}\lambda_0$", xlabel=r"$\lambda'/\Lambda'$", ylabel=ylabel)

        # Modify axes
        cm_to_inch = 0.393701
        fig_width = 20.85*cm_to_inch
        fig_height = 17.6*cm_to_inch
        fig.set_size_inches(fig_width/1.2, fig_height/1.2)
        
        return fig, ax
    

    def show_depth_dependence(self, angle: float=0., efficiency_quantity: str="PDr", depth_range: list=[0., 1.], num_plot_points: int=200):
        """
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
                Rs,Ts = self.RT()
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
    
    def show_fields(self, heights: np.ndarray, field_output: str="real", fill_style: str="gouraud", show_eps_profile: bool=False):
        """
        Show out-of-plane electric field at given heights for the twobox.

        Parameters
        ----------
        field_output     :   "real", "square" or "abs" fields
        heights          :   Heights to calculate fields (not normalised)
        fill_style       :   Field plot shading style
        show_eps_profile :   Overlay epsilon profile onto fields 
        """
        ## CALCULATE FIELDS ##
        Eys = np.zeros((len(heights), self.Nx), dtype=np.complex128) # need complex to assign complex Ey to Eys
        for idx, d in enumerate(heights):
            Eys[idx,:] = self.calculate_y_fields(d)

        if field_output == "real":
            Eys = np.real(Eys)
            cbar_label = r"$\Re(E_y)$"
            max_colour_scale = np.maximum(np.abs(np.min(Eys)), np.abs(np.max(Eys)))
            vmax = max_colour_scale
            vmin = -max_colour_scale
        elif field_output == "abs":
            Eys = np.abs(Eys)
            cbar_label = r"$|E_y|$"
            max_colour_scale = np.max(Eys)
            vmax = max_colour_scale
            vmin = 0
        elif field_output == "square":
            Eys = np.power(np.real(Eys),2)
            cbar_label = r"$\Re(E_y)^2$"
            max_colour_scale = np.max(Eys)
            vmax = max_colour_scale
            vmin = 0
        else:
            return print("Field output form not valid. Must be 'real' or 'abs'.")
        Eys = Eys[::-1]
        
        # In the plot, the incident light comes from positive z with this convention
        # Can check by setting the resonator to -ve permittivity
        
        x0 = np.linspace(-0.5,0.5,self.Nx)

        ## Plot ##
        fig, axs = plt.subplots(nrows=1, ncols=1)
        
        if show_eps_profile:
            axs2 = axs.twinx()
            
            # eps_array = self.RCWA.Return_eps(1,self.Nx,self.Ny,component='xx')
            # eps_array = np.flip(eps_array.real)
            eps_array = self.grating_grid
            
            eps_min = np.min(eps_array)
            eps_max = np.max(eps_array)
            
            ## Plot ## 
            eps_color = 'b'
            axs2.plot(x0, eps_array, color=eps_color)
            axs2.set(ylabel=r"$\varepsilon$")
            axs2.set_ylim(bottom=eps_min, top=eps_max)
            axs2.yaxis.label.set_color(eps_color)
        E_mesh = axs.pcolormesh(x0, heights/self.grating_pitch, Eys, vmin=vmin, vmax=vmax, shading=fill_style, cmap='hot')

        # create an axes on the right side of ax. The width of cax will be x%
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
        # axs.set_ylim(bottom=np.min(heights), top=np.max(heights))
        # axs.set_aspect('equal')
        
        fig_mult = 4
        fig_width = 9
        if self.grating_depth/self.grating_pitch < 0.2:
            fig_height = 3*self.grating_depth/self.grating_pitch*fig_mult
        else:
            fig_height = self.grating_depth/self.grating_pitch*fig_mult
        fig.set_size_inches(fig_width, fig_height)
        
        return fig, axs