"""
A module to store helpful functions and classes for plotting.

In particular, the PlotBox class contains methods for plotting the permittivity profile, 
angular efficiency, spectrum, and field distributions of a TwoBox grating.
"""

import numpy as np

import torcwa

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

import fom
from parameters import Parameters
I0, L, m, c = Parameters()

class MinorSymLogLocator(Locator):
    """
    Place minor ticks on symlog plots. 
    Dynamically find minor tick positions based on the positions of major ticks for a symlog scaling.
    From: https://stackoverflow.com/questions/20470892/how-to-place-minor-ticks-on-symlog-scale
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



class PlotBox:
    """
    Plotting methods for TwoBox gratings.
    """
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

        allowed_quantities = ("r", "t", "PDr", "PDt")
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
    
    def show_Eigs(self, wavelength_range: list=[1., 1.5],  I: float=10e9, num_plot_points: int=200, 
                eig_real_log_axis: bool=True, eig_imag_log_axis: bool=True, marker: str='o'):
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
            real, imag = fom.Eigs(self, I=I,m=m,c1=c, grad_method="grad", return_vec=False)
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

    def show_FOM_spectrum(self, angle: float=0., wavelength_range: list=[1., 1.5], num_plot_points: int=200, I: float=10e9, grad_method: str="grad"):
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
        
        wavelengths = np.linspace(*wavelength_range, num_plot_points)
        init_wavelength = self.wavelength  # record user-initialised wavelength
        inc_angle_deg = angle*180/np.pi
        
        efficiencies = np.zeros(num_plot_points, dtype=float)
        for idx, lam in enumerate(wavelengths):
            self.wavelength = lam
            efficiencies[idx] = fom.FoM(self, I=I, grad_method="grad")
        self.wavelength = init_wavelength

        fig, ax = plt.subplots(1)         
        p = self.to_numpy(self.grating_pitch)
        ax.set_xlim(wavelength_range/p)  # normalise wavelength to grating pitch
        ax.plot(wavelengths/p, efficiencies, color=(0.7, 0, 0), linestyle='-', lw=LINE_WIDTH)
        ax.set(title=rf"{self.title} $h_1' = {self.grating_depth/self.wavelength:.3f}\lambda_0$, $\Lambda' = {self.grating_pitch/self.wavelength:.3f}\lambda_0$", xlabel=r"$\lambda'/\Lambda'$", ylabel="FoM")
        ax.axhline(y=0, color='black', linestyle='-', lw = '1')
        ax.tick_params(axis='both', which='both', direction='in') # ticks inside box
        
        cm_to_inch = 0.393701
        fig_width = 20.85*cm_to_inch
        fig_height = 17.6*cm_to_inch
        fig.set_size_inches(fig_width/1.2, fig_height/1.2)
        
        return fig, ax