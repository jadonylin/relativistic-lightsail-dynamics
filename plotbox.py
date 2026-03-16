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

import fom as fom_module
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

        x0,eps_array=self.return_epsilon()
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