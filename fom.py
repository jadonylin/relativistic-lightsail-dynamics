"""
A module to store figure of merit (FoM) functions and helper functions that deal
with linear stability analysis (LSA) of the twobox. 

User figure of merit functions should be defined here.

TODO: need better separation between opt and fom, otherwise figures of merit are mixed between them
"""

import numpy as np

import matplotlib.pyplot as plt
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

from parameters import Parameters
I0, L, m, c = Parameters()
from plothelp import MinorSymLogLocator
from twobox import TwoBox


def FoM(grating: TwoBox, I: float=1e9, grad_method: str="finite") -> float:
    """
    Calculate the grating single-wavelength figure of merit F_lam.

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
    grating     :   Calculate figure of merit for this grating
    I           :   Laser intensity
    grad_method :   Method to calculate gradient ("finite","grad"). Must be "finite" for optimisation
    
    Returns
    -------
    F_lam :   Figure of merit
    """
    
    eigReal, eigImag = Eigs(grating, I=I, m=m, c1=c, grad_method=grad_method, return_vec=False)

    # MdS FoM: Minimise the eigenvalue with the largest real part. Equivalent to maximising the 
    #          negative eigenvalue with the smallest real part. 
    F_lam = grating.npa.min(-eigReal)  # standard minimum
    # F_lam = grating.npa.sum(-eigReal*grating.npa.softmin(-eigReal,1.))  # softened minimum
    # F_lam = grating.npa.min(-eigReal) + grating.npa.max(-eigReal)
    
    return F_lam

def FoM_quality_factor(grating: TwoBox, I: float=1e9, grad_method: str="finite") -> float:
    """
    Calculate the grating single-wavelength figure of merit F_lam.

    Quality factor FoM: Maximise the magnitude of the quality factor (Re(xi)/Im(xi)) 
                        for the eigenvalue with the smallest quality factor. Issue:
                        Im(xi) --> 0 will blow this up, and we need to track the sign.

    Parameters
    ----------
    grating     :   Calculate figure of merit for this grating
    I           :   Laser intensity
    grad_method :   Method to calculate gradient ("finite","grad"). Must be "finite" for optimisation
    
    Returns
    -------
    F_lam :   Figure of merit
    """
    
    raise NotImplementedError("Must determine how to handle signs and avoid Im(xi) = 0.")

def FoM_LvR(grating: TwoBox, I: float=1e9, grad_method: str="finite") -> float:
    """
    Last FoM implemented by Liam - not working with TORCWA
    Calculate the grating single-wavelength figure of merit F_lam using LvR's most updated method.

    Parameters
    ----------
    grating     :   Calculate figure of merit for this grating
    I           :   Laser intensity
    grad_method :   Method to calculate gradient ("finite","grad"). Must be "finite" for optimisation
    
    Returns
    -------
    F_lam :   Figure of merit
    """
    
    eigReal, eigImag = Eigs(grating, I=I, m=m, c1=c, grad_method=grad_method, return_vec=False)

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
        sorted_x = grating.npa.sort(x.flatten())
        unique_values = sorted_x[grating.npa.concatenate(([True], grating.npa.diff(sorted_x) != 0))]

        # Append filled_value as needed
        k = len(unique_values)
        for i in range(4-k):
            unique_values = grating.npa.append(unique_values,filled_value)

        return unique_values

    # NOTE: In the following penalty and reward terms, all operations must be done element-wise to avoid 
    #       "RuntimeWarning: invalid value encountered in divide" during optimisation
    # TODO: Determine why we can't use npa functions here

    # LvR FoM: Reward all Re(eig) being negative
    # Fill repeated entries in eigReal with -1 so that, after squaring, they don't influence the product
    eig_real_unique     =   unique_filled(eigReal, -1)
    eig_real_neg_unique =   grating.npa.minimum(0., eig_real_unique)
    func_real_neg_array =   grating.npa.power(eig_real_neg_unique, 2)
    func_real_neg       =   func_real_neg_array[0] * func_real_neg_array[1] * func_real_neg_array[2] * func_real_neg_array[3]
    # func_real_neg       =   npa.prod(func_real_neg_array) 

    # Remove Re(eig)<0 contribution if no restoring behaviour
    # log(1+x^2) chosen as a smooth function that moves away from zero
    # NOTE: This function has zero gradient at x=0, which is bad for stepping away from zero imaginary 
    #       part. Also, the gradient saturates at large x, which doesn't matter in the sense of 
    #       needng the imaginary part to be nonzero.
    func_imag_array     =   grating.npa.log(1 + grating.npa.power(eigImag,2))
    func_imag           =   func_imag_array[0] * func_imag_array[1] * func_imag_array[2] * func_imag_array[3]
    # func_imag           =   npa.prod(func_imag_array)

    # Penalise mixed positive and negative Re(eig)
    # Fill repeated entries in eigReal with 0 so that they don't influence the sum
    real_unique_0       =   unique_filled(eigReal, 0.)
    neg_array           =   grating.npa.power(grating.npa.minimum(0.,real_unique_0), 2)
    pos_array           =   grating.npa.power(grating.npa.maximum(0.,real_unique_0), 2)
    # penalty             =   npa.sum(neg_array) * npa.sum(pos_array)
    neg_sum             =   neg_array[0] + neg_array[1] + neg_array[2] + neg_array[3]
    pos_sum             =   pos_array[0] + pos_array[1] + pos_array[2] + pos_array[3]
    penalty             =   neg_sum * pos_sum

    # Penalise all positive Re(eig)
    # Fill repeated entries in eigReal with 1 so that they don't influence the product
    real_unique_1       =   unique_filled(eigReal, 1)
    all_pos_array       =   grating.npa.power(grating.npa.maximum(0.,real_unique_1), 2)
    penalty2            =   all_pos_array[0] * all_pos_array[1] * all_pos_array[2] * all_pos_array[3]
    # penalty2            =   npa.prod(all_pos_array)


    F_lam = func_real_neg * func_imag - penalty - penalty2
    return F_lam



def sail_stiffness(grating: TwoBox, I: float=10e9, m: float=1/1000, c1:float=299792458, 
                   grad_method: str='finite', out: str="tr", normalise: bool=False):
    """
    Calculate stiffness coefficients/Jacobian coefficients for a symmetric lightsail at equilibrium. Here, symmetric 
    means symmetric with respect to reflections about the laser-beam axis (when the CoM lies on the laser-beam axis).

    Parameters
    ----------
    grating     :   Calculate stiffnesses for this grating
    I           :   Laser intensity
    m           :   Spacecraft mass (sail membrane + payload)
    c1          :   speed of light  # TODO: why is this a parameter?
    grad_method :   Method to calculate gradient ("finite","grad"). Must be "finite" for optimisation
    out         :   Output format 
                    "tr" for translation coefficients first, then rotation coefficients. Use when outputting to Jacobian.
                    "rd" for restoring coefficients first, then damping coefficients
    normalise   :   Normalise all Jacobian coefficients by their individual dimensional factors
    
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
            Q1R, Q2R, dQ1ddeltaR, dQ2ddeltaR, dQ1dlambdaR, dQ2dlambdaR = grating.return_Qs(h_angle, h_wavelength)
        case "grad":
            Q1R, Q2R, dQ1ddeltaR, dQ2ddeltaR, dQ1dlambdaR, dQ2dlambdaR = grating.return_Qs_auto(return_Q=True)
        case _:
            raise ValueError("grad_method not recognised. Must be 'finite' or 'grad'.")

    w = grating.gaussian_width
    w_bar = w/L  # width normalised to total grating length
    lam = grating.wavelength 

    # Convert velocity factors to wavelength factors
    # TODO: may need to change these factors to account for non-unity starting wavelengths
    D = 1/lam  # Doppler factor assuming starting wavelength is 1
    g = (grating.npa.power(lam,2) + 1)/(2*lam)  # Lorentz factor

    # Lightsail reflection-symmetry conditions
    Q1L = Q1R                ; Q2L = -Q2R;   
    dQ1ddeltaL  = -dQ1ddeltaR; dQ2ddeltaL  = dQ2ddeltaR
    dQ1dlambdaL = dQ1dlambdaR; dQ2dlambdaL = -dQ2dlambdaR        
    
    # y acceleration terms
    # NOTE: derivatives with respect to lambda differ from derivatives with respect to frequency offset, the latter
    # being presented in Liam's thesis
    fy_y      = - D**2 * I/(m*c1) * (Q2R - Q2L) * (1 - grating.npa.exp(-1/(2*w_bar**2)))
    fy_phi    = - D**2 * I/(m*c1) * (dQ2ddeltaR + dQ2ddeltaL) * w/2 * np.sqrt(np.pi/2) * grating.npa.erf(1/(w_bar*np.sqrt(2)))
    fy_vy     = - D**2 * I/(m*c1) * 1/c1 * (D+1)/(D*(g+1)) * (Q1R + Q1L + dQ2ddeltaR + dQ2ddeltaL) * w/2 * np.sqrt(np.pi/2) * grating.npa.erf(1/(w_bar*np.sqrt(2)))
    fy_phidot =   D**2 * I/(m*c1) * 1/c1 * (2*(Q2R - Q2L) - lam*(dQ2dlambdaR - dQ2dlambdaL)) * (w/2)**2 * (1 - grating.npa.exp(-1/(2*w_bar**2)))

    # phi acceleration terms
    J = m*L**2/12  # moment of inertia about the CoM
    fphi_y      =  D**2 * I/(J*c1) * (Q1R + Q1L) * (w/2*np.sqrt(np.pi/2) * grating.npa.erf(1/(w_bar*np.sqrt(2))) - L/2*grating.npa.exp(-1/(2*w_bar**2))) 
    fphi_phi    =  D**2 * I/(J*c1) * (dQ1ddeltaR - dQ1ddeltaL - (Q2R - Q2L)) * (w/2)**2 * (1 - grating.npa.exp(-1/(2*w_bar**2)))
    fphi_vy     =  D**2 * I/(J*c1) * 1/c1 * (D+1)/(D*(g+1)) * (dQ1ddeltaR - dQ1ddeltaL - (Q2R - Q2L)) * (w/2)**2 * (1 - grating.npa.exp(-1/(2*w_bar**2)))
    fphi_phidot = -D**2 * I/(J*c1) * 1/c1 * (2*(Q1R + Q1L) - lam*(dQ1dlambdaR + dQ1dlambdaL)) * (w/2)**2 * (w/2*np.sqrt(np.pi/2) * grating.npa.erf(1/(w_bar*np.sqrt(2))) - L/2*grating.npa.exp(-1/(2*w_bar**2))) 

    if normalise:
        fy_y        /= I/(m*c1)
        fy_phi      /= I*L/(m*c1)
        fy_vy       /= I*L/(m*c1**2)
        fy_phidot   /= I*L**2/(m*c1**2)
        fphi_y      /= I*L/(J*c1)
        fphi_phi    /= I*L**2/(J*c1)
        fphi_vy     /= I*L**2/(J*c1**2)
        fphi_phidot /= I*L**3/(J*c1**2)

    match out:
        case "tr":
            return grating.npa.stack((fy_y, fy_phi, fy_vy, fy_phidot, fphi_y, fphi_phi, fphi_vy, fphi_phidot))
        case "rd":
            return grating.npa.stack((fy_y, fy_phi, fphi_y, fphi_phi, fy_vy, fy_phidot, fphi_vy, fphi_phidot))
        case "mat":
            row1 = grating.npa.stack((fy_y, fy_phi,fy_vy, fy_phidot))
            row2 = grating.npa.stack((fphi_y, fphi_phi, fphi_vy, fphi_phidot))
            mat = grating.npa.stack((row1,row2))
            return mat
        case _:
            raise ValueError("Invalid output format. Must be 'tr' or 'rd'.")
        
def Eigs(grating: TwoBox, I: float=10e9, m: float=1/1000, c1:float=299792458, 
         grad_method: str='finite', return_vec: bool = False, normalise: bool=False):
    """
    Calculate eigendecomposition of Jacobian matrix at equilibrium

    Parameters
    ----------
    grating     :   Calculate eigenvalues for this grating
    I           :   Laser intensity
    m           :   Spacecraft mass (sail membrane + payload)
    c1          :   speed of light  # TODO: why is this a parameter?
    grad_method :   Method to calculate gradient ("finite","grad"). Must be "finite" for optimisation
    return_vec  :   If true, return eigenvectors as well as eigenvalues
    normalise   :   Normalise all Jacobian coefficients by their individual dimensional factors
    
    Returns
    -------
    eigReal :   Real part of Jacobian eigenvalues
    eigImag :   Imaginary part of Jacobian eigenvalues
    eigvecs :   Eigenvectors of Jacobian matrix, normalised to unit length
    """
    stiffnesses = sail_stiffness(grating,I,m,c1,grad_method,out="mat",normalise=normalise)
    J = grating.npa.concatenate((grating.npa.array([[0,0,1,0],[0,0,0,1]]), stiffnesses))  # Jacobian matrix
    if return_vec:
        eigvals, eigvecs = grating.npa.eig(J)
        eigReal = grating.npa.real(eigvals)
        eigImag = grating.npa.imag(eigvals)
        return eigReal, eigImag, eigvecs
    else:
        if grating.RCWA_engine == "TORCWA":
            eigvals = grating.npa.eigvals(J)
        else:  # eigvals is not differentiable using HIPS/autograd in GRCWA
            eigvals, _ = grating.npa.eig(J)
        eigReal = grating.npa.real(eigvals)
        eigImag = grating.npa.imag(eigvals)
        return eigReal, eigImag


def lsa_info(grating: TwoBox, I: float=0.5e9, normalise: bool=False):
    """
    Calculate quantities relevant to linear stability analysis (LSA) of the twobox dynamics. Also calculates
    the radiation pressure cross sections and their derivatives.

    Parameters
    ----------
    grating   :   Calculate linear-stability info for this grating
    I         :   Incident light intensity
    normalise :   Normalise all Jacobian coefficients by their individual dimensional factors
    
    Returns
    -------
    efficiencies :   Radiation pressure cross sections and their derivatives  
    rest_coeffs  :   Restoring force/torque coefficients
    damp_coeffs  :   Damping force/torque coefficients
    eigReal      :   Real component of eigenvalues
    eigImag      :   Imaginary component of eigenvalues
    """
    efficiencies = tuple(grating.return_Qs_auto(return_Q=True))
    stiffnesses = sail_stiffness(grating,I,m,c,grad_method="grad",out="rd",normalise=normalise)
    rest_coeffs = tuple([*stiffnesses[:4]])
    damp_coeffs = tuple([*stiffnesses[4:]])
    eigReal, eigImag, eigvecs = Eigs(grating,I,m,c,grad_method="grad",return_vec=True,normalise=normalise)
    return efficiencies, rest_coeffs, damp_coeffs, eigReal, eigImag, eigvecs


def show_Eigs(grating: TwoBox, wavelength_range: list=[1., 1.5],  I: float=10e9, num_plot_points: int=200, 
              eig_real_log_axis: bool=True, eig_imag_log_axis: bool=True, marker: str='o', normalise: bool=False):
    """
    Show eigenvalue spectrum for the twobox.

    Parameters
    ----------
    grating             :   Calculate eigenvalues for this grating
    wavelength_range    :   Wavelength range to plot spectrum (same units as grating pitch)
    I                   :   Laser intensity
    num_plot_points     :   Number of points to plot
    eig_real_log_axis   :   If true, logarithmic scale for real part of eigenvalues
    eig_imag_log_axis   :   If true, logarithmic scale for imaginary part of eigenvalues
    marker              :   Marker style passed to plt.plot()
    normalise           :   Normalise all Jacobian coefficients by their individual dimensional factors

    Returns
    -------
    fig         :   Spectrum figure object
    (ax1, ax2)  :   Real and imaginary spectrum axis objects
    """

    wavelengths = np.linspace(*wavelength_range, num_plot_points)
    init_wavelength = grating.wavelength  # record user-initialised wavelength

    ## CALCULATE EIGS ##
    eigvals = grating.npa.zeros((4,num_plot_points), dtype=np.complex128)
    
    for idx, lam in enumerate(wavelengths):
        # Calculate eigs for each order
        grating.wavelength = grating.npa.array(lam)
        real, imag = Eigs(grating, I=I,m=m,c1=c, grad_method="grad", return_vec=False, normalise=normalise)
        eigvals[:,idx] = real + 1j*imag
        
    grating.wavelength = init_wavelength # restore user-initialised wavelength


    # I'm assuming the dummy subplot creates spacing between the two other subplots
    fig, (ax1, dummy, ax2) = plt.subplots(nrows=1, ncols=3, width_ratios=(1,0.1,1))
    dummy.axis('off')
    p = grating.to_numpy(grating.grating_pitch)
    ax1.set_xlim(np.array(wavelength_range)/p) # normalise to grating pitch
    ax2.set_xlim(np.array(wavelength_range)/p) # normalise to grating pitch
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")

    colorReal = (0.7, 0, 0)
    colorImag = 'blue'
    for i in range(4):            
        ax1.plot(wavelengths/p,np.real(grating.to_numpy(eigvals[i,:])), marker, markersize=0.5, markerfacecolor=colorReal, fillstyle='full',  color=colorReal)
        ax2.plot(wavelengths/p,np.imag(grating.to_numpy(eigvals[i,:])), marker, markersize=0.5, markerfacecolor=colorImag, fillstyle='full',  color=colorImag)
        

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

    # fig.suptitle(t=rf"$h_1' = {grating.grating_depth/grating.wavelength:.3f}\lambda_0$, $\Lambda' = {grating.grating_pitch/grating.wavelength:.3f}\lambda_0$")

    # Modify axes
    cm_to_inch = 0.393701
    fig_width = 30*cm_to_inch
    fig_height = 17.6*cm_to_inch
    fig.set_size_inches(fig_width/1.2, fig_height/1.2)

    return fig, (ax1, ax2)

def show_FOM_spectrum(grating: TwoBox, angle: float=0., wavelength_range: list=[1., 1.5], num_plot_points: int=200, I: float=10e9, grad_method: str="grad"):
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
    init_wavelength = grating.wavelength  # record user-initialised wavelength
    inc_angle_deg = angle*180/np.pi
    
    efficiencies = np.zeros(num_plot_points, dtype=float)
    for idx, lam in enumerate(wavelengths):
        grating.wavelength = lam
        efficiencies[idx] = FoM(grating, I=I, grad_method="grad")
    grating.wavelength = init_wavelength

    fig, ax = plt.subplots(1)         
    p = grating.to_numpy(grating.grating_pitch)
    ax.set_xlim(wavelength_range/p)  # normalise wavelength to grating pitch
    ax.plot(wavelengths/p, efficiencies, color=(0.7, 0, 0), linestyle='-', lw=LINE_WIDTH)
    ax.set(title=rf"{grating.title} $h_1' = {grating.grating_depth/grating.wavelength:.3f}\lambda_0$, $\Lambda' = {grating.grating_pitch/grating.wavelength:.3f}\lambda_0$", xlabel=r"$\lambda'/\Lambda'$", ylabel="FoM")
    ax.axhline(y=0, color='black', linestyle='-', lw = '1')
    ax.tick_params(axis='both', which='both', direction='in') # ticks inside box
    
    cm_to_inch = 0.393701
    fig_width = 20.85*cm_to_inch
    fig_height = 17.6*cm_to_inch
    fig.set_size_inches(fig_width/1.2, fig_height/1.2)
    
    return fig, ax