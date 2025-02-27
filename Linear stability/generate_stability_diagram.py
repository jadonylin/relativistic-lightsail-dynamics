"""
Generate data for 2D-stability-diagram contour plots to show the linear stability of the twobox grating.
The data is saved in a pickle file for extraction in plot_stability_diagram.ipynb.
"""

import numpy as np
from numpy import linalg as LA

import pickle

import sys
sys.path.append("../")

import time

from Dynamics.specrel import Parameters, erf
from twobox import TwoBox
from parameters import D1_ND


_, L, m, c = Parameters()
I = 10e9


grating_type = "Ilic"
final_speed_percent  = 15  # final speed as a percentage of c
final_speed = final_speed_percent/100

k_width = 1000  # Number of widths
width_start = 0.01  # widths normalised by grating length L
width_end = 5  

k_lambda = 1000  # Number of wavelengths
v_start = 0
v_end = final_speed


width_array = np.linspace(width_start, width_end, k_width)
v_array = np.linspace(v_start, v_end, k_lambda)
lambda_array = np.zeros(k_lambda)
for i in range(k_lambda):
    lambda_array[i] = 1/D1_ND(v_array[i])  # Assumes starting wavelength is 1


# Jacobian terms/stiffness coefficients
# The "k" terms are the restoring coefficients (derivatives of forces with respect to displacement), 
# the "mu" terms are the damping coefficients (derivatives of forces with respect to velocities). 
# The subscript y or phi indicates a derivative with respect to either a y or phi coordinate, or a 
# y or phi velocity. For the k terms, the subscript y or phi denotes derivative with respect to 
# coordinate y or phi. For the mu terms, the subscript y or phi denotes derivative with respect to 
# velocity vy or angular velocity vphi.
ky_y_array      = np.zeros((k_lambda, k_width))
ky_phi_array    = np.zeros((k_lambda, k_width))
muy_y_array     = np.zeros((k_lambda, k_width))
muy_phi_array   = np.zeros((k_lambda, k_width))
kphi_y_array    = np.zeros((k_lambda, k_width))
kphi_phi_array  = np.zeros((k_lambda, k_width))
muphi_y_array   = np.zeros((k_lambda, k_width))
muphi_phi_array = np.zeros((k_lambda, k_width))

# Eigenvalues
real_regions = np.zeros((k_lambda, k_width))
imag_regions = np.zeros((k_lambda, k_width))
neg_real     = np.zeros((k_lambda, k_width))
zero_real    = np.zeros((k_lambda, k_width))
neg_imag     = np.zeros((k_lambda, k_width))
zero_imag    = np.zeros((k_lambda, k_width))
eigvals      = np.zeros((k_lambda, k_width, 4), dtype=np.complex128)  # 3rd axis is the eigenvalue float

# Eigenvectors
# 3rd and 4th axis of eigvecs is the eigenvector 2D array with ordered columns corresponding to 
# the eigenvalues stored in the 3rd axis of eigvals.
eigvecs = np.zeros((k_lambda, k_width, 4, 4), dtype=np.complex128)  

# Linear stability conditions ordered along the 3rd axis as follows:
# 0: Sum of damping is negative
# 1: First mixed condition
# 2: Second mixed condition
# 3: Pure restoring terms are stronger than mixed restoring terms
conditions = np.zeros((k_lambda, k_width, 4))  


if grating_type=="Ilic":
    wavelength      = 1.5 
    grating_pitch   = 1.8/wavelength
    grating_depth   = 0.5/wavelength
    box1_width      = 0.15*grating_pitch
    box2_width      = 0.35*grating_pitch
    box_centre_dist = 0.60*grating_pitch
    box1_eps        = 3.5**2 
    box2_eps        = 3.5**2
    gaussian_width  = 2*L
    substrate_depth = 0.5/wavelength
    substrate_eps   = 1.45**2
if grating_type=="Optimised":
    ## Optimised - second
    grating_pitch   = 1.5384469388251338
    grating_depth   = 0.5580762361523982
    box1_width      = 0.10227122552871484
    box2_width      = 0.07605954942866577
    box_centre_dist = 0.2669020979549422
    box1_eps        = 9.614975107945112
    box2_eps        = 9.382304398409568
    gaussian_width  = 33.916288616522735
    substrate_depth = 0.17299998450776535
    substrate_eps   = 9.423032644325023

wavelength = 1.
angle = 0.
Nx = 100
numG = 25
Qabs = np.inf

grating = TwoBox(grating_pitch, grating_depth, box1_width, box2_width, box_centre_dist, box1_eps, box2_eps, 
                 gaussian_width, substrate_depth, substrate_eps,
                 wavelength, angle, Nx, numG, Qabs)


timeSTART = time.time()
for i in range(k_lambda):
    """
    At a fixed wavelength, calculate width-independent terms
    """
    
    lam = lambda_array[i]

    # Retrieve efficiency factors
    grating.wavelength = lam
    Q1R, Q2R, dQ1ddeltaR, dQ2ddeltaR, dQ1dlambdaR, dQ2dlambdaR = grating.return_Qs_auto(return_Q=True)

    # Convert velocity dependence to wavelength dependence
    D = 1/lam 
    g = (np.power(lam,2) + 1)/(2*lam) 

    # Apply symmetry conditions to obtain left-grating efficiencies
    Q1L = Q1R;   Q2L = -Q2R;   
    dQ1ddeltaL  = -dQ1ddeltaR;    dQ2ddeltaL  = dQ2ddeltaR
    dQ1dlambdaL = dQ1dlambdaR;    dQ2dlambdaL = -dQ2dlambdaR

    # Calculate width-independent terms for minor time save
    ky_y      = -D**2 * I/(m*c) * (Q2R - Q2L) 
    ky_phi    = -D**2 * I/(m*c) * (dQ2ddeltaR + dQ2ddeltaL)
    muy_y     = -D**2 * I/(m*c) * 1/c * (D+1)/(D*(g+1)) * (Q1R + Q1L + dQ2ddeltaR + dQ2ddeltaL) 
    muy_phi   =  D**2 * I/(m*c) * 1/c * (2*(Q2R - Q2L) - lam*(dQ2dlambdaR - dQ2dlambdaL))

    kphi_y    =  D**2 * 12*I/(m*c*L**2) * (Q1R + Q1L)
    kphi_phi  =  D**2 * 12*I/(m*c*L**2) * (dQ1ddeltaR - dQ1ddeltaL - (Q2R - Q2L))
    muphi_y   =  D**2 * 12*I/(m*c*L**2) * 1/c * (D+1)/(D*(g+1)) * (dQ1ddeltaR - dQ1ddeltaL - (Q2R - Q2L))
    muphi_phi = -D**2 * 12*I/(m*c*L**2) * 1/c * (2*(Q1R + Q1L) - lam*(dQ1dlambdaR + dQ1dlambdaL))


    for j in range(k_width):
        w_bar = width_array[j]
        w = w_bar * L

        ky_y_w      = ky_y      * (1 - np.exp(-1/(2*w_bar**2)))
        ky_phi_w    = ky_phi    * w/2 * np.sqrt(np.pi/2) * erf(1/(w_bar*np.sqrt(2)))
        muy_y_w     = muy_y     * w/2 * np.sqrt(np.pi/2) * erf(1/(w_bar*np.sqrt(2)))
        muy_phi_w   = muy_phi   * (w/2)**2 * (1 - np.exp(-1/(2*w_bar**2)))

        kphi_y_w    = kphi_y    * (w/2*np.sqrt(np.pi/2) * erf(1/(w_bar*np.sqrt(2))) - L/2*np.exp(-1/(2*w_bar**2)))  
        kphi_phi_w  = kphi_phi  * (w/2)**2 * (1 - np.exp(-1/(2*w_bar**2)))
        muphi_y_w   = muphi_y   * (w/2)**2 * (1 - np.exp(-1/(2*w_bar**2)))
        muphi_phi_w = muphi_phi * (w/2)**2 * (w/2*np.sqrt(np.pi/2) * erf(1/(w_bar*np.sqrt(2))) - L/2*np.exp(-1/(2*w_bar**2))) 

        ky_y_array[i,j]      = ky_y_w
        ky_phi_array[i,j]    = ky_phi_w
        muy_y_array[i,j]     = muy_y_w
        muy_phi_array[i,j]   = muy_phi_w
        kphi_y_array[i,j]    = kphi_y_w
        kphi_phi_array[i,j]  = kphi_phi_w
        muphi_y_array[i,j]   = muphi_y_w
        muphi_phi_array[i,j] = muphi_phi_w

        J00 = ky_y_w;   J01 = ky_phi_w;   J02 = muy_y_w;   J03 = muy_phi_w
        J10 = kphi_y_w; J11 = kphi_phi_w; J12 = muphi_y_w; J13 = muphi_phi_w
        J = np.array([[0,0,1,0],[0,0,0,1],[J00,J01,J02,J03],[J10,J11,J12,J13]])
        

        EIGVALVEC = LA.eig(J)
        eig = EIGVALVEC[0]
        eigvals[i,j] = eig

        eigvec = EIGVALVEC[1]
        eigvecs[i,j] = eigvec

        # Conditions for stability
        a1 = -(J02 + J13)
        a2 = J02*J13 - (J00 + J11 + J03*J12)
        a3 = J00*J13 + J11*J02 - (J01*J12 +  J10*J03)
        a4 = J00*J11 - J01*J10
        
        cond1 = a1
        cond2 = a1*a2 - a3
        cond3 = a3*cond2 -a1**2 * a4
        cond4 = a4

        conditions[i,j] = np.array([cond1,cond2,cond3,cond4])

        # Count negative and zero eigenvalue components
        EIGreal = np.real(eig)
        EIGimag = np.imag(eig)

        neg_real[i,j]  = sum(n<0 for n in EIGreal)
        zero_real[i,j] = sum(n==0 for n in EIGreal)
        neg_imag[i,j]  = sum(n<0 for n in EIGimag)
        zero_imag[i,j] = sum(n==0 for n in EIGimag)

        # Determine the regions of 2D space that have a different number of negative real part eigenvalues
        # and negative imaginary part eigenvalues. This is used to colour the stability diagram.
        # The regions are denoted by (arbitrary) values 0.1 to 0.5 (inclusive), corresponding to the 
        # following values:
        # 0.1 - all negative real part eigenvalues
        # 0.2 - one positive (or zero) real part eigenvalue
        # 0.3 - two positive (or zero) real part eigenvalues, two negative real part eigenvalues
        # 0.4 - three positive (or zero) real part eigenvalues, one negative real part eigenvalue
        # 0.5 - all positive (or zero) real part eigenvalues
        real_regions[i,j] = sum(n>=0 for n in EIGreal)/10 + 0.1
        imag_regions[i,j] = sum(n>=0 for n in EIGimag)/10 + 0.1


timeDIFF = time.time() - timeSTART
t_end = timeDIFF
t_end_sec = round(t_end)
t_end_min = round(t_end/60)
t_end_hours = round(t_end/60**2)
print(rf"duration: {t_end_sec} (sec), {t_end_min} min")
print(rf"#lambda: {k_lambda}, #width: {k_width}")

data1 = {'grating_type': grating_type, 'Nx': Nx, 'numG': numG, 'Qabs': Qabs, 'Intensity': I, 'duration (min)':t_end_min, 
        'start speed': v_start, 'final speed': final_speed_percent, 'start width': width_start, 'final width': width_end,
        'grating length': L,
        'lambda_array': lambda_array, 'v_array': v_array, 'width_array': width_array,
        'neg_real': neg_real, 'zero_real':zero_real,
        'neg_imag': neg_imag, 'zero_imag':zero_imag,
        'real_stability': real_regions, 'imag_stability': imag_regions}
data2 = {'eigvals': eigvals, 'eigvecs': eigvecs,
        'kyy': ky_y_array, 'kyphi': ky_phi_array, 'muyy': muy_y_array, 'muyphi': muy_phi_array,
        'kphiy': kphi_y_array, 'kphiphi': kphi_phi_array, 'muphiy': muphi_y_array, 'muphiphi': muphi_phi_array, 
        'conditions': conditions}

pkl_fname1 = rf'./Data/{grating_type}_Stability_Diagram_klambda{k_lambda}_by_kwidth{k_width}_num_neg_zero.pkl'
pkl_fname2 = rf'./Data/{grating_type}_Stability_Diagram_klambda{k_lambda}_by_kwidth{k_width}_Jacobian_eigen.pkl'
with open(pkl_fname1, 'wb') as data1_file, open(pkl_fname2, 'wb') as data2_file:
    pickle.dump(data1, data1_file)
    pickle.dump(data2, data2_file)