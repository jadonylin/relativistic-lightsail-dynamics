"""
A module to store helper functions that calculate Qpr and 
adjacent reflection quantities. Stored here to avoid cluttering twobox class.
"""

import numpy as np
import torch

class QprBox:
    """
    Radiation-pressure effiency factor calculations for TwoBox gratings.
    """
    def rNeg1(self, angle):
        """
        Calculates r_{-1} reflection efficiency at given excitation-plane-wave angle
        """
        self.angle = angle
        r = self.eff()[0][0]
        return r
    
    def tNeg1(self, angle):
        """
        Calculates t_{-1} reflection efficiency at given excitation-plane-wave angle
        """
        self.angle = angle
        t = self.eff()[1][0]
        return t

    def r0(self, angle):
        """
        Calculates r_{0} reflection efficiency at given excitation-plane-wave angle
        """
        self.angle = angle
        r = self.eff()[0][1]
        return r
    
    def t0(self, angle):
        """
        Calculates t_{0} reflection efficiency at given excitation-plane-wave angle
        """
        self.angle = angle
        t = self.eff()[1][1]
        return t

    def r1(self, angle):
        """
        Calculates r_{1} reflection efficiency at given excitation-plane-wave angle
        """
        self.angle = angle
        r = self.eff()[0][2]
        return r
    
    def t1(self, angle):
        """
        Calculates t_{1} reflection efficiency at given excitation-plane-wave angle
        """
        self.angle = angle
        t = self.eff()[1][2]
        return t

    def PDrNeg1(self, angle: float=0.):
        """
        Calculates gradient of r_{-1} reflection efficiency with respect to excitation-plane-wave angle 
        at a given excitation-plane-wave angle
        
        NOTE: I think it is necessary to set and restore self.angle for the angle derivatives to be gradable
        """
        input_angle = self.angle  
        _PDrNeg1 = self.npa.grad(self.rNeg1)(self.npa.array(angle)) 
        self.angle = input_angle  
        return _PDrNeg1
    
    def PDtNeg1(self, angle: float=0.):
        """
        Calculates gradient of t_{-1} reflection efficiency with respect to excitation-plane-wave angle 
        at a given excitation-plane-wave angle
        """
        input_angle = self.angle
        _PDtNeg1 = self.npa.grad(self.tNeg1)(self.npa.array(angle)) # PD_both_Q(self, params)
        self.angle = input_angle  
        return _PDtNeg1
    
    def PDr0(self, angle: float=0.):
        """
        Calculates gradient of r_{0} reflection efficiency with respect to excitation-plane-wave angle 
        at a given excitation-plane-wave angle
        """
        input_angle = self.angle
        _PDr0 = self.npa.grad(self.r0)(self.npa.array(angle)) # PD_both_Q(self, params)
        self.angle = input_angle  
        return _PDr0
    
    def PDt0(self, angle: float=0.):
        """
        Calculates gradient of t_{0} reflection efficiency with respect to excitation-plane-wave angle 
        at a given excitation-plane-wave angle
        """
        input_angle = self.angle
        _PDt0 = self.npa.grad(self.t0)(self.npa.array(angle)) # PD_both_Q(self, params)
        self.angle = input_angle  
        return _PDt0

    def PDr1(self, angle: float=0.):
        """
        Calculates gradient of r_{1} reflection efficiency with respect to excitation-plane-wave angle 
        at a given excitation-plane-wave angle
        """
        input_angle = self.angle
        _PDr1 = self.npa.grad(self.r1)(self.npa.array(angle)) # PD_both_Q(self, params)
        self.angle = input_angle  
        return _PDr1
    
    def PDt1(self, angle: float=0.):
        """
        Calculates gradient of t_{1} reflection efficiency with respect to excitation-plane-wave angle 
        at a given excitation-plane-wave angle
        """
        input_angle = self.angle
        _PDt1 = self.npa.grad(self.t1)(self.npa.array(angle)) # PD_both_Q(self, params)
        self.angle = input_angle  
        return _PDt1

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
        
        Q1 = self.npa.array(0.0)
        Q2 = self.npa.array(0.0)
        M = [-1,0,1]
        
        # M = self.grating_orders()  # this works in pytorch, but not in autograd
        # begin debugging torch pytorch jacobian returning 0:
        # M = [0]
        # end debug
        if self.RCWA_engine == 'TORCWA':
            M = self.grating_orders() # this works in pytorch, but not in autograd, which throws an error that int isn't differentiable
            # however, doing it this way doesn't help with NaN being by  returned for derivatives when only m=0 orders are propagative in torcwa
        for ord in M:
            m = 1+ord # convert grating order to index of array, assumes -1,0,1
            delta_m = self.diffraction_angle(ord)
            # # if isinstance(delta_m,str):
            if self.npa.isnan(delta_m):
                """
                If no diffraction order, Q_{pr,j}' is unchanged
                """
                Q1 = Q1 + 0
                Q2 = Q2 + 0
            else:
            # take back to real as Q are real, complex intermediary only for gradient tracking compatibility
                Q1 = Q1+ r[m]*(1 + self.npa.cos(self.angle+delta_m)) + t[m]*(1 - self.npa.cos(delta_m-self.angle))
                Q2 = Q2+ r[m]*self.npa.sin(self.angle+delta_m) + t[m]*self.npa.sin(delta_m-self.angle)
        Q1 =  self.npa.cos(self.angle)*Q1
        Q2 = -self.npa.cos(self.angle)*Q2
        if self.RCWA_engine == 'TORCWA':
            return torch.stack((Q1,Q2))
            # return self.npa.array( [Q1, Q2] )  
        else:
            return self.npa.array( [Q1, Q2] )


    def return_Qs(self, h_angle, h_wavelength):
        """
        Calculate efficiency factors and their derivatives
        NOTE: unchanged from pre-torcwa as it doesn't use autograd or grcwa
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
        params = self.npa.array([input_angle, input_wavelength])
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
    