"""
A module containing functions for special relativity calculations. Several standard special relativity
functions are included, such as the Lorentz transformation, Doppler shift and relativistic velocity addition. 
The module also includes lesser-known relativistic aberration and Wigner rotation functions.


This module also implements functions for convenience factors that appear in the equations of motion
of symmetric lightsails, labelled A, B, C, S and E.

TODO: update documentation
"""
import agfunc
# import torch
# from autograd import numpy as npautograd
# from autograd.scipy.special import erf as autograd_erf
import scipy


#### All velocities are in m/s (NOT normalised by c)
c = scipy.constants.c
class specrel:
    def __init__(self,lib='autograd',device:str = "cpu"):
        if lib == 'autograd':
            self.npa = agfunc.agfunc("autograd")
        elif lib=="torch":
            self.npa = agfunc.agfunc("torch",device=device)
    def set_num_lib(self,lib,device:str = "cpu"):
        if lib == 'autograd':
            self.npa = agfunc.agfunc("autograd")
        elif lib=="torch":
            self.npa = agfunc.agfunc("torch",device=device)
    def identity(self,x):
        """ returns x, for debugging of autograd"""
        return x
    
    def square(self,x):
        """ returns x, for debugging of autograd"""
        return x*x
    
    def norm_squared(self,v):
        """
        ## Inputs
        v: [vx, vy] 2D array
        ## Output
        """
        return self.npa.norm(v)

    def Gamma(self,v):
        """
        ## Inputs
        v: [vx, vy] 2D array
        ## Output
        Lorentz gamma factor gamma(v)
        """
        return (1 - self.norm_squared(v)/c**2)**(-1/2)

    def Dv(self,v):
        """
        ## Inputs
        v: [vx, vy] 2D array
        ## Output
        Doppler factor gamma(v)(1- vx/c)
        """
        return self.Gamma(v)*(1 - v[0]/c)

    def vadd(self,v,u):
        """
        ## Inputs
        v: [vx, vy] 2D array
        u: [vx, vy] 2D array
        ## Output
        v + u - SR addition of velocities
        """
        g = self.Gamma(v)
        return (1/(1+self.npa.dot(v/c,u/c))) * ( v + u/g + ( (g/(g+1)) * self.npa.dot(v/c,u/c) * v ) )

    def Lorentz(self,v,z):
        """
        ## Inputs
        v: [vx, vy] 2D array
        ## Output
        returns ([t',x',y'])
        """
        t = z[0]; x = z[1]; y = z[2]
        g = self.Gamma(v)
        f = g**2/(g+1)
        
        t2 = g * ( t - (v[0]*x)/c**2 - (v[1]*y)/c**2 )
        x2 = -(g*v[0]*t) + (1+f*(v[0]/c)**2) * x + f*((v[0]/c)*(v[1]/c)) * y
        y2 = -(g*v[1]*t) + f*((v[0]/c)*(v[1]/c)) * x + (1+f*(v[1]/c)**2) * y

        return self.npa.array([t2,x2,y2])

    def SinCosTheta(self,v):
        """
        ## Inputs
        v: [vx, vy] 2D array
        ## Output
        sin(theta'), cos(theta') - relativistic aberration
        """
        D = self.Dv(v)
        g = self.Gamma(v)
        sin = (1/D)*( -g*v[1]/c + (g**2/(g+1))*(v[0]*v[1])/c**2 )
        cos = (1/D)*( -g*v[0]/c + 1 + (g**2/(g+1))*(v[0]/c)**2 )
        theta = self.npa.arcsin(sin)
        return sin, cos, theta

    def SinCosEpsilon(self,v,u):
        """
        ## Inputs
        v: [vx, vy] 2D array
        u: [ux, uy] 2D array
        ## Output
        sin(eps), cos(eps), eps - rotation angle from M to M+1
        """
        gv = self.Gamma(v)
        gu = self.Gamma(u)
        g = gv * gu * (1 + self.npa.dot(v/c,u/c) )
        cross = ( (u[0]/c) * (v[1]/c) - (u[1]/c) * (v[0]/c) )
        sin = cross * (gv*gu*(1+g+gv+gu)) / ( (1+g)*(1+gv)*(1+gu) )
        cos = (1+g+gv+gu)**2 /( (1+g)*(1+gv)*(1+gu) ) - 1
        eps = self.npa.arcsin(sin)
        return sin, cos, eps

    # def EpsRateTest(v,u,h):

    def ABSC(self,v,phi):
        """
        ## Inputs
        v: [vx, vy] 2D array \n
        phi': grating angle
        ## Output
        A,B,S,C - sin(theta'), cos(theta'), S,C - linear corrections
        """
        sintheta = self.SinCosTheta(v)[0]
        costheta = self.SinCosTheta(v)[1]
        cos = self.npa.cos(phi)
        sin = self.npa.sin(phi)
        g = self.Gamma(v)
        D = self.Dv(v)
        bx = v[0] / c
        by = v[1] / c
        dot = (bx*cos +g*by*sin)
        dot2 = (by*cos +g*bx*sin)
        
        A = sintheta * cos/(g*D) - sin/D - (
            sintheta*dot) + dot2/(D*(g+1))- (
            g*by*dot/D ) + (
            ((g**2*(g+2))/(D*(g+1)**2))*bx*by*dot  )
        
        B = -costheta*dot + (2*bx*cos)/(D*(g+1)) - (
            (g*bx*dot)/D + (g*by**2 * cos)/(D**2 *(g+1)) ) + (
            ( (g**2 *(g+2))/(D*(g+1)**2) ) * bx**2 * dot )
        S = cos * A - sin * B
        C = sin * A + cos * B
        return A, B, S, C

    def E_eps(self,v, phi):
        """
        ## Inputs
        v: [vx, vy] 2D array \n
        phi': grating angle
        ## Output
        $mathcal(E)$ - epsilon linear correction
        """
        g = self.Gamma(v)
        return (g/(g+1)) * ( self.npa.sin(phi)*v[0]/c - self.npa.cos(phi)*v[1]/c )

    def erf(self,x):
        return self.npa.erf(x)