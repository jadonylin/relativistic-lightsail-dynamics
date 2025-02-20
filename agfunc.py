import autograd.numpy as npa
from autograd import grad, jacobian
from torch.autograd import grad as grad_torch
from torch.autograd import jacobian as jacobian_torch
from torch import erf as torch_erf
from torch import linalg as torchLA
from autograd.scipy.special import erf as autograd_erf
from autograd.numpy import linalg as npaLA
import torch

class agfunc:
    """ wrapper class for autograd and torch functions """
    def __init__(self,lib):
        self.lib = lib
        if lib=="autograd":
            self.sqrt= npa.sqrt
            self.erf = autograd_erf
            self.norm = npaLA.norm
            self.jacobian = jacobian
            self.array = npa.array
            self.sin = npa.sin
            self.cos = npa.cos
            self.arcsin = npa.arcsin
            self.exp = npa.exp
            self.sum = npa.sum
            self.abs = npa.abs
            self.softmax = softmax
            self.linspace=npa.linspace
            self.minimum=npa.minimum
            self.sort=npa.sort
            self.diff=npa.diff
            self.append=npa.append
            self.real=npa.real
            self.imag=npa.imag
            self.power=npa.power
            self.log=npa.log
            self.eig=npaLA.eig
            self.det=npaLA.det

        elif lib=="torch":
            self.sqrt = torch.sqrt
            self.erf = torch_erf
            self.norm=torchLA.norm
            self.jacobian = jacobian_torch
            self.array = torch.tensor
            self.sin=torch.sin
            self.cos=torch.cos
            self.arcsin=torch.asin
            self.exp=torch.exp
            self.sum=torch.sum
            self.abs=torch.abs
            self.softmax = softmax_torch
            self.linspace=torch.linspace
            self.minimum=torch.minimum
            self.sort=lambda x: torch.sort(x)[0]
            self.diff=torch.diff
            self.append=lambda x,y: torch.cat((x,y),dim=0)
            self.real=torch.real
            self.imag=torch.imag
            self.power=torch.pow
            self.log=torch.log
            self.eig=lambda x: torch.eig(x,eigenvectors=True)
            self.det=torch.det