import autograd.numpy as npa
from autograd import grad, jacobian
from torch.autograd import grad as grad_torch
# from torch.autograd.functional import jacobian as jacobian_torch_raw
from torch import erf as torch_erf
from torch import linalg as torchLA
from autograd.scipy.special import erf as autograd_erf
from autograd.numpy import linalg as npaLA
import torch
import functools
# Smoothing if conditionals for backpropagation
def jacobian_torch(f,argnum=0):
    return torch.func.jacrev(f,argnums=argnum)
def grad_torch(f,argnum=0):
    return lambda x: grad_torch_value(f,x,argnum)
    # return torch.func.grad(f,argnums=argnum)
# def grad_torch_value(f,x,argnum=0):
#     ya=f(x)
#     if ya.dim()>0:
#         y=ya[argnum]
#     else:
#         y=ya
#     y.retain_grad()
#     y.backward()
#     return x.grad
def grad_torch_value(f,x,argnum=0):
     a=f(x)
     return torch.autograd.grad(a,x,create_graph=True, materialize_grads=True) # materialize_grads=True to avoid returning None when functino does not depend on x  - added during debugging of PDtNeg1 but may not be needed? 

@functools.wraps(torch.tensor)
def torch_tensor_with_grad(*args, **kwargs):
    # Force requires_grad to be True (overriding any passed value)
    kwargs['requires_grad'] = True
    return torch.tensor(*args, **kwargs)

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
            self.softmax = self._softmax
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
            self.grad=grad

        elif lib=="torch":
            self.sqrt = torch.sqrt
            self.erf = torch_erf
            self.norm=torchLA.norm
            self.jacobian = jacobian_torch
            self.array = torch_tensor_with_grad
            self.sin=torch.sin
            self.cos=torch.cos
            self.arcsin=torch.asin
            self.exp=torch.exp
            self.sum=torch.sum
            self.abs=torch.abs
            self.softmax = self._softmax_torch
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
            self.grad=grad_torch
    def _softmax(self,sigma,p):
        e_x = npa.exp(sigma*(p - npa.max(p)))
        return e_x/npa.sum(e_x)

    def _softmax_torch(self,sigma,p):
        e_x = torch.exp(sigma*(p - torch.max(p)))
        return e_x/torch.sum(e_x)