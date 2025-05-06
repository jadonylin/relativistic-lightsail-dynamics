import numpy as np

import autograd.numpy as npa
from autograd import grad, jacobian
from autograd.scipy.special import erf as autograd_erf
from autograd.numpy import linalg as npaLA

import torch
from torch.autograd import grad as grad_torch
# from torch.autograd.functional import jacobian as jacobian_torch_raw
from torch import erf as torch_erf
from torch import linalg as torchLA

import functools

# Try to import ArrayBox from autograd, if available.
try:
    from autograd.numpy.numpy_boxes import ArrayBox
except ImportError:
    ArrayBox = None

# Smoothing if conditionals for backpropagation
def jacobian_torch(f,argnum=0):
    return torch.func.jacrev(f,argnums=argnum)

# this only really works for a function of a single variable...
# def grad_torch(f,argnum=0):
#     return lambda x: grad_torch_value(f,x,argnum)
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

# def grad_torch_value(f,x,argnum=0):
#      a=f(x)     
#      return torch.autograd.grad(a,x,create_graph=True, materialize_grads=True) # materialize_grads=True to avoid returning None when functino does not depend on x  - added during debugging of PDtNeg1 but may not be needed? 

# attempt with torch.func :
def grad_torch(f,argnum=0):
    return torch.func.jacrev(f,argnums=argnum)

def autograd_jacobian(f,argnum=0):
    # return lambda x: jacobian(f,argnum=argnum)(x).squeeze()
    return jacobian(f,argnum=argnum)

class agfunc:
    """ 
    Wrapper class for autograd and torch functions 
    Optional arguments:
      - device="cpu" or "cuda" 
      - precision="double" (torch.complex128 and torch.float64 are used) or "single" (torch.complex64 and torch.float32 are used)
    """
    def __init__(self, lib, device: str="cpu", precision: str="double") -> None:
        self.lib = lib
        self.device = device
        
        if lib == "autograd":
            self.sqrt = npa.sqrt
            self.erf = autograd_erf
            self.norm = npaLA.norm
            self.jacobian = autograd_jacobian
            self.array = npa.array
            self.sin = npa.sin
            self.cos = npa.cos
            self.arcsin = npa.arcsin
            self.exp = npa.exp
            self.sum = npa.sum
            self.abs = npa.abs
            self.softmax = self._softmax
            self.linspace = npa.linspace
            self.minimum = npa.minimum
            self.min = npa.min
            self.max = npa.max
            self.sort = npa.sort
            self.diff = npa.diff
            self.append = npa.append
            self.real = npa.real
            self.imag = npa.imag
            self.power = npa.power
            self.log = npa.log
            self.eig = npaLA.eig
            self.det = npaLA.det
            self.grad = grad
            self.isnan = npa.isnan
            self.concatenate = npa.concatenate
            self.diff = npa.diff
            self.maximum = npa.maximum
            self.int = lambda x: x.astype(int)
            self.zeros = npa.zeros
            self.dot = npa.dot
            self.stack = npa.stack
        elif lib == "torch":
            self.sqrt = torch.sqrt
            self.erf = torch_erf
            self.norm = torchLA.norm
            self.jacobian = jacobian_torch
            self.array = self.torch_tensor_with_grad
            self.sin = torch.sin
            self.cos = torch.cos
            self.arcsin = torch.asin
            self.exp = torch.exp
            self.sum = torch.sum
            self.abs = torch.abs
            self.softmax = self._softmax_torch
            self.linspace = torch.linspace
            self.minimum = torch.minimum
            self.min = torch.min
            self.max = torch.max
            self.sort = lambda x: torch.sort(x)[0]
            self.diff = torch.diff
            self.append = lambda x,y: torch.cat((x,y),dim=0)
            self.real = torch.real
            self.imag = torch.imag
            self.power = torch.pow
            self.log = torch.log
            self.eig = lambda x: torch.linalg.eig(x)
            self.det = torch.det
            self.grad = grad_torch
            self.isnan = torch.isnan
            self.concatenate = torch.cat
            self.diff = torch.diff
            self.maximum = torch.maximum
            self.int = lambda x: x.long()
            self.zeros = self.zeros_torch
            self.dot = torch.dot
            self.stack = torch.stack
            if precision == "double":
                self.ctype = torch.complex128
                self.ftype = torch.float64
                torch.backends.cuda.matmul.allow_tf32 = False
            elif precision == "single":
                self.ctype = torch.complex64
                self.ftype = torch.float32
            if torch.cuda.is_available() and device=="cuda":
                device = torch.device('cuda')
            else:
                device = torch.device('cpu')

    def zeros_torch(self,*args, **kwargs):
        if 'dtype' in kwargs:
            if not isinstance(kwargs['dtype'], torch.dtype):
                if isinstance(kwargs['dtype'],np.dtype):
                    if np.issubdtype(np.dtype(kwargs['dtype']), np.complexfloating):
                        kwargs['dtype']=self.ctype
                    elif np.issubdtype(np.dtype(kwargs['dtype']), np.floating):
                        kwargs['dtype']=self.ftype
                else:
                    try:
                        dtype = np.dtype(kwargs['dtype'])
                    except TypeError:
                        raise ValueError(f"Unsupported NumPy dtype: {kwargs['dtype']}")
                    if isinstance(dtype,np.dtype):
                        if np.issubdtype(dtype, np.complexfloating):
                            kwargs['dtype']=self.ctype
                        elif np.issubdtype(dtype, np.floating):
                            kwargs['dtype']=self.ftype

        else:
            kwargs['dtype']=self.ftype # by default return array of default float (not complex) type
        return torch.zeros(*args, **kwargs)
            
    def _softmax(self,p,sigma):
        e_x = npa.exp(sigma*(p - npa.max(p)))
        return e_x/npa.sum(e_x)

    def _softmax_torch(self,p,sigma):
        e_x = torch.exp(sigma*(p - torch.max(p)))
        return e_x/torch.sum(e_x)

    def softmin(self,p,sigma):
        """Used to approximate min via expected value of array p with probability distribution given by softmin(p)"""
        e_x = self.exp(sigma*(self.min(p) - p))
        return e_x/self.sum(e_x)

    @functools.wraps(torch.tensor)
    def torch_tensor_with_grad(self,*args, **kwargs):
        # creates tensor from input args with gradient tracking
        # Force requires_grad to be True (overriding any passed value)
        # if args[0] is a torch.Tensor, return it directly.
        # if args[0] is a numpy array, convert it to a torch.Tensor
        # Beware, if args[0] is a list or tuple of torch tensors, the result will not be autogradable
        #      Use stack or cat instead
        kwargs['requires_grad'] = True
        kwargs['device'] = self.device
        if 'dtype' in kwargs:
            if not isinstance(kwargs['dtype'], torch.dtype):
                if isinstance(kwargs['dtype'],np.dtype):
                    if np.issubdtype(np.dtype(kwargs['dtype']), np.complexfloating):
                        kwargs['dtype']=self.ctype
                    elif np.issubdtype(np.dtype(kwargs['dtype']), np.floating):
                        kwargs['dtype']=self.ftype
                else:
                    try:
                        dtype = np.dtype(kwargs['dtype'])
                    except TypeError:
                        raise ValueError(f"Unsupported NumPy dtype: {kwargs['dtype']}")
                    if isinstance(dtype,np.dtype):
                        if np.issubdtype(dtype, np.complexfloating):
                            kwargs['dtype']=self.ctype
                        elif np.issubdtype(dtype, np.floating):
                            kwargs['dtype']=self.ftype
        else: # type not specified
            if self.contains_complex(args):
                kwargs['dtype']=self.ctype
            else:
                kwargs['dtype']=self.ftype
        
        if isinstance(args[0], torch.Tensor):
            return args[0]

        # If anything else is provided, convert it to a torch tensor. Note autograd may be lost !
        else:
            # if isinstance(args[0], (np.ndarray,list,tuple)):            
            return torch.tensor(*args, **kwargs) 
        
      
        
        
        
    
    def contains_complex(self,x):
        """ written by chatgpt
        Check if any element in a list, tuple, NumPy array, PyTorch tensor,
        or autograd ArrayBox is complex.
        """
        # For Python lists or tuples:
        if isinstance(x, (list, tuple)):
            return any(isinstance(item, complex) for item in x)
        
        # For NumPy arrays:
        elif isinstance(x, np.ndarray):
            # If the array is of object type, check each element individually.
            if x.dtype == np.object_:
                return any(isinstance(item, complex) for item in x)
            # Otherwise, use NumPy's built-in check.
            return np.iscomplexobj(x)
        
        # For NumPy scalars:
        elif isinstance(x, np.generic):
            return np.iscomplexobj(x)
        
        # For PyTorch tensors:
        elif torch.is_tensor(x):
            return x.is_complex()
        
        # For autograd ArrayBox (if autograd is installed):
        elif ArrayBox is not None and isinstance(x, ArrayBox):
            # Convert the ArrayBox to a NumPy array and check.
            return np.iscomplexobj(np.asarray(x))
        
        else:
            raise TypeError("Unsupported type for complex check")
    def numpy_to_torch_dtype(numpy_dtype):
        """ (this funciton written by chatgpt - not currently used
        Convert a NumPy dtype to the corresponding PyTorch dtype.

        Parameters:
            numpy_dtype (numpy.dtype or type): A NumPy data type (e.g., np.float32, np.int64, np.complex64).

        Returns:
            A PyTorch dtype corresponding to the provided NumPy dtype.

        Raises:
            ValueError: If the provided numpy_dtype is not supported.
        """
        # Define a mapping from NumPy dtypes to PyTorch dtypes, including complex types.
        mapping = {
            np.dtype('bool'): torch.bool,
            np.dtype('int8'): torch.int8,
            np.dtype('int16'): torch.int16,
            np.dtype('int32'): torch.int32,
            np.dtype('int64'): torch.int64,
            np.dtype('uint8'): torch.uint8,
            np.dtype('float16'): torch.float16,
            np.dtype('float32'): torch.float32,
            np.dtype('float64'): torch.float64,
            np.dtype('complex64'): torch.complex64,
            np.dtype('complex128'): torch.complex128,
        }
        
        # Normalize the input to a numpy.dtype
        np_dtype = np.dtype(numpy_dtype)
        
        if np_dtype in mapping:
            return mapping[np_dtype]
        else:
            raise ValueError(f"Unsupported NumPy dtype: {numpy_dtype}")