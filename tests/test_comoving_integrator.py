"""
Test functions in the comoving integrator module.
"""

from copy import deepcopy

import numpy as np

import pytest

import sys
sys.path.append("../")
sys.path.append("../Dynamics")

import Dynamics.cmvint as cmvint
from twobox import TwoBox


# SETTINGS ################################################################################################
np.random.seed(2024)
abs_tol = 0.5
rel_tol = 0.1
Delta = 1e-7



# TESTS ################################################################################################
# pytestmark = pytest.mark.parametrize("grating", [standard_grating, random_grating, thin_pillars_grating, closebox_grating])

class TestClass:
    def test_append_coordinate_arrays(self):
        """
        """
        coordinate_arrays = [[0.,1.], [[1.,2.], [3.,4.]], [4.,5.]]
        items = [2., [5.,6.], 6.]
        output = cmvint.append_coordinate_arrays(coordinate_arrays, items)
        truth = [[0.,1.,2.], [[1.,2.], [3.,4.], [5.,6.]], [4.,5.,6.]]
        assert output == truth