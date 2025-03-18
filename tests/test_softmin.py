"""
Test softmin function
"""

import numpy as np

import pytest

import sys
sys.path.append("../")

from twobox import softmin


abs_tol = 1e-4
abs_tol_precise = 1e-9
rel_tol = 0.1
sigmas = [1e6, 100., 10., 1.]  # softmin smoothing factors (larger means softmin is closer to np.min)
def expected_value(x, prob):
    return np.sum(x*prob)

@pytest.fixture(params=sigmas)
def _sigmas(request):
    return request.param

class TestClass:
    def test_softmin_normal(self, _sigmas):
        """
        "Normal" values (order of magnitude between 0.01 and 100)
        """
        values = np.array([0.1, 100, -0.1, -100])
        prob = softmin(values, _sigmas)
        output = expected_value(values,prob)
        truth = np.min(values)
        assert np.allclose(output, truth, atol=abs_tol)
    
    def test_softmin_some_equal(self, _sigmas):
        """
        Some values are equal
        """
        values = np.array([0.1, -100, -0.1, -100])
        prob = softmin(values, _sigmas)
        output = expected_value(values,prob)
        truth = np.min(values)
        assert np.allclose(output, truth, atol=abs_tol)
    
    def test_softmin_all_equal(self, _sigmas):
        """
        All values are equal
        """
        values = np.array([-100, -100, -100, -100])
        prob = softmin(values, _sigmas)
        output = expected_value(values,prob)
        truth = np.min(values)
        assert np.allclose(output, truth, atol=abs_tol)
    
    def test_softmin_large_values(self, _sigmas):
        """
        Values have large magnitude
        """
        values = np.array([1e4, 1e5, 1e6, 1e7])
        prob = softmin(values, _sigmas)
        output = expected_value(values,prob)
        truth = np.min(values)
        assert np.allclose(output, truth, atol=abs_tol)
    
    def test_softmin_mega_values(self, _sigmas):
        """
        Values have really large magnitude
        """
        values = np.array([1e9, 1e10, 1e11, 1e12])
        prob = softmin(values, _sigmas)
        output = expected_value(values,prob)
        truth = np.min(values)
        assert np.allclose(output, truth, atol=abs_tol)
    
    def test_softmin_small_values(self, _sigmas):
        """
        Values have near-zero magnitude

        FAILS
        """
        values = np.array([1e-4, 1e-5, 1e-6, 1e-7])
        prob = softmin(values, _sigmas)
        output = expected_value(values,prob)
        truth = np.min(values)
        assert np.allclose(output, truth, atol=abs_tol_precise)