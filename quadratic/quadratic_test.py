import numpy as np
import pytest
from pytest import approx
from statsmodels.stats.moment_helpers import mc2mnc, cum2mc
from math import factorial

import quadratic

class TestMomentsFromCumulants():
    """Unit tests for moments_from_cumulants."""
    def test_standard_normal(self):
        """Test on the standard normal distribution."""
        ### Setup ###
        # Cumulants of the standard normal distribution
        k = np.array([0., 1., 0., 0.])
        # Moments (about 0) of the standard normal distribution
        correct_moments = np.array([1., 0., 1., 0., 3.])

        ### Action ###
        moments = quadratic.moments_from_cumulants(k)

        ### Verification ###
        assert len(moments) == len(correct_moments)
        np.testing.assert_array_equal(moments, correct_moments)

    def test_standard_gamma(self):
        """Test on a gamma distribution with alpha = beta = 1."""
        ### Setup ###
        # Cumulants of the gamma distribution with alpha = beta = 1
        k = np.array([1., 1., 2., 6., 24.])
        # Moments (about 0) of the gamma distribution with alpha = beta = 1
        correct_moments = np.array([1., 1., 2., 6., 24., 120.])

        ### Action ###
        moments = quadratic.moments_from_cumulants(k)

        ### Verification ###
        assert len(moments) == len(correct_moments)
        np.testing.assert_array_equal(moments, correct_moments)

    def test_against_statsmodels(self):
        """Test against statsmodels.stats.moment_helpers."""
        ### Setup ###
        k = np.array([3, 4, 6, 2])
        sm_moments = np.insert(mc2mnc(cum2mc(k)), 0, 1)

        ### Action ###
        moments = quadratic.moments_from_cumulants(k)

        ### Verification ###
        assert len(moments) == len(sm_moments)
        np.testing.assert_array_equal(moments, sm_moments)


class TestCumulantQuadForm():
    """Unit tests for cumulant_quad_form."""
    def test_chi_squared(self):
        """If A = Sigma = I and mu = 0, the quadratic form Q
        is distributed as a chi-squared random variable, which has known
        cumulants."""
        n_cumulants = 8
        for dof in range(1, 9):
            ### Setup ###
            A = np.eye(dof)
            covar = np.eye(dof)
            mean = np.zeros(dof)
            correct_cumulants = np.zeros(n_cumulants)
            cumulants = np.zeros(n_cumulants)

            ### Action ###
            for s in range(1, n_cumulants + 1):
                correct_cumulants[s - 1] = 2**(s - 1) * factorial(s - 1) * dof
                cumulants[s - 1] = quadratic.cumulant_quad_form(
                    s, mean, covar, A)

            ### Verification ###
            np.testing.assert_array_equal(cumulants, correct_cumulants)

class TestGammaMoment():
    """Unit tests for gamma_moment."""

    def test_1_1(self):
        ### Setup ###
        # Moments (about 0) of the gamma distribution with alpha = beta = 1
        correct_moments = np.array([1., 1., 2., 6., 24., 120.])
        i = np.array(range(len(correct_moments)))

        ### Action ###
        moments = quadratic.gamma_moment(i, alpha=1., beta=1.)

        ### Verification ###
        np.testing.assert_array_equal(moments, correct_moments)
