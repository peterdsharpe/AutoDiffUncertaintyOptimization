import numpy as np
import scipy.stats
from statsmodels.stats.moment_helpers import mnc2mc
import pytest
from pytest import approx

import johnson

class TestJohnsonSuParams():
    """Unit tests for johnson_su_params_from_moments."""

    def test_sym_xi_0(self):
        """Symmetric distribution, xi = 0."""
        xi_true = 0.
        lamb_true = 1.
        gamma_true = 0.
        delta_true = 2.
        dist = scipy.stats.johnsonsu(
            loc=xi_true, scale=lamb_true,
            a=gamma_true, b=delta_true)

        moments = np.zeros(5)
        central_moments = np.zeros(5)
        for i in range(5):
            moments[i] = dist.moment(i)
        central_moments[1:] = mnc2mc(moments[1:])
        central_moments[0] = 1.

        xi, lamb, gamma, delta = johnson.johnson_su_params_from_moments(
            central_moments)
        assert xi == approx(xi_true)
        assert lamb == approx(lamb_true)
        assert gamma == approx(gamma_true)
        assert delta == approx(delta_true)

    def test_sym_xi_2(self):
        """Symmetric distribution, xi = 2."""
        xi_true = 2.
        lamb_true = 1.
        gamma_true = 0.
        delta_true = 2.
        dist = scipy.stats.johnsonsu(
            loc=xi_true, scale=lamb_true,
            a=gamma_true, b=delta_true)

        moments = np.zeros(5)
        central_moments = np.zeros(5)
        for i in range(5):
            moments[i] = dist.moment(i)
        central_moments[1:] = mnc2mc(moments[1:])
        central_moments[0] = 1.

        xi, lamb, gamma, delta = johnson.johnson_su_params_from_moments(
            central_moments)
        assert xi == approx(xi_true)
        assert lamb == approx(lamb_true)
        assert gamma == approx(gamma_true)
        assert delta == approx(delta_true)
