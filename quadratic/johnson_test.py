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

    def test_asym_xi_0(self):
        """Asymmetric distribution, xi = 0."""
        xi_true = 0.
        lamb_true = 1.
        gamma_true = 1.
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
        assert xi == approx(xi_true, abs=1e-6)
        assert lamb == approx(lamb_true)
        assert gamma == approx(gamma_true)
        assert delta == approx(delta_true)

    def test_asym_xi_2(self):
        """Asymmetric distribution, xi = 2."""
        xi_true = 2.
        lamb_true = 1.
        gamma_true = 1.
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
        assert xi == approx(xi_true, abs=1e-6)
        assert lamb == approx(lamb_true)
        assert gamma == approx(gamma_true)
        assert delta == approx(delta_true)

class TestFindGammaDelta():
    """unit tests for find_gamma_delta."""

    def test_eld69_example(self):
        ### Setup ###
        beta_1 = 0.0053656
        beta_2 = 3.172912

        ### Action ###
        gamma, delta = johnson.find_gamma_delta(
            1, beta_1, beta_2,
            max_iters=3, w_guess=1.0408108)

        ### Verification ###
        gamma_correct = 0.604277
        delta_correct = 5.061155
        assert gamma == approx(gamma_correct, rel=5e-3)
        assert delta == approx(delta_correct, rel=5e-3)

    def test_eld69_example_no_guess(self):
        ### Setup ###
        beta_1 = 0.0053656
        beta_2 = 3.172912

        ### Action ###
        gamma, delta = johnson.find_gamma_delta(
            1, beta_1, beta_2)

        ### Verification ###
        gamma_correct = 0.604277
        delta_correct = 5.061155
        assert gamma == approx(gamma_correct, rel=5e-3)
        assert delta == approx(delta_correct, rel=5e-3)
