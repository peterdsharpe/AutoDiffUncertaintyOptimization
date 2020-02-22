import numpy as np
import pytest
import scipy.stats
from pytest import approx
from statsmodels.stats.moment_helpers import mnc2cum

import nig

class TestParametersFromCumulants():
    """unit tests for parameters_from_cumulants."""
    def test_mu0_delta1(self):
        ### Setup ###
        alpha_true = 2.
        beta_true = 1.
        delta_true = 1.
        mu_true = 0.
        dist = scipy.stats.norminvgauss(
            a=alpha_true, b=beta_true, loc=mu_true, scale=delta_true)
        # Compute the moments
        moments = [dist.moment(i) for i in (1, 2, 3, 4)]
        # Compute the cumulants
        cumulants = mnc2cum(moments)

        ### Action ###
        alpha_bar, beta_bar, mu, delta = nig.parameters_from_cumulants(cumulants)

        ### Verification ###
        assert alpha_bar == approx(alpha_true)
        assert beta_bar == approx(beta_true)
        assert mu == approx(mu_true)
        assert delta == approx(delta_true)

    def test_mu0_delta2(self):
        ### Setup ###
        alpha_true = 2.
        beta_true = 1.
        delta_true = 2.
        mu_true = 0.
        dist = scipy.stats.norminvgauss(
            a=alpha_true, b=beta_true, loc=mu_true, scale=delta_true)
        # Compute the moments
        moments = [dist.moment(i) for i in (1, 2, 3, 4)]
        # Compute the cumulants
        cumulants = mnc2cum(moments)

        ### Action ###
        alpha_bar, beta_bar, mu, delta = nig.parameters_from_cumulants(cumulants)

        ### Verification ###
        assert alpha_bar == approx(alpha_true)
        assert beta_bar == approx(beta_true)
        assert mu == approx(mu_true)
        assert delta == approx(delta_true)

    def test_mu1_delta2(self):
        ### Setup ###
        alpha_true = 2.
        beta_true = 1.
        delta_true = 2.
        mu_true = 1.
        dist = scipy.stats.norminvgauss(
            a=alpha_true, b=beta_true, loc=mu_true, scale=delta_true)
        # Compute the moments
        moments = [dist.moment(i) for i in (1, 2, 3, 4)]
        # Compute the cumulants
        cumulants = mnc2cum(moments)

        ### Action ###
        alpha_bar, beta_bar, mu, delta = nig.parameters_from_cumulants(cumulants)

        ### Verification ###
        assert alpha_bar == approx(alpha_true)
        assert beta_bar == approx(beta_true)
        assert mu == approx(mu_true)
        assert delta == approx(delta_true)

    def test_beta_neg(self):
        ### Setup ###
        alpha_true = 2.
        beta_true = -1.
        delta_true = 1.
        mu_true = 0.
        dist = scipy.stats.norminvgauss(
            a=alpha_true, b=beta_true, loc=mu_true, scale=delta_true)
        # Compute the moments
        moments = [dist.moment(i) for i in (1, 2, 3, 4)]
        # Compute the cumulants
        cumulants = mnc2cum(moments)

        ### Action ###
        alpha_bar, beta_bar, mu, delta = nig.parameters_from_cumulants(cumulants)

        ### Verification ###
        assert alpha_bar == approx(alpha_true)
        assert beta_bar == approx(beta_true)
        assert mu == approx(mu_true)
        assert delta == approx(delta_true)
