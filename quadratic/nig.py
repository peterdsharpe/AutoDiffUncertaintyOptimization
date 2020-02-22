"""Normailized Inverse Gaussian distribution.

References:
    [WikiNIG] https://en.wikipedia.org/wiki/Normal-inverse_Gaussian_distribution
    [Eri04] A. Eriksson, L. Forsberg, and E. Ghysels,
        "Approximating the probability distribution of functions
        of random variables: A new approach," Econometric Society,
        503, Aug. 2004. https://www.semanticscholar.org/paper/Approximating-the-probability-distribution-of-of-A-Eriksson-Forsberg/dbf77d60d87b39e5547b3fa70fde81b4798d7877
"""
import numpy as np


def is_feasible(cumulants):
    """Are these cumulants feasible for the NIG distribution?

    See [Eri04] Lemma 2.1
    """
    assert len(cumulants) >= 4
    k2 = cumulants[1]
    k3 = cumulants[2]
    k4 = cumulants[3]
    return 3 * k4 * k2 / k3**2 > 5


def parameters_from_cumulants(cumulants):
    """
    cumulants[i] is the (i+1)th cumulant

    See [Eri04] equations 2.11-2.14
    """
    if not is_feasible(cumulants):
        raise ValueError()
    k1 = cumulants[0]
    k2 = cumulants[1]
    k3 = cumulants[2]
    k4 = cumulants[3]
    rho = 3 * k4 * k2 / k3**2 - 4
    alpha_bar = 3 * (4 / rho + 1) * (1 - 1 / rho)**(-0.5) * k2**2 / k4
    beta_bar = np.sign(k3) / rho**0.5 * alpha_bar
    mu = k1 - np.sign(k3) / rho**0.5 * (
        (12 / rho + 3) * k2**3 / k4)**0.5
    delta = (3 * k2**3 * (4 / rho + 1) * (1 - 1 / rho) / k4)**0.5
    return alpha_bar, beta_bar, mu, delta
