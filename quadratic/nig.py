"""Normal-Inverse Gaussian distribution.

References:
    [WikiNIG] https://en.wikipedia.org/wiki/Normal-inverse_Gaussian_distribution
    [Eri04] A. Eriksson, L. Forsberg, and E. Ghysels,
        "Approximating the probability distribution of functions
        of random variables: A new approach," Econometric Society,
        503, Aug. 2004. https://www.semanticscholar.org/paper/Approximating-the-probability-distribution-of-of-A-Eriksson-Forsberg/dbf77d60d87b39e5547b3fa70fde81b4798d7877
"""
import copy
import numpy as np
import scipy.special as sc
from scipy.stats import rv_continuous, invgauss, norm

def is_feasible(cumulants):
    """Are these cumulants feasible for the NIG distribution?

    See [Eri04] Lemma 2.1
    """
    assert len(cumulants) >= 4
    k2 = cumulants[1]
    k3 = cumulants[2]
    k4 = cumulants[3]
    if k3 == 0:
        return True
    return 3 * k4 * k2 / k3**2 > 5


def make_feasible(cumulants):
    """Increase k4 (~kurtosis?) so that the cumulants are feasible for the NIG distribution."""
    if is_feasible(cumulants):
        return cumulants
    cumulants = copy.deepcopy(cumulants)
    k2 = cumulants[1]
    k3 = cumulants[2]
    k4 = cumulants[3]
    k4_min = 5 / 3 * k3**2 / k2 + 1e-6
    cumulants[3] = k4_min
    return cumulants


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
    if k3 == 0:
        # rho_inv is 1 / rho.
        rho_inv = 0
    else:
        rho = 3 * k4 * k2 / k3**2 - 4
        rho_inv = 1 / rho
    alpha_bar = 3 * (4 * rho_inv + 1) * (1 - rho_inv)**(-0.5) * k2**2 / k4
    beta_bar = np.sign(k3) * rho_inv**0.5 * alpha_bar
    mu = k1 - np.sign(k3) * rho_inv**0.5 * (
        (12 * rho_inv + 3) * k2**3 / k4)**0.5
    delta = (3 * k2**3 * (4 * rho_inv + 1) * (1 - rho_inv) / k4)**0.5
    return alpha_bar, beta_bar, mu, delta


class norminvgauss_gen(rv_continuous):
    _support_mask = rv_continuous._open_support_mask

    def _argcheck(self, a, b):
        return (a > 0) & (np.absolute(b) < a)

    def _pdf(self, x, a, b):
        """This implementation of the NIG pdf is slightly different
        that the implementation in scipy. The scipy implementation has overflow
        errors for a > ~710; this one does not."""
        gamma = np.sqrt(a**2 - b**2)
        fac1 = a / np.pi
        sq = np.hypot(1, x)  # reduce overflows
        exp_arg = gamma + b * x - a * sq
        return fac1 * sc.k1e(a * sq) * np.exp(exp_arg) / sq

    def _rvs(self, a, b):
        # note: X = b * V + sqrt(V) * X is norminvgaus(a,b) if X is standard
        # normal and V is invgauss(mu=1/sqrt(a**2 - b**2))
        gamma = np.sqrt(a**2 - b**2)
        sz, rndm = self._size, self._random_state
        ig = invgauss.rvs(mu=1/gamma, size=sz, random_state=rndm)
        return b * ig + np.sqrt(ig) * norm.rvs(size=sz, random_state=rndm)

    def _stats(self, a, b):
        gamma = np.sqrt(a**2 - b**2)
        mean = b / gamma
        variance = a**2 / gamma**3
        skewness = 3.0 * b / (a * np.sqrt(gamma))
        kurtosis = 3.0 * (1 + 4 * b**2 / a**2) / gamma
        return mean, variance, skewness, kurtosis

norminvgauss = norminvgauss_gen(name="norminvgauss")
