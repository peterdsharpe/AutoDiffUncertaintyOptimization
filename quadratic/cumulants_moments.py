"""Functions for cumulants and moments of various distributions.

References:
    [Moh12] Ali A Mohsenipour, "On the Distribution of Quadratic Expressions
        in Various Types of Random Vectors," PhD thesis, University of Western Ontario, 2012.
"""
from math import factorial
import numpy as np
from numpy.linalg import matrix_power
from scipy.special import gamma as gamma_func
from scipy.special import hyp1f1

def cumulant_quad_form(s: int, mean: np.ndarray,
                       covar: np.ndarray, A: np.ndarray):
    """Get the s-th cumulant of a quadratic form of a normal random variable.

    This function computes the cumulants of Q, where
        Q = x^t A x
    and
        x ~ N(mean, covar)

    Arguments:
        s: Compute the `s`-th cumulant.
        mean: Mean vector of x.
        covar: Covariance matrix of x.
        A: Symmetric matrix in the quadratic form.

    See [Moh12] equation 2.15.
    """
    # Check inputs
    assert s >= 1
    s = int(s)
    assert mean.ndim == 1
    assert covar.ndim == 2
    assert covar.shape[0] == mean.shape[0]
    assert covar.shape[1] == mean.shape[0]
    assert A.ndim == 2
    assert A.shape[0] == mean.shape[0]
    assert A.shape[1] == mean.shape[0]

    ASig = A @ covar

    if s == 1:
        return np.trace(ASig) + mean @ A @ mean
    return (
        2**(s - 1) * factorial(s)
        * (
            np.trace(matrix_power(ASig, s)) / s
            + mean @ matrix_power(ASig, s - 1) @ A @ mean)
        )

def cumulant_quad_expr(s: int, mean: np.ndarray,
                       covar: np.ndarray, A: np.ndarray,
                       a: np.ndarray):
    """Get the s-th cumulant of a quadratic expression of a normal random variable.

    This function computes the cumulants of Q*, where
        Q* = x^T A x +  a^T x
    and
        x ~ N(mean, covar)

    Arguments:
        s: Compute the `s`-th cumulant.
        mean: Mean vector of x.
        covar: Covariance matrix of x.
        A: Symmetric matrix in the quadratic expression.
        a: Vector in the quadratic expression.

    See [Moh12] equation 2.14.
    """
    # Check inputs
    assert s >= 1
    s = int(s)
    assert mean.ndim == 1
    assert covar.ndim == 2
    assert covar.shape[0] == mean.shape[0]
    assert covar.shape[1] == mean.shape[0]
    assert A.ndim == 2
    assert A.shape[0] == mean.shape[0]
    assert A.shape[1] == mean.shape[0]
    assert a.ndim == 1
    assert a.shape[0] == mean.shape[0]

    ASig = A @ covar

    if s == 1:
        return np.trace(ASig) + mean @ A @ mean + a @ mean
    return (
        2**(s - 1) * factorial(s)
        * (
            np.trace(matrix_power(ASig, s)) / s
            + 0.25 * a @ matrix_power(covar @ A, s - 2) @ covar @ a
            + mean @ matrix_power(ASig, s - 1) @ A @ mean
            + a @ matrix_power(covar @ A, s - 1) @ A @ mean)
        )


def moments_from_cumulants(k: np.ndarray):
    """Compute the first h moments from the first h-1 cumulants.

    k[s - 1] is the s-th cumulant (cumulants index from 1).

    See [Moh12] equation 2.17.
    """
    moments = np.zeros(len(k) + 1)
    moments[0] = 1
    for h in range(1, len(moments)):
        h_1_fac = factorial(h - 1)
        for i in range(0, h):
            moments[h] += (
                h_1_fac / (factorial(h - 1 - i) * factorial(i))
                * k[h - i - 1] * moments[i]
                )
    return moments


def gamma_moment(i, alpha, beta):
    """Compute a moment (about zero) of a gamma distribution.

    There are different notations for the parameters of the
    gamma distribution. The notation used here matches [Moh12].
    It is different than the notation used on the Wikipedia page
    (wikipedia k = alpha here, wiki theta = beta here).

    Arguments:
        i: which moment
        alpha: shape parameter
        beta: scale parameter
    """
    return beta**i * gamma_func(alpha + i) / gamma_func(alpha)

def normal_moment(i, mu, sigma):
    """Compute a moment (about 0) of a normal distribution.

    See https://arxiv.org/pdf/1209.4340
    """

    if i % 2 == 0:
        return (
            sigma**i * 2**(i / 2) * gamma_func((i + 1) / 2) / np.pi**0.5
            * hyp1f1(-i / 2, 0.5, - mu**2 / (2 * sigma**2)))
    return (
        mu * sigma**(i - 1) * 2**((i + 1) / 2)
        * gamma_func(i / 2 + 1) / np.pi**0.5
        * hyp1f1((1 - i) / 2, 1.5, - mu**2 / (2 * sigma**2)))
