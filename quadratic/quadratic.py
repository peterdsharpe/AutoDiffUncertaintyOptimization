"""
References:
    [Moh12] Ali A Mohsenipour, "On the Distribution of Quadratic Expressions
        in Various Types of Random Vectors," PhD thesis, University of Western Ontario, 2012.
"""
import numpy as np
from numpy.linalg import matrix_power
from math import factorial

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
            np.trace(ASig)**s / s
            + mean @ matrix_power(ASig, s - 1) @ A @ mean)
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
