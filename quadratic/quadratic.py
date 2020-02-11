import numpy as np
import scipy.stats
from matplotlib import pyplot as plt

import cumulants_moments
from poly_approx import poly_approx

def approx_quad_form(mean: np.ndarray,
                     covar: np.ndarray, A: np.ndarray):
    """Approximate the probability density function of a
    quadratic form of a normal random variable.
    """
    eigvals = np.linalg.eigvals(A)
    n_cumulants = 4
    cumulants = np.zeros(n_cumulants)
    for s in range(1, n_cumulants + 1):
        cumulants[s - 1] = cumulants_moments.cumulant_quad_form(
            s, mean, covar, A)

    moments = cumulants_moments.moments_from_cumulants(cumulants)

    if np.all(eigvals >= 0):
        # A is positive semidefinite.
        # Support of Q is [0, +inf).
        approx = poly_approx(moments, base='gamma')
    elif np.all(eigvals < 0):
        # A is negative definite.
        # Support of Q is (-inf, 0].
        # negate odd moments
        for i in range(len(moments)):
            if i % 2 == 1:
                moments[i] *= -1
        approx_neg = poly_approx(moments, base='gamma')
        approx = lambda x: approx_neg(-x)
    else:
        # A is indefinite.
        # Support of Q is (-inf, +inf)
        approx = poly_approx(moments, base='normal')
    return approx


def demo():
    """ With A = covar = I and mean = 0, Q is a chi-squared random variable."""
    n_dim = 3
    A = np.eye(n_dim)
    covar = np.eye(n_dim)
    mean = np.zeros(n_dim)
    approx = approx_quad_form(mean, covar, A)

    true_dist = scipy.stats.chi2(df=n_dim)

    q = np.linspace(0, 10)

    plt.plot(
        q, true_dist.pdf(q), label='True', color='black')
    plt.plot(
        q, approx(q), label='Approx.',
        color='tab:blue', linestyle='--')
    plt.xlabel('q')
    plt.ylabel('pdf(q) [-]')
    plt.legend()


def demo_neg():
    """A is negative definite."""
    n_dim = 3
    A = -1 * np.eye(n_dim)
    covar = np.eye(n_dim)
    mean = np.zeros(n_dim)
    approx = approx_quad_form(mean, covar, A)

    true_dist = scipy.stats.chi2(df=n_dim)

    q = np.linspace(-10, 0)

    plt.plot(
        q, true_dist.pdf(-1 * q), label='True', color='black')
    plt.plot(
        q, approx(q), label='Approx.',
        color='tab:blue', linestyle='--')
    plt.xlabel('q')
    plt.ylabel('pdf(q) [-]')
    plt.legend()


def demo_indef():
    """A is indefinite, with positive and negative eigenvalues."""
    n_dim = 2
    A = np.eye(n_dim)
    A[1, 1] = -1.
    covar = np.eye(n_dim)
    mean = np.zeros(n_dim)
    approx = approx_quad_form(mean, covar, A)

    # Sample from true dist
    n_sample = 10000
    x = np.random.multivariate_normal(mean, covar, n_sample)
    q_samples = np.zeros(n_sample)
    for i in range(n_sample):
        q_samples[i] = x[i] @ A @ x[i]

    q = np.linspace(-10, 10)

    plt.plot(
        q, approx(q), label='Approx.',
        color='tab:blue', linestyle='--')
    bins = np.linspace(-8, 8, 81)
    bins[0] = -np.inf
    bins[-1] = np.inf
    plt.hist(
        q_samples, density=True, histtype='stepfilled',
        bins=bins,
        alpha=0.5, color='black', label='Samples')
    plt.xlabel('q')
    plt.ylabel('pdf(q) [-]')
    plt.legend()


def demo_diag21():
    """A = diag([2, 1])"""
    n_dim = 2
    A = np.diag([2, 1])
    covar = np.eye(n_dim)
    mean = np.zeros(n_dim)
    approx = approx_quad_form(mean, covar, A)

    # Sample from true dist
    n_sample = 10000
    x = np.random.multivariate_normal(mean, covar, n_sample)
    q_samples = np.zeros(n_sample)
    for i in range(n_sample):
        q_samples[i] = x[i] @ A @ x[i]

    q = np.linspace(0, 10)

    plt.plot(
        q, approx(q), label='Approx.',
        color='tab:blue', linestyle='--')
    bins = np.linspace(0, 10, 51)
    bins[-1] = np.inf
    plt.hist(
        q_samples, density=True, histtype='stepfilled',
        bins=bins,
        alpha=0.5, color='black', label='Samples')
    plt.xlabel('q')
    plt.ylabel('pdf(q) [-]')
    plt.legend()


if __name__ == '__main__':
    # demo()
    # demo_neg()
    # demo_indef()
    demo_diag21()
    plt.show()
