"""Polynomial approximations of probability density functions.

References:
    [Pro05] Serge B Provost, "Moment-based Density Approximations,"
        The Mathematica Journal, 2005.
    [Moh12] Ali A Mohsenipour, "On the Distribution of Quadratic Expressions
        in Various Types of Random Vectors," PhD thesis, University of Western Ontario, 2012.
"""

import numpy as np
import scipy.stats
from cumulants_moments import gamma_moment
from matplotlib import pyplot as plt

def poly_approx(mu):
    """
    Arguments:
        mu: target moments

    See [Pro05] section 4 and [Moh12] section 2.7.3.
    """
    assert mu[0] == 1.
    n = len(mu) - 1

    # Parameters of the base gamma distribution.
    alpha = mu[1]**2 / (mu[2] - mu[1]**2)
    beta = mu[2] / mu[1] - mu[1]
    base = scipy.stats.gamma(alpha, scale=beta)

    # Create the matrix M of base distribution moments
    M = np.zeros((n + 1, n + 1))
    indexes = np.array(range(n + 1))
    for h in range(n + 1):
        M[h] = gamma_moment(h + indexes, alpha, beta)

    coefs = np.linalg.solve(M, mu)

    coefs_flipped = np.flip(coefs)
    approx = lambda x: base.pdf(x) * np.polyval(coefs_flipped, x)
    return approx


def demo():
    true_dist = scipy.stats.gamma(1)
    # Moments (about 0) of the gamma distribution with alpha = beta = 1
    target_moments = np.array([1., 1., 2., 6., 24., 120.])
    approx = poly_approx(target_moments)
    x = np.linspace(0, 10)

    plt.plot(
        x, true_dist.pdf(x), label='True', color='black')
    plt.plot(
        x, approx(x), label='Approx.',
        color='C0', linestyle='--')

    plt.xlabel('x')
    plt.ylabel('pdf(x) [-]')
    plt.legend()

    plt.show()


def demo2():
    """This is flawed -- trying to match a distribution on (0, inf) to one on (-inf, inf)"""
    true_dist = scipy.stats.norm(5, 2)
    # Moments (about 0)
    target_moments = np.array([true_dist.moment(i) for i in range(3)])
    approx = poly_approx(target_moments)
    x = np.linspace(0, 10)

    plt.plot(
        x, true_dist.pdf(x), label='True', color='black')
    plt.plot(
        x, approx(x), label='Approx.',
        color='C0', linestyle='--')

    plt.xlabel('x')
    plt.ylabel('pdf(x) [-]')
    plt.legend()

    plt.show()


def demo3():
    true_dist = scipy.stats.maxwell(scale=2)
    x = np.linspace(0, 10)

    plt.plot(
        x, true_dist.pdf(x), label='True', color='black')

    for n in [3, 4, 5]:
        # Moments (about 0)
        target_moments = np.array([
            true_dist.moment(i) for i in range(n)])
        approx = poly_approx(target_moments)
        plt.plot(
            x, approx(x), label='Approx. {:d} moments'.format(n),
            linestyle='--')

    plt.xlabel('x')
    plt.ylabel('pdf(x) [-]')
    plt.legend()
    plt.title('Approximation of Maxwell distribution')

    plt.show()

if __name__ == '__main__':
    demo3()
