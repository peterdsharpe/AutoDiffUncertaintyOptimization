import numpy as np
import scipy.stats
from matplotlib import pyplot as plt

import cumulants_moments
import poly_approx

def approx_quad_form(mean: np.ndarray,
                     covar: np.ndarray, A: np.ndarray):
    """Approximate the probability density function of a
    quadratic form of a normal random variable.
    """
    eigvals = np.linalg.eigvals(A).sort()
    n_cumulants = 4
    cumulants = np.zeros(n_cumulants)
    for s in range(1, n_cumulants + 1):
        cumulants[s - 1] = cumulants_moments.cumulant_quad_form(
            s, mean, covar, A)

    moments = cumulants_moments.moments_from_cumulants(cumulants)

    return poly_approx.poly_approx(moments)


def demo():
    # With A = covar = I and mean = 0, Q is a chi-squared random variable.
    n_dim = 3
    A = np.eye(n_dim)
    covar = np.eye(n_dim)
    mean = np.zeros(n_dim)
    approx = approx_quad_form(mean, covar, A)

    true_dist = scipy.stats.chi2(df=n_dim)

    x = np.linspace(0, 10)

    plt.plot(
        x, true_dist.pdf(x), label='True', color='black')
    plt.plot(
        x, approx(x), label='Approx.',
        color='tab:blue', linestyle='--')
    plt.xlabel('x')
    plt.ylabel('pdf(x) [-]')
    plt.legend()

if __name__ == '__main__':
    demo()
    plt.show()
