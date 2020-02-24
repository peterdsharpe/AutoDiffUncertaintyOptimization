import numpy as np
import scipy.stats
from matplotlib import pyplot as plt

import quadratic

def demo():
    """ With A = covar = I and mean = 0, Q is a chi-squared random variable."""
    n_dim = 3
    A = np.eye(n_dim)
    covar = np.eye(n_dim)
    mean = np.zeros(n_dim)
    a = np.zeros(n_dim)
    d = 0
    approx_dist = quadratic.approx_quad_expr_nig(
        mean, covar, A, a, d)

    true_dist = scipy.stats.chi2(df=n_dim)

    q = np.linspace(0, 10)

    plt.figure()
    plt.plot(
        q, true_dist.pdf(q), label='True', color='black')
    plt.plot(
        q, approx_dist.pdf(q), label='Approx.',
        color='tab:blue', linestyle='--')
    plt.xlabel('q')
    plt.ylabel('pdf(q) [-]')
    plt.title(
        'Quadratic for $Q = xAx$\n'
        + '$A = \\mathrm{{covar}} = I_{:d}, \\mu = a = 0$'.format(
            n_dim))
    plt.legend()


def demo_neg():
    """A is negative definite."""
    n_dim = 3
    A = -1 * np.eye(n_dim)
    covar = np.eye(n_dim)
    mean = np.zeros(n_dim)
    a = np.zeros(n_dim)
    d = 0
    approx_dist = quadratic.approx_quad_expr_nig(
        mean, covar, A, a, d)

    true_dist = scipy.stats.chi2(df=n_dim)

    q = np.linspace(-10, 0)

    plt.figure()
    plt.plot(
        q, true_dist.pdf(-q), label='True', color='black')
    plt.plot(
        q, approx_dist.pdf(q), label='Approx.',
        color='tab:blue', linestyle='--')
    plt.xlabel('q')
    plt.ylabel('pdf(q) [-]')
    plt.title(
        'Quadratic for $Q = xAx$\n'
        + '$A = -I_{:d}, \\mathrm{{covar}} = I_{:d}, \\mu = 0$'.format(
            n_dim, n_dim))
    plt.legend()


def demo_expr():
    """Quadratic expression, with linear and constant terms."""
    n_dim = 2
    A = np.eye(n_dim)
    A[1, 1] = -0.5
    covar = np.eye(n_dim)
    mean = np.zeros(n_dim)
    a = np.ones(n_dim)
    d = 1.
    approx_dist = quadratic.approx_quad_expr_nig(
        mean, covar, A, a, d)

    # Sample from true dist
    n_sample = int(1e5)
    x = np.random.multivariate_normal(mean, covar, n_sample)
    q_samples = np.zeros(n_sample)
    for i in range(n_sample):
        q_samples[i] = x[i] @ A @ x[i] + a @ x[i] + d

    q = np.linspace(-10, 10)

    plt.figure()
    plt.plot(
        q, approx_dist.pdf(q), label='Approx.',
        color='tab:blue', linestyle='--')
    bins = np.linspace(-10, 10, 101)
    plt.hist(
        q_samples, density=True, histtype='stepfilled',
        bins=bins,
        alpha=0.5, color='black', label='Samples')
    plt.xlabel('q')
    plt.ylabel('pdf(q) [-]')
    plt.title('Quadratic expression, $Q = xAx + ax + d$')
    plt.legend()


if __name__ == '__main__':
    demo()
    demo_neg()
    demo_expr()
    plt.show()
