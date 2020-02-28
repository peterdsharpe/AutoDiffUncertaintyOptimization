import numpy as np
import scipy.stats
from matplotlib import pyplot as plt
from statsmodels.stats.moment_helpers import mnc2mc

import cumulants_moments
from poly_approx import poly_approx
import johnson

def main():
    n_dim = 2
    A = np.eye(n_dim)
    # A[1, 1] = -1.
    A[1, 1] = -0.5
    covar = np.eye(n_dim)
    mean = np.zeros(n_dim)

    # Sample from true dist
    n_sample = int(1e5)
    x = np.random.multivariate_normal(mean, covar, n_sample)
    q_samples = np.zeros(n_sample)
    for i in range(n_sample):
        q_samples[i] = x[i] @ A @ x[i]

    q = np.linspace(-10, 10)


    n_cumulants = 4
    cumulants = np.zeros(n_cumulants)
    for s in range(1, n_cumulants + 1):
        cumulants[s - 1] = cumulants_moments.cumulant_quad_form(
            s, mean, covar, A)

    moments_moh = cumulants_moments.moments_from_cumulants(cumulants)

    # Find central moments
    central_moments_moh = np.zeros(len(moments_moh))
    central_moments_moh[1:] = mnc2mc(moments_moh[1:])
    central_moments_moh[0] = 1.

    # Compute the parameters of the Johnson SU dist which has those moments
    xi, lamb, gamma, delta = johnson.johnson_su_params_from_moments(
        central_moments_moh[:5])
    dist = scipy.stats.johnsonsu(
        loc=xi, scale=lamb,
        a=gamma, b=delta)
    approx = dist.pdf

    # Compute the moments of the approximate distribution
    central_moments_approx = np.zeros(len(moments_moh))
    central_moments_approx[0] = 1.
    for i in range(1, len(moments_moh)):
        central_moments_approx[i] = dist.moment(i)

    # Compute the moments of the sample
    central_moments_sample = scipy.stats.moment(
        q_samples, moment=[0, 1, 2, 3, 4])

    print('\n')
    print('Central moment:    1        2        3       4')
    print('Moh. formula:  {:7.3f}  {:7.3f}  {:7.3f}  {:7.3f}'.format(
        *central_moments_moh[1:5]))
    print('Approx. dist.: {:7.3f}  {:7.3f}  {:7.3f}  {:7.3f}'.format(
        *central_moments_approx[1:5]))
    print('Samples:       {:7.3f}  {:7.3f}  {:7.3f}  {:7.3f}'.format(
        *central_moments_sample[1:5]))

    q = np.linspace(-10, 10)

    plt.plot(
        q, approx(q), label='Approx.',
        color='tab:blue', linestyle='--')
    bins = np.linspace(-10, 10, 101)
    bins[0] = -np.inf
    bins[-1] = np.inf
    plt.hist(
        q_samples, density=True, histtype='stepfilled',
        bins=bins,
        alpha=0.5, color='black', label='Samples')
    plt.xlabel('q')
    plt.ylabel('pdf(q) [-]')
    plt.yscale('log')
    plt.legend()


if __name__ == '__main__':
    main()
    plt.show()
