import numpy as np
import scipy.stats
from matplotlib import pyplot as plt
from statsmodels.stats.moment_helpers import mnc2mc

import cumulants_moments
import nig


def main():
    n_dim = 2
    A = np.eye(n_dim)
    A[1, 1] = -0.001
    # A[1, 1] = -0.5
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

    # Compute the parameters of the NIG dist which has those cumulants
    cumulants_approx = nig.make_feasible(cumulants)
    alpha_bar, beta_bar, mu, delta = nig.parameters_from_cumulants(
        cumulants_approx)

    dist = scipy.stats.norminvgauss(
        loc=mu, scale=delta,
        a=alpha_bar, b=beta_bar)
    approx = dist.pdf

    # Compute the moments of the sample
    cumulants_sample = [scipy.stats.kstat(q_samples, s) for s in (1, 2, 3, 4)]

    print('\n')
    print('Cumulant:          1        2        3       4')
    print('Moh. formula:  {:7.3f}  {:7.3f}  {:7.3f}  {:7.3f}'.format(
        *cumulants))
    print('Approx. dist.: {:7.3f}  {:7.3f}  {:7.3f}  {:7.3f}'.format(
        *cumulants_approx))
    print('Samples:       {:7.3f}  {:7.3f}  {:7.3f}  {:7.3f}'.format(
        *cumulants_sample))

    q = np.linspace(-10, 10, 200)

    bins = np.linspace(-10, 10, 101)
    bins[0] = -np.inf
    bins[-1] = np.inf

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(6, 7))
    for ax in axes:
        ax.plot(
            q, approx(q), label='Approx.',
            color='tab:blue', linestyle='--')
        ax.hist(
            q_samples, density=True, histtype='stepfilled',
            bins=bins,
            alpha=0.5, color='black', label='Samples')
        ax.set_xlabel('q')
        ax.set_ylabel('pdf(q) [-]')
    axes[1].set_yscale('log')
    plt.legend()


if __name__ == '__main__':
    main()
    plt.show()
