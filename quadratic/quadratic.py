import numpy as np
import scipy.stats

import cumulants_moments
import nig


def approx_quad_expr_nig(
        mean: np.ndarray, covar: np.ndarray,
        A: np.ndarray, a: np.array, d: float):
    """Approximate the distribution of a quadratic expression with a
    Normal-Inverse Gaussian distribution."""
    n_cumulants = 4
    cumulants = np.zeros(n_cumulants)
    for s in range(1, n_cumulants + 1):
        cumulants[s - 1] = cumulants_moments.cumulant_quad_expr(
            s, mean, covar, A, a, d)

    # Compute the parameters of the NIG dist which has those cumulants
    cumulants_approx = nig.make_feasible(cumulants)
    alpha_bar, beta_bar, mu, delta = nig.parameters_from_cumulants(
        cumulants_approx)

    approx_dist = scipy.stats.norminvgauss(
        loc=mu, scale=delta,
        a=alpha_bar, b=beta_bar)
    return approx_dist
