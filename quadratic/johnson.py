"""Utilities for the Johnson SU distribution.

References:
    [Hil76] I. D. Hill, R. Hill, and R. L. Holder,
    "Algorithm AS 99: Fitting Johnson Curves by Moments,"
    Journal of the Royal Statistical Society. Series C (Applied Statistics),
    vol. 25, no. 2, pp. 180–189, 1976, doi: 10.2307/2346692.
"""
import numpy as np
from scipy.optimize import root


def johnson_su_params_from_moments(central_moments):
    assert len(central_moments) == 5
    assert central_moments[0] == 1
    # beta_1 is a measure of skewness
    beta_1 = central_moments[3] / central_moments[2]**(3/2.)
    # beta_2 is a measure of kurtosis
    beta_2 = central_moments[4] / central_moments[2]**2

    ### Find delta and gamma parameters ###
    if abs(beta_1) < 1e-9:
        # Symmetric case
        omega = ((2 * beta_2 - 2)**0.5 - 1)**0.5
        delta = np.log(omega)**(-0.5)
        gamma = 0.
    else:
        # Asymmetric case
        # TODO
        raise NotImplementedError()

    ### Find xi and lambda parameters ###
    sinh_term = omega**0.5 * np.sinh(gamma / delta)
    cosh_term = 0.5 * (omega - 1) * (omega * np.cosh(2 * gamma / delta) + 1)
    args = (
        central_moments[1], central_moments[2], sinh_term, cosh_term)
    guess = (central_moments[1], central_moments[2]**0.5)
    sol = root(_xi_lambda_helper, guess, args, jac=True)
    if not sol.success:
        raise RuntimeError('Root finding failed.')
    xi, lamb = sol.x
    return (xi, lamb, gamma, delta)


def _xi_lambda_helper(x, mu_1, mu_2, sinh_term, cosh_term):
    """Helper function to solve for xi and lambda parameters of Johnson SU distribution.

    Arguments:
        mu_1: desired first moment
        mu_2: desired second central moment
        sinh_term: sinh term,
            omega**0.5 * sinh(gamma / delta)
        cosh_term: cosh term,
            0.5 * (omega - 1) * (omega * cosh(2 * gamma / delta) + 1)
    """
    xi, lamb = x
    error = [
        (xi - lamb * sinh_term) - mu_1,
        (lamb**2 * cosh_term) - mu_2
    ]
    jacobian = np.array([
        [1., -sinh_term],
        [0., 2 * lamb * cosh_term]
    ])
    return (error, jacobian)
