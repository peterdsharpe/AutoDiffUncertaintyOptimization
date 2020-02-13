"""Utilities for the Johnson SU distribution.

References:
    [Hil76] I. D. Hill, R. Hill, and R. L. Holder,
        "Algorithm AS 99: Fitting Johnson Curves by Moments,"
        Journal of the Royal Statistical Society. Series C (Applied Statistics),
        vol. 25, no. 2, pp. 180–189, 1976, doi: 10.2307/2346692.
    [Eld69] W. P. Elderton and N. L. Johnson,
        "Systems of Frequency Curves,"
        Cambridge university Press, 1969.
        Ch. 7 "Translation systems".
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


def find_gamma_delta(gamma_sign, beta_1, beta_2, tol=1e-9, max_iters=100):
    assert abs(gamma_sign) == 1
    assert beta_1 >= 0
    assert beta_2 >= 0

    # TODO check that beta-1, beta_2 are within the region that is possible
    # for Johnson SU.
    # This just checks that beta_1, beta_1 are in the possible region for *any*
    # distribution.
    assert beta_2 > beta_1 + 1

    # w is shorthand for little omega in [Eld69].
    # Initial guess at w
    # w = ((2 * beta_2 - 16 * beta_1 - 2)**0.5 - 1)**0.5
    w = 1.0408

    beta_1_d = np.inf
    for i in range(max_iters):
        # Solve the quadratic on m
        A_0 = w**5 + 3 * w**4 + 6 * w**3 + 10 * w**2 + 9 * w + 3
        A_1 = 8 * (w**4 + w**3 + 6 * w**2 + 7 * w + 3)
        A_2 = 8 * (w**3 + 3 * w**2 + 6 * w + 6)
        b = (beta_2 - 3) / (w - 1)
        m_roots = np.roots([
            A_2 - 8 * b, A_1 - 8 * b * (w + 1),
            A_0 - b * (w + 1)**2])
        print('m_roots = ' + repr(m_roots))
        m = max(m_roots)

        # Update the beta_1 which results from this m
        beta_1_d = (
            m * (w - 1) * (4 * (w + 2) * m + 3 * (w + 1)**2)**2
            / (2 * (2 * m + w  + 1)**3)
            )

        if abs(beta_1 - beta_1_d) < tol:
            break

        # Solve a quadratic for the updated value of w
        w2_roots = np.roots([
            -0.5, -0.5,
            -1.5 - beta_2 + beta_1 / beta_1_d * (beta_2 - 0.5 * (
                w**4 + 2 * w**2 + 3))])
        print('w2_roots = ' + repr(w2_roots))
        w2 = max(w2_roots)
        w = w2**0.5
    delta = (np.log(w))**(-0.5)
    gamma = delta * np.arcsinh((m / w)**0.5)
    gamma *= gamma_sign
    return gamma, delta
