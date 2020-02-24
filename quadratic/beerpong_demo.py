"""Beer pong demo."""
import numpy as np
import scipy.stats
from scipy import constants
from matplotlib import pyplot as plt

import quadratic


def projectile_range(v_init, theta):
    return v_init**2 * np.sin(2 * theta) / constants.g


def projectile_range_jac(v_init, theta):
    return np.array([
        2 * v_init * np.sin(2 * theta) / constants.g,
        2 * v_init**2 * np.cos(2 * theta) / constants.g
        ])


def projectile_range_hes(v_init, theta):
    dr2dv2 = 2 * np.sin(2 * theta) / constants.g
    dr2dtheta2 = -4 * v_init**2 * np.sin(2 * theta) / constants.g
    dr2dvdtheta = 4 * v_init * np.cos(2 * theta) / constants.g
    return np.array([
        [dr2dv2, dr2dvdtheta],
        [dr2dvdtheta, dr2dtheta2]
        ])


def plot_quad_approx():
    """Plot the quadratic approximation of the range function
        wrt theta."""
    theta = np.linspace(0, np.pi / 2)
    v_init = 2.
    range_true = projectile_range(v_init, theta)

    plt.figure()
    plt.plot(
        np.rad2deg(theta), range_true,
        color='black', label='True range')

    # Quadratic approximation of the range function.
    fit_points = (np.pi / 8, np.pi / 4, (3/8) * np.pi)
    for i, theta_0 in enumerate(fit_points):
        r_0 = projectile_range(v_init, theta_0)
        a = projectile_range_jac(v_init, theta_0)
        A = projectile_range_hes(v_init, theta_0)
        range_quad_approx = np.zeros(len(theta))
        for j in range(len(theta)):
            dx = np.array([0, theta[j] - theta_0])
            range_quad_approx[j] = dx @ A @ dx + a @ dx + r_0
        color = 'C{:d}'.format(i)
        plt.plot(
            np.rad2deg(theta), range_quad_approx,
            color=color, linestyle='--',
            label='Quadratic approx.')
        plt.scatter(
            np.rad2deg(theta_0), r_0,
            color=color, marker='x', label='Approx. eval. point'
            )
    plt.xlabel('Launch angle $\\theta$ [deg]')
    plt.ylabel('Range $r$ [m]')
    plt.ylim([0, plt.ylim()[1]])
    plt.legend()


def plot_range_pdf():
    """TODO Something is very wrong!"""
    v_init = 2.
    sigma_v = 0.01
    sigma_theta = np.deg2rad(5.)
    covar = np.diag([sigma_v**2, sigma_theta**2])

    plt.figure()

    thetas = (np.pi / 6, np.pi / 4, np.pi / 3)
    for i, theta in enumerate(thetas):
        mean = np.array([v_init, theta])

        r_0 = projectile_range(v_init, theta)
        a = projectile_range_jac(v_init, theta)
        A = projectile_range_hes(v_init, theta)

        approx_dist = quadratic.approx_quad_expr_nig(
            mean, covar, A, a, r_0)

        r = np.linspace(-2., 4., 100)

        color  = 'C{:d}'.format(i)
        plt.plot(
            r, approx_dist.pdf(r),
            label='$\\theta = {:.0f}$ deg. (approx. dist.)'.format(
                np.rad2deg(theta)),
            color=color)
        plt.scatter(
            r_0, 0.5, marker='x', color=color)

    plt.xlabel('Range $r$ [m]')
    plt.ylabel('pdf($r$) [m$^{-1}$]')
    plt.ylim([0, plt.ylim()[1]])
    plt.legend()


if __name__ == '__main__':
    # plot_quad_approx()
    plot_range_pdf()
    plt.show()
