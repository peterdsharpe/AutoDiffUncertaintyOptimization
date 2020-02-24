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
    v_init = 5.
    range_true = projectile_range(v_init, theta)

    plt.figure()
    plt.plot(
        np.rad2deg(theta), range_true,
        color='black', label='True range')

    # Quadratic approximation of the range function.
    fit_points = (np.pi / 6, np.pi / 4, np.pi / 3)
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
    plt.title('Range vs. launch angle')


def plot_derivs():
    """Plot the derivatives of the range function
        wrt theta."""
    theta = np.linspace(0, np.pi / 2)
    v_init = 5.
    range_true = projectile_range(v_init, theta)
    jac = np.array([projectile_range_jac(v_init, t) for t in theta])

    fig, axes = plt.subplots(
        nrows=2, ncols=1, figsize=(6, 7),
        sharex=True)
    ax_dv = axes[0]
    ax_dtheta = axes[1]
    ax_dv.plot(
        np.rad2deg(theta), jac[:, 0],
        color='black', label='True range')
    ax_dtheta.plot(
        np.rad2deg(theta), jac[:, 1],
        color='black', label='True range')
    ax_dtheta.axhline(y=0, color='grey', zorder=0)
    for ax in axes:
        ax.legend()
    ax_dtheta.set_xlabel('Launch angle $\\theta$ [deg]')
    ax_dv.set_ylim([0, ax_dv.get_ylim()[1]])
    ax_dv.set_ylabel('$dr / d v_{init}$ [s]')
    ax_dtheta.set_ylabel('$dr / d\\theta$ [m rad$^{-1}$]')
    plt.suptitle('Partial derivatives of range')


def plot_range_pdf():
    v_init = 5.
    sigma_v = 0.1
    sigma_theta = np.deg2rad(1.)
    covar = np.diag([sigma_v**2, sigma_theta**2])
    mean = np.array([0., 0.])

    plt.figure()

    thetas = (np.pi / 6, np.pi / 4, np.pi / 3)
    for i, theta in enumerate(thetas):
        r_0 = projectile_range(v_init, theta)
        a = projectile_range_jac(v_init, theta)
        A = projectile_range_hes(v_init, theta)

        approx_dist = quadratic.approx_quad_expr_nig(
            mean, covar, A, a, r_0)

        r = np.linspace(0., 3.5, 100)

        color = 'C{:d}'.format(i)
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
    plt.title(
        'Range pdf for different launch angles'
        + '\nstddev($v_{{init}}$) = {:.2f} m s$^{{-1}}$, stddev($\\theta$) = {:.1f} deg'.format(
            sigma_v, np.rad2deg(sigma_theta)))

def plot_range_pdf_vs_samples(sigma_v=0.1, sigma_theta=np.deg2rad(1.)):
    v_init = 5.
    covar = np.diag([sigma_v**2, sigma_theta**2])
    mean = np.array([0., 0.])

    plt.figure()

    theta = np.pi / 6
    r_0 = projectile_range(v_init, theta)
    a = projectile_range_jac(v_init, theta)
    A = projectile_range_hes(v_init, theta)

    approx_dist = quadratic.approx_quad_expr_nig(
        mean, covar, A, a, r_0)

    # Sample from true dist
    n_sample = int(1e5)
    x_samples = np.random.multivariate_normal(
        [v_init, theta], covar, n_sample)
    range_samples = np.array([
        projectile_range(*x) for x in x_samples])

    r = np.linspace(0., 3.5, 200)
    bins = np.linspace(1, 3.5, 100)

    plt.plot(
        r, approx_dist.pdf(r),
        label='Quadratic approximation dist.', color='tab:blue')
    plt.hist(
        range_samples, density=True, histtype='stepfilled',
        bins=bins,
        alpha=0.5, color='black', label='Samples from true dist.')


    plt.xlabel('Range $r$ [m]')
    plt.ylabel('pdf($r$) [m$^{-1}$]')
    plt.ylim([0, plt.ylim()[1]])
    plt.legend()
    plt.title(
        'Range pdf: quad. approx. vs. samples from true distribution'
        + '\nstddev($v_{{init}}$) = {:.2f} m s$^{{-1}}$, stddev($\\theta$) = {:.1f} deg'.format(
            sigma_v, np.rad2deg(sigma_theta)))


def plot_stddev_vs_launch_angle():
    v_init = 5.
    sigma_theta = np.deg2rad(1.)
    mean = np.array([0., 0.])

    plt.figure()
    theta = np.linspace(0, np.pi / 2)

    for iv, sigma_v in enumerate((0.1, 0.02)):
        covar = np.diag([sigma_v**2, sigma_theta**2])
        stddev_r = np.zeros(len(theta))

        for i, theta_ in enumerate(theta):
            r_0 = projectile_range(v_init, theta_)
            a = projectile_range_jac(v_init, theta_)
            A = projectile_range_hes(v_init, theta_)

            approx_dist = quadratic.approx_quad_expr_nig(
                mean, covar, A, a, r_0)
            stddev_r[i] = approx_dist.std()

        color = 'C{:d}'.format(iv)
        plt.plot(
            np.rad2deg(theta), stddev_r,
            label='stddev($v_{{init}}$) = {:.2f} m s$^{{-1}}$'.format(
                sigma_v),
            color=color)

    plt.xlabel('Launch angle $\\theta$ [deg]')
    plt.ylabel('stddev($r$) [m]')
    plt.ylim([0, plt.ylim()[1]])
    plt.legend()
    plt.title(
        'Range standard deviation vs. launch angle, evaluated by quad. approx.\n'
        + 'for different levels of initial velocity uncertainty\n'
        + 'stddev($\\theta$) = {:.1f} deg'.format(
            np.rad2deg(sigma_theta)))
    plt.tight_layout()


if __name__ == '__main__':
    plot_derivs()
    plot_quad_approx()
    plot_range_pdf()
    plot_range_pdf_vs_samples()
    plot_range_pdf_vs_samples(sigma_theta=np.deg2rad(5.))
    plot_stddev_vs_launch_angle()
    plt.show()
