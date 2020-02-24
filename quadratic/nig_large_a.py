"""Experiment with implementations of the normal-inverse gaussian pdf
which can handle large values of the shape parameter a (aka alpha_bar).

The default scipy implementation cannot handle large values of a.
"""
import numpy as np
import scipy.special as sc
from matplotlib import pyplot as plt
import nig

def pdf_scipy(x, a, b):
    """Copied from scipy.stats._continuous_distns"""
    gamma = np.sqrt(a**2 - b**2)
    fac1 = a / np.pi * np.exp(gamma)
    sq = np.hypot(1, x)  # reduce overflows
    return fac1 * sc.k1e(a * sq) * np.exp(b*x - a*sq) / sq

def pdf_erik(x, a, b):
    gamma = np.sqrt(a**2 - b**2)
    sq = np.hypot(1, x)  # reduce overflows
    return (
        a / np.pi
        * np.exp(gamma) * sc.k1e)

def pdf_mine(x, a, b):
    """my alternative implementation for large a."""
    gamma = np.sqrt(a**2 - b**2)
    fac1 = a / np.pi
    sq = np.hypot(1, x)  # reduce overflows
    exp_arg = gamma + b * x - a * sq
    return fac1 * sc.k1e(a * sq) * np.exp(exp_arg) / sq


def main(a, b):
    x = np.linspace(-1, 1, 100)
    plt.figure()
    plt.plot(
        x, pdf_scipy(x, a, b),
        color='black', label='scipy impl.')
    plt.plot(
        x, pdf_mine(x, a, b),
        color='tab:blue', linestyle=':', label='my impl.')
    dist = nig.norminvgauss(a=a, b=b)
    plt.plot(
        x, dist.pdf(x),
        color='tab:orange', linestyle='--', label='my impl. in nig.py')
    plt.xlabel('$x$')
    plt.ylabel('pdf($x$)')
    plt.title('NIG distribution, $a = {:.0f}$, $b = {:.0f}$'.format(
        a, b))
    plt.legend()


if __name__ == '__main__':
    main(a=2., b=1.)
    main(a=1800., b=200.)
    plt.show()
