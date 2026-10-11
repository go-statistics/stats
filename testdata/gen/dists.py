"""Helpers shared by the distribution fixture families (fam_*.py).

The oracle for a quantile is the exact inverse of the exact distribution
function at the float64 probability, found in mpmath by a safeguarded
Newton iteration on the logarithm of the cdf (or sf): the logarithm keeps
the iteration well scaled from p = 1e-300 to p = 1 - 1e-16.
"""

import math

import mpmath
from mpmath import mp, mpf

import common


def invert(logf, dlogf, target, x0, lo, hi, decreasing=False):
    """Solve logf(x) = log(target) for x in (lo, hi) (mpf bounds, may be
    +-inf). logf is increasing (a log cdf) unless decreasing (a log sf).
    dlogf is its derivative. x0 is a float starting guess (SciPy's answer,
    which is close). Returns an mpf, or None if no bracket is found."""
    t = mpmath.log(mpf(target))
    s = -1 if decreasing else 1

    def g(x):
        return s * (logf(x) - t)  # increasing in x

    x = mpf(x0)
    if not (lo < x < hi) or not mpmath.isfinite(x):
        x = (lo + hi) / 2 if mpmath.isfinite(lo) and mpmath.isfinite(hi) else mpf(0)
    # bracket [a, b] with g(a) < 0 < g(b)
    gx = g(x)
    if gx == 0:
        return x
    step = max(abs(x) * mpf(2) ** -20, mpf(2) ** -60)
    a = b = x
    for _ in range(4000):
        if gx < 0:
            b = x + step
            if b >= hi:
                b = (x + hi) / 2 if mpmath.isfinite(hi) else x + step
            gb = g(b)
            if gb > 0:
                a = x
                break
            x, gx = b, gb
        else:
            a = x - step
            if a <= lo:
                a = (x + lo) / 2 if mpmath.isfinite(lo) else x - step
            ga = g(a)
            if ga < 0:
                b = x
                break
            x, gx = a, ga
        step *= 2
    else:
        return None
    # safeguarded Newton inside [a, b]
    x = (a + b) / 2 if not (a < mpf(x0) < b) else mpf(x0)
    tol = mpf(2) ** (-(mp.prec - 8))
    for _ in range(500):
        gx = g(x)
        if gx == 0:
            return x
        if gx < 0:
            a = x
        else:
            b = x
        d = s * dlogf(x)
        nx = x - gx / d if d != 0 else (a + b) / 2
        if not (a < nx < b):
            nx = (a + b) / 2
        if abs(nx - x) <= tol * max(abs(x), mpf(2) ** -1000):
            return nx
        if abs(b - a) <= tol * max(abs(a), abs(b), mpf(2) ** -1000):
            return (a + b) / 2
        x = nx
    return x


def logu(r, lo, hi, n, sign=1):
    return common.loguniform(r, lo, hi, n, sign)


def uni(r, lo, hi, n):
    return common.uniform(r, lo, hi, n)


def probs_lower(r, n, lo=1e-300, hi=1e-3):
    return logu(r, lo, hi, n)


def probs_core(r, n):
    return uni(r, 1e-3, 1 - 1e-3, n)


def probs_upper(r, n, lo=1.2e-16, hi=1e-3):
    """p close to 1: 1 - q for q log-uniform (the float p, as a caller holds it)."""
    return [1.0 - q for q in logu(r, lo, hi, n)]
