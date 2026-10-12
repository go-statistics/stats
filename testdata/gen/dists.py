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
    which is usually close). Returns an mpf, mpf(0) when lo = 0 and the root
    is below the smallest subnormal (it rounds to 0), or None if no bracket
    is found.

    On a positive support the iteration runs in t = log x, where a tail
    like x^a with tiny a is close to linear; elsewhere in x."""
    t_ = mpmath.log(mpf(target))
    s = -1 if decreasing else 1
    pos = lo >= 0

    def g(x):
        return s * (logf(x) - t_)  # increasing in x

    if pos:
        tiny = mpf(2) ** -1080
        if g(tiny) > 0:
            return mpf(0)
        # work in u = log x
        def G(u):
            return g(mpmath.exp(u))

        def dG(u):
            x = mpmath.exp(u)
            return s * dlogf(x) * x
        ulo = mpmath.log(tiny)
        uhi = mpmath.log(hi) if mpmath.isfinite(hi) else mpf(800)
        u0 = mpmath.log(mpf(x0)) if (x0 is not None and x0 > 0 and math.isfinite(x0)) else (ulo + uhi) / 2
        r = _solve(G, dG, u0, ulo, uhi)
        return None if r is None else mpmath.exp(r)
    x00 = mpf(x0) if (x0 is not None and math.isfinite(x0)) else mpf(0)
    blo = lo if mpmath.isfinite(lo) else mpf(-1e300)
    bhi = hi if mpmath.isfinite(hi) else mpf(1e300)
    return _solve(g, lambda x: s * dlogf(x), x00, blo, bhi)


def _solve(G, dG, x0, lo, hi):
    """Root of the increasing G in (lo, hi): bracket from x0 by doubling
    steps, then safeguarded Newton (bisection when a step leaves the
    bracket) to working precision."""
    x = mpf(x0)
    if not (lo < x < hi):
        x = (lo + hi) / 2
    gx = G(x)
    if gx == 0:
        return x
    step = max(abs(x) * mpf(2) ** -20, mpf(2) ** -40)
    a = b = x
    for _ in range(4000):
        if gx < 0:
            b = min(x + step, hi)
            gb = G(b)
            if gb > 0:
                a = x
                break
            if b >= hi:
                return None
            x, gx = b, gb
        else:
            a = max(x - step, lo)
            ga = G(a)
            if ga < 0:
                b = x
                break
            if a <= lo:
                return None
            x, gx = a, ga
        step *= 2
    else:
        return None
    x = (a + b) / 2 if not (a < mpf(x0) < b) else mpf(x0)
    tol = mpf(2) ** (-(mp.prec - 8))
    for _ in range(1000):
        gx = G(x)
        if gx == 0:
            return x
        if gx < 0:
            a = x
        else:
            b = x
        d = dG(x)
        nx = x - gx / d if d != 0 else (a + b) / 2
        if not (a < nx < b):
            nx = (a + b) / 2
        if abs(nx - x) <= tol * max(abs(x), mpf(1)):
            return nx
        if abs(b - a) <= tol * max(abs(a), abs(b), mpf(1)):
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
