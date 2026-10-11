"""Zipf fixtures (scipy.stats.zipf). Rows are (k, a) and (probability, a).

Exact values from mpmath's Riemann and Hurwitz zeta functions: mass
k^-a / ζ(a), sf ζ(a, k+1)/ζ(a), cdf its complement at working precision.
Regions: a near 1 (to 1 + 1e-6), moderate and large a, k up to 1e15.

SciPy's zipf has no cdf, sf or quantile of its own: the generic ones sum
the masses from 1 to k, which for k beyond a few million exhausts memory.
The SciPy column is therefore left out ("-") where k (or, for the
quantiles, the answer) exceeds 1e6.
"""

import math

import mpmath
import scipy.stats as st
from mpmath import mpf

import discfix
from discfix import ilogu, logu, probs

N = 250
KMAX_SCIPY = 1e6


def logpmf(k, a):
    a = mpf(a)
    return -a * mpmath.log(k) - mpmath.log(mpmath.zeta(a))


def cdfsf(k, a):
    a = mpf(a)
    sf = mpmath.zeta(a, k + 1) / mpmath.zeta(a)
    return 1 - sf, sf


def support(a):
    return 1, math.inf


def _a(r, region):
    if region == "near1":
        return 1 + logu(r, 1e-6, 0.1)
    if region == "large_a":
        return logu(r, 10, 1000)
    return r.uniform(1.1, 10)


def _estimate(q, a):
    """k with sf(k) about q (sf(k) ~ k^(1-a) / ((a-1) ζ(a)))."""
    return ((a - 1) * float(mpmath.zeta(a)) * q) ** (-1 / (a - 1))


def _guard_x(f):
    def g(k, a):
        if k > KMAX_SCIPY:
            raise ValueError("SciPy cannot sum this far")
        return f(k, a)
    return g


def _guard_p(f, upper):
    def g(p, a):
        q = p if upper else 1 - p
        if not (_estimate(max(q, 1e-300), a) <= KMAX_SCIPY):
            raise ValueError("SciPy cannot sum this far")
        return f(p, a)
    return g


def xcases(name, fn):
    r = discfix.rng(name)
    out = []
    for region in ("near1", "core", "large_a"):
        for _ in range(N):
            a = _a(r, region)
            k = ilogu(r, 1, 1e15) if r.random() < 0.5 else r.randint(1, 50)
            out.append((region, (k, a)))
    return out


def pcases(name, fn):
    r = discfix.rng(name)
    return [(region, (probs(r, region), _a(r, ar))) for region in ("core", "lower", "upper")
            for ar in ("near1", "core", "large_a") for _ in range(N // 3)]


_sp = {fn: _guard_x(getattr(st.zipf, fn)) for fn in ("pmf", "logpmf", "cdf", "sf", "logcdf", "logsf")}
_sp["ppf"] = _guard_p(st.zipf.ppf, False)
_sp["isf"] = _guard_p(st.zipf.isf, True)

FIXTURES = discfix.make("zipf", dict(dist=st.zipf, logpmf=logpmf, cdfsf=cdfsf, support=support,
                                     xcases=xcases, pcases=pcases, spfns=_sp))
