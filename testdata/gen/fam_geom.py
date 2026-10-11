"""Geometric fixtures (scipy.stats.geom). Rows are (k, p) and
(probability, p).

Exact closed forms at working precision: mass (1-p)^(k-1) p, sf
(1-p)^k = exp(k log1p(-p)), cdf -expm1 of the same. Regions: p from 1e-18
to 1 - 1e-16, k up to 2^53, sf from 1 down to the underflow and, for the
logarithms, below it.
"""

import math

import mpmath
import scipy.stats as st
from mpmath import mpf

import discfix
from discfix import logu, probs

N = 300


def logpmf(k, p):
    p = mpf(p)
    return (k - 1) * mpmath.log1p(-p) + mpmath.log(p)


def cdfsf(k, p):
    t = k * mpmath.log1p(-mpf(p))
    return -mpmath.expm1(t), mpmath.exp(t)


def support(p):
    return 1, math.inf


def _p(r, region):
    if region == "small_p":
        return logu(r, 1e-18, 1e-3)
    if region == "near1":
        return 1.0 - logu(r, 1.2e-16, 1e-3)
    return r.uniform(1e-3, 0.999)


def xcases(name, fn):
    r = discfix.rng(name)
    out = []
    deep = fn in ("logpmf", "logcdf", "logsf")
    for region in ("small_p", "core", "near1"):
        for _ in range(N):
            p = _p(r, region)
            lq = -math.log1p(-p)
            # k·(-log(1-p)) from 1e-6 to 700 (to 1e5 for the logarithms)
            t = logu(r, 1e-6, 1e5 if deep else 700)
            k = int(min(2 ** 53, max(1, round(t / lq))))
            out.append((region, (k, p)))
    for _ in range(N):
        out.append(("small_k", (r.randint(1, 30), _p(r, r.choice(["small_p", "core", "near1"])))))
    return out


def pcases(name, fn):
    r = discfix.rng(name)
    return [(region, (probs(r, region), _p(r, pr))) for region in ("core", "lower", "upper")
            for pr in ("small_p", "core", "near1") for _ in range(N // 3)]


FIXTURES = discfix.make("geom", dict(dist=st.geom, logpmf=logpmf, cdfsf=cdfsf, support=support,
                                     xcases=xcases, pcases=pcases))
