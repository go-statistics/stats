"""Negative binomial fixtures (scipy.stats.nbinom). Rows are (k, n, p) and
(probability, n, p).

The exact mass is Γ(n+k)/(Γ(n) k!) p^n (1-p)^k from log-gamma at working
precision; cdf and sf are I_p(n, k+1) and its complement from
mpincomplete.ib_I_Ic. Regions: n from 1e-6 to 1e12 (real), p from 1e-12 to
1 - 1e-16, the bulk, the tails and, for the logarithms, beyond the
underflow.
"""

import math

import mpmath
import scipy.stats as st
from mpmath import mpf

import discfix
from discfix import logu, probs
from mpincomplete import ib_I_Ic

N = 250
NQ = 120


def logpmf(k, n, p):
    n, p = mpf(n), mpf(p)
    return (mpmath.loggamma(n + k) - mpmath.loggamma(k + 1) - mpmath.loggamma(n)
            + n * mpmath.log(p) + k * mpmath.log1p(-p))


def cdfsf(k, n, p):
    return ib_I_Ic(mpf(n), mpf(k + 1), mpf(p))


def ratio(j, n, p):
    n, p = mpf(n), mpf(p)
    return (n + j) * (1 - p) / (j + 1)


def support(n, p):
    return 0, math.inf


def _shape(r, region):
    if region == "small_n":
        return logu(r, 1e-6, 1), r.uniform(0.05, 0.95)
    if region == "large_n":
        return logu(r, 1e4, 1e12), r.uniform(0.05, 0.95)
    if region == "small_p":
        return logu(r, 0.1, 100), logu(r, 1e-12, 1e-3)
    if region == "p_near_1":
        return logu(r, 0.1, 1e4), 1.0 - logu(r, 1.2e-16, 1e-3)
    return logu(r, 1, 1e4), r.uniform(0.02, 0.98)


def _k(n, p, z):
    q = 1 - p
    m, s = n * q / p, math.sqrt(n * q) / p
    return int(max(0, round(m + z * s)))


def xcases(name, fn):
    r = discfix.rng(name)
    cdfish = fn not in ("pmf", "logpmf")
    out = []
    for region, count in (("small_n", N), ("bulk", N), ("large_n", NQ), ("small_p", N), ("p_near_1", N)):
        for _ in range(count):
            n, p = _shape(r, region)
            if region == "small_n":
                k = ilog(r, _k(n, p, 6) + 2)
            else:
                k = _k(n, p, r.uniform(-5, 5))
            out.append((region, (k, n, p)))
    if cdfish:
        for region, sgn in (("lower", -1), ("upper", 1)):
            for _ in range(N):
                n, p = logu(r, 30, 1e8), r.uniform(0.05, 0.95)
                out.append((region, (_k(n, p, sgn * r.uniform(3, 30)), n, p)))
        if fn in ("logcdf", "logsf"):
            for _ in range(N):
                n, p = logu(r, 1, 1e3), r.uniform(0.05, 0.95)
                z = r.choice([-1, 1]) * r.uniform(30, 300)
                out.append(("deep", (_k(n, p, z), n, p)))
    else:
        for _ in range(N):
            n, p = logu(r, 10, 1e8), r.uniform(0.05, 0.95)
            out.append(("tail", (_k(n, p, r.choice([-1, 1]) * r.uniform(6, 30)), n, p)))
        if fn == "logpmf":
            for _ in range(N):
                n, p = logu(r, 1e-3, 1e12), r.uniform(0.01, 0.99)
                out.append(("deep", (int(logu(r, 1, 1e15)), n, p)))
    return out


def ilog(r, hi):
    return int(logu(r, 1, hi + 1)) - 1


def pcases(name, fn):
    r = discfix.rng(name)
    out = []
    for region in ("core", "lower", "upper"):
        for _ in range(N):
            n, p = _shape(r, r.choice(["small_n", "bulk", "small_p", "p_near_1"]))
            out.append((region, (probs(r, region), n, p)))
    for _ in range(NQ):
        n, p = _shape(r, "large_n")
        out.append(("large_n", (probs(r, "core"), n, p)))
    return out


FIXTURES = discfix.make("nbinom", dict(dist=st.nbinom, logpmf=logpmf, cdfsf=cdfsf, support=support,
                                       xcases=xcases, pcases=pcases, ratio=ratio))
