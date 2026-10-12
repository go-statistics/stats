"""Binomial and Bernoulli fixtures (scipy.stats.binom, scipy.stats.bernoulli).

Rows are (k, n, p) and (probability, n, p) for binom, (k, p) and
(probability, p) for bernoulli. The exact mass is C(n,k) p^k (1-p)^(n-k)
from log-gamma at working precision; cdf and sf are I_(1-p)(n-k, k+1) and
I_p(k+1, n-k) from mpincomplete.ib_I_Ic (series, or quadrature for large
parameters). Regions: small n, the bulk of moderate n, n up to 1e15, p down
to 1e-300 and up to 1 - 1e-16, the tails to 37 standard deviations and,
for the logarithms, beyond the underflow of the probabilities.
"""

import math

import mpmath
import scipy.stats as st
from mpmath import mpf

import discfix
from discfix import ilogu, logu, probs
from mpincomplete import ib_I_Ic

N = 250
NQ = 120  # regions whose cdf oracle integrates numerically


def _lbin(n, k):
    return mpmath.loggamma(n + 1) - mpmath.loggamma(k + 1) - mpmath.loggamma(n - k + 1)


def logpmf(k, n, p):
    n, p = int(n), mpf(p)
    return _lbin(mpf(n), mpf(k)) + k * mpmath.log(p) + (n - k) * mpmath.log1p(-p)


def cdfsf(k, n, p):
    n = int(n)
    if k >= n:
        return mpf(1), mpf(0)
    i, ic = ib_I_Ic(mpf(k + 1), mpf(n - k), mpf(p))
    return ic, i


def ratio(j, n, p):
    n, p = int(n), mpf(p)
    return (n - j) * p / ((j + 1) * (1 - p))


def support(n, p):
    return 0, int(n)


def _k(n, p, z):
    m = n * p
    s = math.sqrt(n * p * (1 - p))
    return int(min(n, max(0, round(m + z * s))))


def _shape(r, region):
    """(n, p) for a region."""
    if region == "small":
        return r.randint(1, 100), r.uniform(0.01, 0.99)
    if region == "large_n":
        n = ilogu(r, 1e6, 1e15)
        return n, r.choice([r.uniform(0.01, 0.99), logu(r, 1e3 / n, 0.5)])
    if region == "small_p":
        return ilogu(r, 2, 1e15), logu(r, 1e-300, 1e-3)
    if region == "p_near_1":
        return ilogu(r, 2, 1e6), 1.0 - logu(r, 1.2e-16, 1e-3)
    return ilogu(r, 100, 1e6), r.uniform(0.02, 0.98)


def xcases(name, fn):
    r = discfix.rng(name)
    cdfish = fn not in ("pmf", "logpmf")
    out = []

    def add(region, count, zlo, zhi, sym=True):
        for _ in range(count):
            n, p = _shape(r, region if region in ("small", "large_n", "small_p", "p_near_1") else "bulk")
            if region == "small_p":
                k = r.randint(0, min(n, 3 + int(3 * n * p)))
            elif region == "p_near_1":
                k = n - r.randint(0, min(n, 3 + int(3 * n * (1 - p))))
            elif region in ("small", "bulk", "large_n") and not cdfish and region == "small":
                k = r.randint(0, n)
            else:
                z = r.uniform(zlo, zhi) * (r.choice([-1, 1]) if sym else 1)
                k = _k(n, p, z)
            if cdfish and k >= n:
                k = n - 1
            out.append((region, (k, n, p)))

    add("small", N, -4, 4)
    add("bulk", N, -5, 5)
    add("large_n", NQ, -6, 6)
    add("small_p", N, 0, 0)
    add("p_near_1", N, 0, 0)
    if cdfish:
        add("lower", N, -37, -3, sym=False)
        add("upper", N, 3, 37, sym=False)
        if fn in ("logcdf", "logsf"):
            # beyond the underflow, for a moderate spread (the tail sums)
            for _ in range(N):
                n, p = ilogu(r, 20, 1e4), r.uniform(0.05, 0.95)
                k = r.randint(0, n - 1)
                out.append(("deep", (k, n, p)))
    else:
        add("tail", N, 6, 35)
        if fn == "logpmf":
            for _ in range(N):
                n, p = ilogu(r, 20, 1e15), r.uniform(0.01, 0.99)
                out.append(("deep", (r.randint(0, n), n, p)))
    return out


def pcases(name, fn):
    r = discfix.rng(name)
    out = []
    for region, count in (("core", N), ("lower", N), ("upper", N)):
        for _ in range(count):
            n, p = ilogu(r, 1, 1e6), r.uniform(0.01, 0.99)
            out.append((region, (probs(r, region), n, p)))
    for _ in range(NQ):
        n = ilogu(r, 1e6, 1e15)
        out.append(("large_n", (probs(r, "core"), n, r.uniform(0.01, 0.99))))
    for _ in range(N):
        out.append(("small_p", (probs(r, "core"), ilogu(r, 2, 1e15), logu(r, 1e-300, 1e-3))))
    return out


# ------------------------------------------------------------ bernoulli

def b_logpmf(k, p):
    p = mpf(p)
    return mpmath.log1p(-p) if k == 0 else mpmath.log(p)


def b_cdfsf(k, p):
    if k >= 1:
        return mpf(1), mpf(0)
    p = mpf(p)
    return 1 - p, p


def b_support(p):
    return 0, 1


def _bp(r, region):
    if region == "tiny":
        return logu(r, 1e-300, 1e-3)
    if region == "near1":
        return 1.0 - logu(r, 1.2e-16, 1e-3)
    return r.uniform(1e-3, 1 - 1e-3)


def b_xcases(name, fn):
    r = discfix.rng(name)
    ks = [0] if fn not in ("pmf", "logpmf") else [0, 1]
    return [(region, (r.choice(ks), _bp(r, region))) for region in ("tiny", "core", "near1") for _ in range(N)]


def b_pcases(name, fn):
    r = discfix.rng(name)
    return [(region, (probs(r, region), _bp(r, pr))) for region in ("core", "lower", "upper")
            for pr in ("tiny", "core", "near1") for _ in range(N // 3)]


FIXTURES = {}
FIXTURES.update(discfix.make("binom", dict(dist=st.binom, logpmf=logpmf, cdfsf=cdfsf, support=support,
                                           xcases=xcases, pcases=pcases, ratio=ratio)))
FIXTURES.update(discfix.make("bernoulli", dict(dist=st.bernoulli, logpmf=b_logpmf, cdfsf=b_cdfsf,
                                               support=b_support, xcases=b_xcases, pcases=b_pcases)))
