"""Hypergeometric fixtures (scipy.stats.hypergeom). Rows are (k, M, n, N)
and (probability, M, n, N): M items, n good, N drawn.

The exact mass is C(n,k) C(M-n,N-k) / C(M,N) from log-gamma at working
precision; cdf and sf sum the masses of the tail on k's side of the mean
by the exact ratio of successive masses (the other one is the
complement). Regions: small and moderate populations, populations to 1e8,
the tails and, for the logarithms, beyond the underflow.
"""

import mpmath
import scipy.stats as st
from mpmath import mp, mpf

import discfix
from discfix import ilogu, probs

N_ = 250
NQ = 100


def _lbin(n, k):
    return mpmath.loggamma(n + 1) - mpmath.loggamma(k + 1) - mpmath.loggamma(n - k + 1)


def logpmf(k, M, n, N):
    M, n, N = mpf(int(M)), mpf(int(n)), mpf(int(N))
    return _lbin(n, k) + _lbin(M - n, N - k) - _lbin(M, N)


def support(M, n, N):
    M, n, N = int(M), int(n), int(N)
    return max(0, N - (M - n)), min(n, N)


def _tail(k, M, n, N, lower):
    """sum of the masses j <= k (lower) or j > k."""
    M, n, N = int(M), int(n), int(N)
    lo, hi = support(M, n, N)
    eps = mpf(2) ** (-mp.prec - 16)
    if lower:
        j = k
        t = mpmath.exp(logpmf(j, M, n, N))
        s = t
        while j > lo:
            t *= mpf(j * (M - n - N + j)) / ((n - j + 1) * (N - j + 1))
            j -= 1
            s += t
            if t < eps * s:
                break
        return s
    j = k + 1
    if j > hi:
        return mpf(0)
    t = mpmath.exp(logpmf(j, M, n, N))
    s = t
    while j < hi:
        t *= mpf((n - j) * (N - j)) / ((j + 1) * (M - n - N + j + 1))
        j += 1
        s += t
        if t < eps * s:
            break
    return s


def cdfsf(k, M, n, N):
    lower = k < int(n) * int(N) / int(M)
    v = _tail(k, M, n, N, lower)
    if v > 0.5:
        lower = not lower
        v = _tail(k, M, n, N, lower)
    return (v, 1 - v) if lower else (1 - v, v)


def _shape(r, region):
    if region == "small":
        M = r.randint(2, 100)
    elif region == "large":
        M = ilogu(r, 1e6, 1e8)
    else:
        M = ilogu(r, 100, 1e6)
    n = r.randint(1, M - 1)
    N = r.randint(1, M - 1)
    if region == "skewed":
        n = max(1, int(M * r.uniform(1e-4, 1e-2)))
    return M, n, N


def _k(M, n, N, z):
    lo, hi = support(M, n, N)
    m = n * N / M
    v = n * N * (M - n) * (M - N) / (M * M * (M - 1)) if M > 1 else 0
    return int(min(hi, max(lo, round(m + z * v ** 0.5))))


def xcases(name, fn):
    r = discfix.rng(name)
    cdfish = fn not in ("pmf", "logpmf")
    out = []
    for region, count, zr in (("small", N_, 4), ("bulk", N_, 5), ("skewed", N_, 5), ("large", NQ, 6)):
        for _ in range(count):
            M, n, N = _shape(r, region)
            lo, hi = support(M, n, N)
            if cdfish and lo == hi:
                continue
            k = _k(M, n, N, r.uniform(-zr, zr))
            if cdfish and k >= hi:
                k = hi - 1
            out.append((region, (k, M, n, N)))
    tails = [("lower", -1), ("upper", 1)] if cdfish else [("tail", 0)]
    for region, sgn in tails:
        for _ in range(N_):
            M, n, N = _shape(r, "bulk")
            lo, hi = support(M, n, N)
            if lo == hi:
                continue
            z = (sgn or r.choice([-1, 1])) * r.uniform(3 if cdfish else 6, 37)
            k = _k(M, n, N, z)
            if cdfish and k >= hi:
                k = hi - 1
            out.append((region, (k, M, n, N)))
    if fn in ("logcdf", "logsf", "logpmf"):
        for _ in range(N_):
            M, n, N = _shape(r, "bulk")
            lo, hi = support(M, n, N)
            if lo == hi:
                continue
            k = r.randint(lo, hi - 1)
            out.append(("deep", (k, M, n, N)))
    return out


def pcases(name, fn):
    r = discfix.rng(name)
    out = []
    for region in ("core", "lower", "upper"):
        for _ in range(N_):
            M, n, N = _shape(r, r.choice(["small", "bulk", "skewed"]))
            out.append((region, (probs(r, region), M, n, N)))
    for _ in range(NQ):
        out.append(("large", (probs(r, "core"),) + _shape(r, "large")))
    return out


FIXTURES = discfix.make("hypergeom", dict(dist=st.hypergeom, logpmf=logpmf, cdfsf=cdfsf, support=support,
                                          xcases=xcases, pcases=pcases))
