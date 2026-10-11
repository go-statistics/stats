"""Poisson fixtures (scipy.stats.poisson). Rows are (k, mu) and
(probability, mu).

The exact mass is mu^k e^-mu / k! from log-gamma at working precision; cdf
and sf are Q(k+1, mu) and P(k+1, mu) from mpincomplete.gam_P_Q (series, or
quadrature for large parameters). Regions: mu from 1e-300 to 1e15, the bulk,
the tails to 37 standard deviations and, for the logarithms, beyond the
underflow of the probabilities.
"""

import math

import mpmath
import scipy.stats as st
from mpmath import mpf

import discfix
from discfix import ilogu, logu, probs
from mpincomplete import gam_P_Q

N = 250
NQ = 120


def logpmf(k, mu):
    mu = mpf(mu)
    return k * mpmath.log(mu) - mu - mpmath.loggamma(k + 1)


def cdfsf(k, mu):
    p, q = gam_P_Q(mpf(k + 1), mpf(mu))
    return q, p


def support(mu):
    return 0, math.inf


def _mu(r, region):
    return {
        "tiny_mu": lambda: logu(r, 1e-300, 1e-3),
        "small_mu": lambda: logu(r, 1e-3, 10),
        "large_mu": lambda: logu(r, 1e6, 1e15),
    }.get(region, lambda: logu(r, 10, 1e6))()


def _k(mu, z):
    return int(max(0, round(mu + z * math.sqrt(mu))))


def xcases(name, fn):
    r = discfix.rng(name)
    cdfish = fn not in ("pmf", "logpmf")
    out = []
    for region, count, zlo, zhi, sym in (("small_mu", N, 0, 4, True), ("bulk", N, 0, 5, True),
                                          ("large_mu", NQ, 0, 6, True), ("tiny_mu", N, 0, 0, True)):
        for _ in range(count):
            mu = _mu(r, region)
            if region == "tiny_mu":
                k = r.randint(0, 4)
            elif region == "small_mu":
                k = r.randint(0, int(mu + 6 * math.sqrt(mu)) + 3)
            else:
                k = _k(mu, r.uniform(zlo, zhi) * r.choice([-1, 1]))
            out.append((region, (k, mu)))
    if cdfish:
        for region, sgn in (("lower", -1), ("upper", 1)):
            for _ in range(N):
                mu = logu(r, 20, 1e9)
                k = _k(mu, sgn * r.uniform(3, 37))
                out.append((region, (k, mu)))
        if fn in ("logcdf", "logsf"):
            for _ in range(N):
                mu = logu(r, 1, 1e4)
                sgn = r.choice([-1, 1])
                k = r.randint(0, int(mu)) if sgn < 0 else int(mu + r.uniform(40, 200) * math.sqrt(mu) + 50)
                out.append(("deep", (k, mu)))
    else:
        for _ in range(N):
            mu = logu(r, 10, 1e9)
            out.append(("tail", (_k(mu, r.choice([-1, 1]) * r.uniform(6, 35)), mu)))
        if fn == "logpmf":
            for _ in range(N):
                mu = logu(r, 1, 1e15)
                out.append(("deep", (ilogu(r, 1, 1e16), mu)))
    return out


def pcases(name, fn):
    r = discfix.rng(name)
    out = []
    for region in ("core", "lower", "upper"):
        for _ in range(N):
            out.append((region, (probs(r, region), logu(r, 1e-3, 1e6))))
    for _ in range(NQ):
        out.append(("large_mu", (probs(r, r.choice(["core", "lower", "upper"])), logu(r, 1e6, 1e15))))
    return out


FIXTURES = discfix.make("poisson", dict(dist=st.poisson, logpmf=logpmf, cdfsf=cdfsf, support=support,
                                        xcases=xcases, pcases=pcases))
