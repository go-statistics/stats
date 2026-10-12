"""Discrete uniform fixtures (scipy.stats.randint). Rows are (k, low, high)
and (probability, low, high). Everything is an exact rational."""

import mpmath
import scipy.stats as st
from mpmath import mpf

import discfix
from discfix import ilogu, probs

N = 300


def logpmf(k, lo, hi):
    return -mpmath.log(mpf(int(hi) - int(lo)))


def cdfsf(k, lo, hi):
    lo, hi = int(lo), int(hi)
    d = mpf(hi - lo)
    return mpf(k - lo + 1) / d, mpf(hi - 1 - k) / d


def support(lo, hi):
    return int(lo), int(hi) - 1


def _shape(r, region):
    if region == "small":
        lo = r.randint(-50, 50)
        return lo, lo + r.randint(2, 100)
    lo = r.choice([0, -ilogu(r, 1, 1e15), ilogu(r, 1, 1e15)])
    return lo, lo + ilogu(r, 100, 2 ** 52)


def xcases(name, fn):
    r = discfix.rng(name)
    out = []
    for region in ("small", "large"):
        for _ in range(N):
            lo, hi = _shape(r, region)
            k = r.randint(lo, hi - 2) if r.random() < 0.7 else r.choice([lo, hi - 2, lo + 1])
            out.append((region, (k, lo, hi)))
    return out


def pcases(name, fn):
    r = discfix.rng(name)
    return [(region, (probs(r, region),) + _shape(r, sh)) for region in ("core", "lower", "upper")
            for sh in ("small", "large") for _ in range(N // 2)]


FIXTURES = discfix.make("randint", dict(dist=st.randint, logpmf=logpmf, cdfsf=cdfsf, support=support,
                                        xcases=xcases, pcases=pcases))
