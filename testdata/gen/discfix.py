"""Shared machinery of the discrete distribution fixtures (fam_binom.py,
fam_poisson.py, ...).

A distribution is described by its exact log-mass and its exact (cdf, sf)
pair in mpmath, its support, and samplers for the arguments. make() turns
that into the eight fixtures <name>_{pmf,logpmf,cdf,sf,logcdf,logsf,ppf,isf}
whose rows are (k, shapes...) or (probability, shapes...).

The quantile oracle follows the definition this package and SciPy share,
applied to the correctly rounded distribution function: ppf(p) is the
smallest k in the support with RN(cdf(k)) >= p, isf(q) the smallest k with
RN(sf(k)) <= q, RN the rounding to the nearest float64 (mpmath's float()).
It starts from SciPy's answer when SciPy has one and searches by doubling
steps then bisection on exact evaluations.
"""

import math

import mpmath
from mpmath import mpf

import common

FNS = ("pmf", "logpmf", "cdf", "sf", "logcdf", "logsf", "ppf", "isf")


def smallest_k(ok, k0, lo, hi):
    """Smallest integer k in [lo, hi] with ok(k) (monotone false..true);
    hi when none below it. k0 is a starting guess."""
    if k0 is None or not math.isfinite(k0):
        k0 = lo
    k = int(min(max(k0, lo), hi))
    if ok(k):
        top, step = k, 1
        while True:
            b = max(lo, top - step)
            if b == top:
                return lo
            if not ok(b):
                return _bisect(b, top, ok)
            top, step = b, step * 2
    bot, step = k, 1
    while True:
        t = min(hi, bot + step)
        if t == hi or ok(t):
            return _bisect(bot, t, ok)
        bot, step = t, step * 2


def _bisect(bad, good, ok):
    while good - bad > 1:
        mid = (bad + good) // 2
        if ok(mid):
            good = mid
        else:
            bad = mid
    return good


def _logpair(v, w):
    """log v for a probability v whose complement w is also exact: through
    log1p(-w) when v > 1/2, where v itself may have been rounded at the
    working precision (1 - 1e-300)."""
    return mpmath.log1p(-w) if v > 0.5 else mpmath.log(v)


def _scipy(f, p, sh):
    """SciPy's quantile as a starting guess, None when it has none."""
    import numpy as np
    try:
        with np.errstate(all="ignore"):
            return float(f(p, *sh))
    except Exception:
        return None


def make(name, d):
    """d: dict with
         dist      the scipy.stats distribution
         logpmf    (k, *shapes) -> mpf, exact
         cdfsf     (k, *shapes) -> (cdf, sf) as mpf, exact
         support   (*shapes) -> (lo, hi) ints (hi may be math.inf)
         xcases    (fixture name, fn) -> [(region, (k, *shapes))]
         pcases    (fixture name, fn) -> [(region, (p, *shapes))]
         spfns     optional {fn: callable} replacing SciPy's method for the
                   SciPy column (and the quantile oracle's starting guess),
                   e.g. to skip arguments where SciPy cannot finish
    """
    st = d["dist"]
    spf = {fn: d.get("spfns", {}).get(fn, getattr(st, fn)) for fn in FNS}

    def logpmf(k, *sh):
        return d["logpmf"](int(k), *sh)

    def ppf(p, *sh):
        lo, hi = d["support"](*sh)

        def ok(k):
            return k >= hi or float(d["cdfsf"](k, *sh)[0]) >= p
        return mpf(smallest_k(ok, _scipy(spf["ppf"], p, sh), lo, hi))

    def isf(q, *sh):
        lo, hi = d["support"](*sh)

        def ok(k):
            return k >= hi or float(d["cdfsf"](k, *sh)[1]) <= q
        return mpf(smallest_k(ok, _scipy(spf["isf"], q, sh), lo, hi))

    def cdf(k, *sh):
        return d["cdfsf"](int(k), *sh)[0]

    def sf(k, *sh):
        return d["cdfsf"](int(k), *sh)[1]

    mp_of = {
        "pmf": lambda *a: mpmath.exp(logpmf(*a)),
        "logpmf": logpmf,
        "cdf": cdf,
        "sf": sf,
        "logcdf": lambda *a: _logpair(*d["cdfsf"](int(a[0]), *a[1:])),
        "logsf": lambda *a: _logpair(*reversed(d["cdfsf"](int(a[0]), *a[1:]))),
        "ppf": ppf,
        "isf": isf,
    }
    fixtures = {}
    for fn in FNS:
        def builder(fn=fn):
            fname = "%s_%s" % (name, fn)
            cases = (d["pcases"] if fn in ("ppf", "isf") else d["xcases"])(fname, fn)
            nargs = len(cases[0][1])
            return nargs, cases, mp_of[fn], spf[fn], "%s %s" % (name, fn)
        fixtures["%s_%s" % (name, fn)] = builder
    return fixtures


# ---------------------------------------------------------------- sampling

def rng(name):
    return common.rng(name)


def logu(r, lo, hi):
    return math.exp(r.uniform(math.log(lo), math.log(hi)))


def ilogu(r, lo, hi):
    return int(round(logu(r, lo, hi)))


def probs(r, region):
    """A float probability for a quantile region."""
    if region == "lower":
        return logu(r, 1e-300, 1e-3)
    if region == "upper":
        return 1.0 - logu(r, 1.2e-16, 1e-3)
    return r.uniform(1e-3, 1 - 1e-3)
