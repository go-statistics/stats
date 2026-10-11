"""Generic builder of the eight ULP fixtures of a continuous distribution.

A family is described by:
  name      SciPy name (fixture names are <name>_<fn>)
  dist      the scipy.stats distribution
  shapes    {region: function(random) -> tuple of shape floats}
  pdf, cdf, sf
            mpmath functions of (*shapes, x) with float arguments
  support   function(*shapes) -> (lo, hi) as mpf (may be +-inf)

The points x come from SciPy's own quantiles at log-uniform probabilities
(lower tail, bulk, upper tail via isf), so that every region sees the
bulk and both tails of the distribution at every shape regime. The
quantile fixtures sample the probability instead (lower: 1e-300..1e-3,
core, upper: 1 - 1e-16..1 - 1e-3).
"""

import math

import mpmath
from mpmath import mpf

import common
from dists import invert

NX = 60  # points per (shape region, side); fam["counts"] may lower it per region


def _sides(r, n):
    return [("lo", common.loguniform(r, 1e-300, 1e-3, n)),
            ("mid", common.uniform(r, 1e-3, 1 - 1e-3, n)),
            ("hi", common.loguniform(r, 1e-300, 1e-3, n))]


def xcases(fam, fname):
    r = common.rng(fam["name"] + "_" + fname)
    out = []
    import numpy as np
    with np.errstate(all="ignore"):
        for reg, shp in fam["shapes"].items():
            for side, ps in _sides(r, fam.get("counts", {}).get(reg, NX)):
                for p in ps:
                    sh = shp(r)
                    if side == "hi":
                        x = float(fam["dist"].isf(p, *sh))
                    else:
                        x = float(fam["dist"].ppf(p, *sh))
                    if not math.isfinite(x):
                        continue
                    lo, hi = fam["support"](*sh)
                    if not (lo < x < hi):
                        continue
                    out.append(("%s_%s" % (reg, side), tuple(sh) + (x,)))
    return out


def pcases(fam, fname):
    r = common.rng(fam["name"] + "_" + fname)
    out = []
    for reg, shp in fam["shapes"].items():
        n = fam.get("counts", {}).get(reg, NX)
        for p in common.loguniform(r, 1e-300, 1e-3, n):
            out.append(("%s_lower" % reg, tuple(shp(r)) + (p,)))
        for p in common.uniform(r, 1e-3, 1 - 1e-3, n):
            out.append(("%s_core" % reg, tuple(shp(r)) + (p,)))
        for q in common.loguniform(r, 1.2e-16, 1e-3, n):
            out.append(("%s_upper" % reg, tuple(shp(r)) + (1.0 - q,)))
    return out


def build(fam):
    name, dist = fam["name"], fam["dist"]
    pdf, cdf, sf, support = fam["pdf"], fam["cdf"], fam["sf"], fam["support"]
    nsh = len(next(iter(fam["shapes"].values()))(common.rng("probe")))
    nargs = nsh + 1

    def logpdf(*a):
        return mpmath.log(pdf(*a))

    def logcdf(*a):
        c = cdf(*a)
        return mpmath.log1p(-sf(*a)) if c > 0.5 else mpmath.log(c)

    def logsf(*a):
        s = sf(*a)
        return mpmath.log1p(-cdf(*a)) if s > 0.5 else mpmath.log(s)

    def ppf(*a):
        sh, p = a[:-1], a[-1]
        lo, hi = support(*sh)
        x0 = float(dist.ppf(p, *sh))
        return invert(lambda x: mpmath.log(cdf(*sh, x)), lambda x: pdf(*sh, x) / cdf(*sh, x),
                      p, x0, lo, hi)

    def isf(*a):
        sh, q = a[:-1], a[-1]
        lo, hi = support(*sh)
        x0 = float(dist.isf(q, *sh))
        return invert(lambda x: mpmath.log(sf(*sh, x)), lambda x: -pdf(*sh, x) / sf(*sh, x),
                      q, x0, lo, hi, decreasing=True)

    fx = {}
    for fn, mp, sp, cases in [
        ("pdf", pdf, dist.pdf, xcases), ("logpdf", logpdf, dist.logpdf, xcases),
        ("cdf", cdf, dist.cdf, xcases), ("sf", sf, dist.sf, xcases),
        ("logcdf", logcdf, dist.logcdf, xcases), ("logsf", logsf, dist.logsf, xcases),
        ("ppf", ppf, dist.ppf, pcases), ("isf", isf, dist.isf, pcases)]:
        def builder(fn=fn, mp=mp, sp=sp, cases=cases):
            spf = (lambda *a: sp(a[-1], *a[:-1]))
            return nargs, cases(fam, fn), mp, spf, "%s %s(shapes..., x)" % (name, fn)
        fx["%s_%s" % (name, fn)] = builder
    return fx
