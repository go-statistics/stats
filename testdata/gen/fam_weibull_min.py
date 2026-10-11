"""Weibull (minimum) distribution fixtures (scipy.stats.weibull_min,
standard form); arguments (x, c). Points are drawn as x = t^(1/c) for
t = x^c in each region."""

import math

import mpmath
from mpmath import mpf
import scipy.stats as st

from cfam import R, xs, probs, make, log1mexp

D = st.weibull_min
SHAPES = [0.3, 1.0, 2.5, 10.0, 50.0]


def _x(r, tlo, thi, log, n=250):
    out = []
    i = 0
    while len(out) < n:
        c = SHAPES[i % len(SHAPES)]
        i += 1
        t = xs(r, tlo, thi, 1, log=log)[0]
        lx = math.log(t) / c
        if not (-700 < lx < 700):
            continue
        out.append((math.exp(lx), c))
    return out


def _regions(name):
    r = R(name)
    return [("tiny", _x(r, 1e-300, 1e-3, True)), ("core", _x(r, 1e-3, 3, False)),
            ("tail", _x(r, 3, 700, True))]


def _p(name):
    r = R(name)
    return [(reg, [(p, SHAPES[i % len(SHAPES)]) for i, p in enumerate(ps)]) for reg, ps in probs(r)]


def _t(x, c):
    return mpf(x) ** mpf(c)


def _pdf(x, c):
    c = mpf(c)
    return c * mpf(x) ** (c - 1) * mpmath.exp(-_t(x, c))


FIXTURES = {
    "weibull_min_pdf": make(2, lambda: _regions("weibull_min_pdf"), _pdf, D.pdf, "pdf"),
    "weibull_min_logpdf": make(2, lambda: _regions("weibull_min_logpdf"), lambda x, c: mpmath.log(mpf(c)) + (mpf(c) - 1) * mpmath.log(mpf(x)) - _t(x, c), D.logpdf, "logpdf"),
    "weibull_min_cdf": make(2, lambda: _regions("weibull_min_cdf"), lambda x, c: -mpmath.expm1(-_t(x, c)), D.cdf, "cdf"),
    "weibull_min_sf": make(2, lambda: _regions("weibull_min_sf"), lambda x, c: mpmath.exp(-_t(x, c)), D.sf, "sf"),
    "weibull_min_logcdf": make(2, lambda: _regions("weibull_min_logcdf"), lambda x, c: log1mexp(_t(x, c)), D.logcdf, "logcdf"),
    "weibull_min_logsf": make(2, lambda: _regions("weibull_min_logsf"), lambda x, c: -_t(x, c), D.logsf, "logsf"),
    "weibull_min_ppf": make(2, lambda: _p("weibull_min_ppf"), lambda p, c: (-mpmath.log1p(-mpf(p))) ** (1 / mpf(c)), D.ppf, "ppf"),
    "weibull_min_isf": make(2, lambda: _p("weibull_min_isf"), lambda q, c: (-mpmath.log(mpf(q))) ** (1 / mpf(c)), D.isf, "isf"),
}
