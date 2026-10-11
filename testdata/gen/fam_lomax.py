"""Lomax distribution fixtures (scipy.stats.lomax, standard form);
arguments (x, c). Points are drawn as x = expm1(t/c) for t = c log1p(x)."""

import math

import mpmath
from mpmath import mpf
import scipy.stats as st

from cfam import R, xs, probs, make, log1mexp

D = st.lomax
SHAPES = [0.1, 1.0, 2.5, 10.0, 300.0]


def _x(r, tlo, thi, log, n=250):
    out = []
    i = 0
    while len(out) < n:
        c = SHAPES[i % len(SHAPES)]
        i += 1
        t = xs(r, tlo, thi, 1, log=log)[0]
        lx = t / c
        if lx > 700:
            continue
        x = math.expm1(lx)
        if x <= 0:
            continue
        out.append((x, c))
    return out


def _regions(name):
    r = R(name)
    return [("tiny", _x(r, 1e-300, 1e-3, True)), ("core", _x(r, 1e-3, 3, False)),
            ("tail", _x(r, 3, 700, True))]


def _p(name):
    r = R(name)
    return [(reg, [(p, SHAPES[i % len(SHAPES)]) for i, p in enumerate(ps)]) for reg, ps in probs(r)]


def _t(x, c):
    return mpf(c) * mpmath.log1p(mpf(x))


FIXTURES = {
    "lomax_pdf": make(2, lambda: _regions("lomax_pdf"), lambda x, c: mpf(c) * mpmath.exp(-(mpf(c) + 1) * mpmath.log1p(mpf(x))), D.pdf, "pdf"),
    "lomax_logpdf": make(2, lambda: _regions("lomax_logpdf"), lambda x, c: mpmath.log(mpf(c)) - (mpf(c) + 1) * mpmath.log1p(mpf(x)), D.logpdf, "logpdf"),
    "lomax_cdf": make(2, lambda: _regions("lomax_cdf"), lambda x, c: -mpmath.expm1(-_t(x, c)), D.cdf, "cdf"),
    "lomax_sf": make(2, lambda: _regions("lomax_sf"), lambda x, c: mpmath.exp(-_t(x, c)), D.sf, "sf"),
    "lomax_logcdf": make(2, lambda: _regions("lomax_logcdf"), lambda x, c: log1mexp(_t(x, c)), D.logcdf, "logcdf"),
    "lomax_logsf": make(2, lambda: _regions("lomax_logsf"), lambda x, c: -_t(x, c), D.logsf, "logsf"),
    "lomax_ppf": make(2, lambda: _p("lomax_ppf"), lambda p, c: mpmath.expm1(-mpmath.log1p(-mpf(p)) / mpf(c)), D.ppf, "ppf"),
    "lomax_isf": make(2, lambda: _p("lomax_isf"), lambda q, c: mpmath.expm1(-mpmath.log(mpf(q)) / mpf(c)), D.isf, "isf"),
}
