"""Pareto distribution fixtures (scipy.stats.pareto, standard form);
arguments (x, b). Points are drawn as x = exp(t/b) for t = b log x."""

import math

import mpmath
from mpmath import mpf
import scipy.stats as st

from cfam import R, xs, probs, make, log1mexp

D = st.pareto
SHAPES = [0.1, 1.0, 2.5, 10.0, 300.0]


def _x(r, tlo, thi, log, n=250):
    out = []
    i = 0
    while len(out) < n:
        b = SHAPES[i % len(SHAPES)]
        i += 1
        t = xs(r, tlo, thi, 1, log=log)[0]
        lx = t / b
        if lx > 700:
            continue
        x = math.exp(lx) if lx > 1e-3 else 1 + math.expm1(lx)
        if x <= 1:
            continue
        out.append((x, b))
    return out


def _regions(name):
    r = R(name)
    return [("near_one", _x(r, 1e-14, 1e-3, True)), ("core", _x(r, 1e-3, 3, False)),
            ("tail", _x(r, 3, 700, True))]


def _p(name):
    r = R(name)
    return [(reg, [(p, SHAPES[i % len(SHAPES)]) for i, p in enumerate(ps)]) for reg, ps in probs(r)]


def _t(x, b):
    return mpf(b) * mpmath.log(mpf(x))


FIXTURES = {
    "pareto_pdf": make(2, lambda: _regions("pareto_pdf"), lambda x, b: mpf(b) * mpf(x) ** (-mpf(b) - 1), D.pdf, "pdf"),
    "pareto_logpdf": make(2, lambda: _regions("pareto_logpdf"), lambda x, b: mpmath.log(mpf(b)) - (mpf(b) + 1) * mpmath.log(mpf(x)), D.logpdf, "logpdf"),
    "pareto_cdf": make(2, lambda: _regions("pareto_cdf"), lambda x, b: -mpmath.expm1(-_t(x, b)), D.cdf, "cdf"),
    "pareto_sf": make(2, lambda: _regions("pareto_sf"), lambda x, b: mpmath.exp(-_t(x, b)), D.sf, "sf"),
    "pareto_logcdf": make(2, lambda: _regions("pareto_logcdf"), lambda x, b: log1mexp(_t(x, b)), D.logcdf, "logcdf"),
    "pareto_logsf": make(2, lambda: _regions("pareto_logsf"), lambda x, b: -_t(x, b), D.logsf, "logsf"),
    "pareto_ppf": make(2, lambda: _p("pareto_ppf"), lambda p, b: mpmath.exp(-mpmath.log1p(-mpf(p)) / mpf(b)), D.ppf, "ppf"),
    "pareto_isf": make(2, lambda: _p("pareto_isf"), lambda q, b: mpmath.exp(-mpmath.log(mpf(q)) / mpf(b)), D.isf, "isf"),
}
