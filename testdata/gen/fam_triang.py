"""Triangular distribution fixtures (scipy.stats.triang, standard form);
arguments (x, c)."""

import mpmath
from mpmath import mpf
import scipy.stats as st

from cfam import R, xs, probs, make

D = st.triang
SHAPES = [0.0, 0.3, 0.5, 0.999, 1.0, 1e-6]


def _x(r, region, n=250):
    out = []
    for i in range(n):
        c = SHAPES[i % len(SHAPES)]
        if region == "near_zero":
            x = xs(r, 1e-300, 1e-3, 1, log=True)[0]
        elif region == "near_one":
            x = 1.0 - xs(r, 1e-16, 1e-3, 1, log=True)[0]
        elif region == "near_mode":
            x = c + (r.random() - 0.5) * 1e-3
            if not (0 < x < 1):
                x = 0.5
        else:
            x = r.uniform(1e-3, 1 - 1e-3)
        out.append((x, c))
    return out


def _regions(name):
    r = R(name)
    return [(reg, _x(r, reg)) for reg in ("near_zero", "core", "near_mode", "near_one")]


def _p(name):
    r = R(name)
    return [(reg, [(p, SHAPES[i % len(SHAPES)]) for i, p in enumerate(ps)]) for reg, ps in probs(r)]


def exact(f):
    """The functions are algebraic in float64 arguments: evaluate them at
    4000 bits, so that 1 - (1-x)² for x = 1e-300 is exact."""
    def g(*a):
        with mpmath.workprec(4000):
            v = f(*a)
        return +v
    return g


@exact
def _pdf(x, c):
    x, c = mpf(x), mpf(c)
    if x < c:
        return 2 * x / c
    if c == 1:
        return 2 * x
    return 2 * (1 - x) / (1 - c)


@exact
def _cdf(x, c):
    x, c = mpf(x), mpf(c)
    if x < c:
        return x * x / c
    return 1 - (1 - x) ** 2 / (1 - c)


@exact
def _sf(x, c):
    x, c = mpf(x), mpf(c)
    if x < c:
        return 1 - x * x / c
    return (1 - x) ** 2 / (1 - c)


@exact
def _ppf(p, c):
    p, c = mpf(p), mpf(c)
    if p < c:
        return mpmath.sqrt(c * p)
    return 1 - mpmath.sqrt((1 - c) * (1 - p))


@exact
def _isf(q, c):
    q, c = mpf(q), mpf(c)
    if 1 - q < c:
        return mpmath.sqrt(c * (1 - q))
    return 1 - mpmath.sqrt((1 - c) * q)


def _log(f):
    @exact
    def g(x, c):
        v = f(x, c)
        return mpmath.log(v) if v > 0 else mpmath.ninf
    return g


FIXTURES = {
    "triang_pdf": make(2, lambda: _regions("triang_pdf"), _pdf, D.pdf, "pdf"),
    "triang_logpdf": make(2, lambda: _regions("triang_logpdf"), _log(_pdf), D.logpdf, "logpdf"),
    "triang_cdf": make(2, lambda: _regions("triang_cdf"), _cdf, D.cdf, "cdf"),
    "triang_sf": make(2, lambda: _regions("triang_sf"), _sf, D.sf, "sf"),
    "triang_logcdf": make(2, lambda: _regions("triang_logcdf"), _log(_cdf), D.logcdf, "logcdf"),
    "triang_logsf": make(2, lambda: _regions("triang_logsf"), _log(_sf), D.logsf, "logsf"),
    "triang_ppf": make(2, lambda: _p("triang_ppf"), _ppf, D.ppf, "ppf"),
    "triang_isf": make(2, lambda: _p("triang_isf"), _isf, D.isf, "isf"),
}
