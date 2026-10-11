"""Log-normal distribution fixtures (scipy.stats.lognorm, standard form);
arguments (x, s). Points are drawn as x = exp(s z) for z in each region."""

import math

import mpmath
from mpmath import mpf
import scipy.stats as st

from cfam import R, probs, make

D = st.lognorm
SHAPES = [0.01, 0.25, 1.0, 2.5, 10.0]


def _x(r, zlo, zhi, n=250):
    out = []
    for i in range(n):
        s = SHAPES[i % len(SHAPES)]
        lo, hi = max(zlo, -700 / s), min(zhi, 700 / s)
        if lo >= hi:
            lo, hi = zlo, zhi
            lo, hi = max(lo, -700 / s), min(hi, 700 / s)
        z = r.uniform(lo, hi) if lo < hi else lo
        out.append((math.exp(s * z), s))
    return out


def _sym(r, a, b, n=250):
    """|z| in [a, b], both signs."""
    out = []
    for i in range(n):
        s = SHAPES[i % len(SHAPES)]
        z = r.uniform(a, min(b, 700 / s)) * (1 if r.random() < 0.5 else -1)
        out.append((math.exp(s * z), s))
    return out


def _pdf_regions(name):
    r = R(name)
    return [("core", _x(r, -3, 3)), ("tail", _sym(r, 3, 37.5))]


def _cdf_regions(name):
    r = R(name)
    return [("core", _x(r, -3, 3)), ("left", _x(r, -37, -3)), ("right", _x(r, 3, 8))]


def _sf_regions(name):
    r = R(name)
    return [("core", _x(r, -3, 3)), ("right", _x(r, 3, 37)), ("left", _x(r, -8, -3))]


def _log_regions(name, sign):
    r = R(name)
    base = _cdf_regions(name) if sign < 0 else _sf_regions(name)
    far = [(math.exp(s * z * sign), s) for (x, s) in _x(r, 38, 1e4) for z in [math.log(x) / s]]
    return base + [("far", far)]


def _p(name):
    r = R(name)
    out = []
    for reg, ps in probs(r):
        out.append((reg, [(p, SHAPES[i % len(SHAPES)]) for i, p in enumerate(ps)]))
    return out


def _z(x, s):
    return mpmath.log(mpf(x)) / mpf(s)


def _pdf(x, s):
    z = _z(x, s)
    return mpmath.exp(-z * z / 2) / (mpf(s) * mpf(x) * mpmath.sqrt(2 * mpmath.pi))


def _cdf(x, s):
    return mpmath.erfc(-_z(x, s) / mpmath.sqrt(2)) / 2


def _sf(x, s):
    return mpmath.erfc(_z(x, s) / mpmath.sqrt(2)) / 2


def _ndtri(p):
    """Normal quantile: Newton on erfc in mpmath from a float start."""
    p = mpf(p)
    z = mpf(float(st.norm.ppf(float(p))))
    for _ in range(100):
        f = mpmath.erfc(-z / mpmath.sqrt(2)) / 2 - p
        d = mpmath.exp(-z * z / 2) / mpmath.sqrt(2 * mpmath.pi)
        nz = z - f / d
        if abs(nz - z) <= abs(z) * mpf(2) ** (-mpmath.mp.prec + 4) + mpf(2) ** -1000:
            return nz
        z = nz
    return z


def _ppf(p, s):
    return mpmath.exp(mpf(s) * _ndtri(p))


def _isf(q, s):
    return mpmath.exp(-mpf(s) * _ndtri(q))


FIXTURES = {
    "lognorm_pdf": make(2, lambda: _pdf_regions("lognorm_pdf"), _pdf, D.pdf, "pdf"),
    "lognorm_logpdf": make(2, lambda: _pdf_regions("lognorm_logpdf"), lambda x, s: mpmath.log(_pdf(x, s)), D.logpdf, "logpdf"),
    "lognorm_cdf": make(2, lambda: _cdf_regions("lognorm_cdf"), _cdf, D.cdf, "cdf"),
    "lognorm_sf": make(2, lambda: _sf_regions("lognorm_sf"), _sf, D.sf, "sf"),
    "lognorm_logcdf": make(2, lambda: _log_regions("lognorm_logcdf", -1), lambda x, s: mpmath.log(_cdf(x, s)), D.logcdf, "logcdf"),
    "lognorm_logsf": make(2, lambda: _log_regions("lognorm_logsf", 1), lambda x, s: mpmath.log(_sf(x, s)), D.logsf, "logsf"),
    "lognorm_ppf": make(2, lambda: _p("lognorm_ppf"), _ppf, D.ppf, "ppf"),
    "lognorm_isf": make(2, lambda: _p("lognorm_isf"), _isf, D.isf, "isf"),
}
