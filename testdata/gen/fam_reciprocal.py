"""Reciprocal (log-uniform) distribution fixtures (scipy.stats.reciprocal,
standard form); arguments (x, a, b)."""

import mpmath
from mpmath import mpf
import scipy.stats as st

from cfam import R, xs, make

D = st.reciprocal
SHAPES = [(1e-3, 1e3), (0.5, 2.0), (1.0, 1.0 + 2 ** -20), (1e-200, 1e200), (2.0, 3.0), (1.0, 1e10),
          (0.1, 0.1000001)]


def _x(r, region, n=250):
    out = []
    for i in range(n):
        a, b = SHAPES[i % len(SHAPES)]
        if region == "near_a":
            u = xs(r, 1e-15, 1e-2, 1, log=True)[0]
            x = a + (b - a) * u
        elif region == "near_b":
            u = xs(r, 1e-15, 1e-2, 1, log=True)[0]
            x = b - (b - a) * u
        else:
            la, lb = mpmath.log(a), mpmath.log(b)
            x = float(mpmath.exp(la + (lb - la) * r.random()))
        x = min(max(x, a), b)
        out.append((x, a, b))
    return out


def _regions(name):
    r = R(name)
    return [(reg, _x(r, reg)) for reg in ("near_a", "core", "near_b")]


def _p(r, n=250):
    out = []
    for reg, lo, hi in (("lower", 1e-300, 1e-3), ("core", None, None), ("upper", 1.2e-16, 1e-3)):
        pts = []
        for i in range(n):
            a, b = SHAPES[i % len(SHAPES)]
            if reg == "core":
                p = r.uniform(1e-3, 1 - 1e-3)
            else:
                p = xs(r, lo, hi, 1, log=True)[0]
                if reg == "upper":
                    p = 1.0 - p
            pts.append((p, a, b))
        out.append((reg, pts))
    return out


def L(a, b):
    return mpmath.log(mpf(b) / mpf(a))


FIXTURES = {
    "reciprocal_pdf": make(3, lambda: _regions("reciprocal_pdf"), lambda x, a, b: 1 / (mpf(x) * L(a, b)), D.pdf, "pdf"),
    "reciprocal_logpdf": make(3, lambda: _regions("reciprocal_logpdf"), lambda x, a, b: -mpmath.log(mpf(x) * L(a, b)), D.logpdf, "logpdf"),
    "reciprocal_cdf": make(3, lambda: _regions("reciprocal_cdf"), lambda x, a, b: L(a, x) / L(a, b), D.cdf, "cdf"),
    "reciprocal_sf": make(3, lambda: _regions("reciprocal_sf"), lambda x, a, b: L(x, b) / L(a, b), D.sf, "sf"),
    "reciprocal_logcdf": make(3, lambda: _regions("reciprocal_logcdf"), lambda x, a, b: mpmath.log(L(a, x) / L(a, b)), D.logcdf, "logcdf"),
    "reciprocal_logsf": make(3, lambda: _regions("reciprocal_logsf"), lambda x, a, b: mpmath.log(L(x, b) / L(a, b)), D.logsf, "logsf"),
    "reciprocal_ppf": make(3, lambda: _p(R("reciprocal_ppf")), lambda p, a, b: mpf(a) * mpmath.exp(mpf(p) * L(a, b)), D.ppf, "ppf"),
    "reciprocal_isf": make(3, lambda: _p(R("reciprocal_isf")), lambda q, a, b: mpf(b) * mpmath.exp(-mpf(q) * L(a, b)), D.isf, "isf"),
}
