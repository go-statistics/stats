"""Cauchy distribution fixtures (scipy.stats.cauchy, standard form)."""

import mpmath
from mpmath import mpf
import scipy.stats as st

from cfam import R, xs, probs, make

D = st.cauchy


def _pdf_regions(name):
    r = R(name)
    return [("core", xs(r, -1, 1)), ("mid", xs(r, 1, 1e3, log=True, sign=-1) + xs(r, 1, 1e3, log=True)),
            ("tail", xs(r, 1e3, 1e300, log=True, sign=-1) + xs(r, 1e3, 1e300, log=True)),
            ("tiny", xs(r, 1e-300, 1e-3, log=True))]


def _cdf_regions(name):
    r = R(name)
    return [("core", xs(r, -1, 1)), ("left", xs(r, 1, 1e300, log=True, sign=-1)),
            ("right", xs(r, 1, 1e15, log=True)), ("tiny", xs(r, 1e-300, 1e-3, log=True))]


def _pdf(x):
    x = mpf(x)
    return 1 / (mpmath.pi * (1 + x * x))


def _cdf(x):
    x = mpf(x)
    if x < 0:
        return mpmath.atan(-1 / x) / mpmath.pi
    return 1 - mpmath.atan(1 / x) / mpmath.pi if x != 0 else mpf(0.5)


def _sf(x):
    return _cdf(-mpf(x))


def _ppf(p):
    p = mpf(p)
    return -mpmath.cot(mpmath.pi * p)


def _isf(q):
    return mpmath.cot(mpmath.pi * mpf(q))


def _neg(regions):
    return [(reg, [-x for x in pts]) for reg, pts in regions]


FIXTURES = {
    "cauchy_pdf": make(1, lambda: _pdf_regions("cauchy_pdf"), _pdf, D.pdf, "pdf"),
    "cauchy_logpdf": make(1, lambda: _pdf_regions("cauchy_logpdf"), lambda x: mpmath.log(_pdf(x)), D.logpdf, "logpdf"),
    "cauchy_cdf": make(1, lambda: _cdf_regions("cauchy_cdf"), _cdf, D.cdf, "cdf"),
    "cauchy_sf": make(1, lambda: _neg(_cdf_regions("cauchy_sf")), _sf, D.sf, "sf"),
    "cauchy_logcdf": make(1, lambda: _cdf_regions("cauchy_logcdf"), lambda x: mpmath.log(_cdf(x)), D.logcdf, "logcdf"),
    "cauchy_logsf": make(1, lambda: _neg(_cdf_regions("cauchy_logsf")), lambda x: mpmath.log(_sf(x)), D.logsf, "logsf"),
    "cauchy_ppf": make(1, lambda: probs(R("cauchy_ppf")), _ppf, D.ppf, "ppf"),
    "cauchy_isf": make(1, lambda: probs(R("cauchy_isf")), _isf, D.isf, "isf"),
}
