"""Laplace distribution fixtures (scipy.stats.laplace, standard form)."""

import mpmath
from mpmath import mpf
import scipy.stats as st

from cfam import R, xs, probs, make

D = st.laplace


def _pm(r, lo, hi, n=125, log=False):
    return xs(r, lo, hi, n, log=log) + [-x for x in xs(r, lo, hi, n, log=log)]


def _pdf_regions(name):
    r = R(name)
    return [("core", xs(r, -1, 1)), ("mid", _pm(r, 1, 50)), ("tail", _pm(r, 50, 745)),
            ("tiny", _pm(r, 1e-300, 1e-3, log=True))]


def _cdf_regions(name, sign=1):
    r = R(name)
    reg = [("core", xs(r, -1, 1)), ("left", xs(r, 1, 745)), ("right", xs(r, 1, 40)),
           ("far_right", xs(r, 40, 745)), ("tiny", _pm(r, 1e-300, 1e-3, log=True))]
    out = []
    for name, pts in reg:
        s = -1 if name == "left" else 1
        out.append((name, [sign * s * x for x in pts]))
    return out


def _cdf(x):
    x = mpf(x)
    if x < 0:
        return mpmath.exp(x) / 2
    return 1 - mpmath.exp(-x) / 2


def _logcdf(x):
    x = mpf(x)
    if x < 0:
        return x - mpmath.log(2)
    return mpmath.log1p(-mpmath.exp(-x) / 2)


def _ppf(p):
    p = mpf(p)
    if p <= 0.5:
        return mpmath.log(2 * p)
    return -mpmath.log(2 * (1 - p))


FIXTURES = {
    "laplace_pdf": make(1, lambda: _pdf_regions("laplace_pdf"), lambda x: mpmath.exp(-abs(mpf(x))) / 2, D.pdf, "pdf"),
    "laplace_logpdf": make(1, lambda: _pdf_regions("laplace_logpdf"), lambda x: -abs(mpf(x)) - mpmath.log(2), D.logpdf, "logpdf"),
    "laplace_cdf": make(1, lambda: _cdf_regions("laplace_cdf"), _cdf, D.cdf, "cdf"),
    "laplace_sf": make(1, lambda: _cdf_regions("laplace_sf", -1), lambda x: _cdf(-mpf(x)), D.sf, "sf"),
    "laplace_logcdf": make(1, lambda: _cdf_regions("laplace_logcdf"), _logcdf, D.logcdf, "logcdf"),
    "laplace_logsf": make(1, lambda: _cdf_regions("laplace_logsf", -1), lambda x: _logcdf(-mpf(x)), D.logsf, "logsf"),
    "laplace_ppf": make(1, lambda: probs(R("laplace_ppf")), _ppf, D.ppf, "ppf"),
    "laplace_isf": make(1, lambda: probs(R("laplace_isf")), lambda q: -_ppf(q), D.isf, "isf"),
}
