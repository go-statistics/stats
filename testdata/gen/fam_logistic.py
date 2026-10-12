"""Logistic distribution fixtures (scipy.stats.logistic, standard form)."""

import mpmath
from mpmath import mpf
import scipy.stats as st

from cfam import R, xs, probs, make

D = st.logistic


def _pm(r, lo, hi, n=125, log=False):
    return xs(r, lo, hi, n, log=log) + [-x for x in xs(r, lo, hi, n, log=log)]


def _pdf_regions(name):
    r = R(name)
    return [("core", xs(r, -2, 2)), ("mid", _pm(r, 2, 40)), ("tail", _pm(r, 40, 745)),
            ("tiny", _pm(r, 1e-300, 1e-3, log=True))]


def _cdf_regions(name, sign=1):
    r = R(name)
    reg = [("core", xs(r, -2, 2)), ("left", xs(r, 2, 745)), ("right", xs(r, 2, 40)),
           ("far_right", xs(r, 40, 745)), ("tiny", _pm(r, 1e-300, 1e-3, log=True))]
    return [(n, [sign * (-1 if n == "left" else 1) * x for x in pts]) for n, pts in reg]


def _cdf(x):
    return 1 / (1 + mpmath.exp(-mpf(x)))


def _pdf(x):
    e = mpmath.exp(-abs(mpf(x)))
    return e / (1 + e) ** 2


def _ppf(p):
    p = mpf(p)
    return mpmath.log(p / (1 - p))


def _pp(name):
    r = R(name)
    out = probs(r)
    out.append(("half", [0.5 + d for d in xs(r, 1e-16, 1e-2, 125, log=True)] +
                [0.5 - d for d in xs(r, 1e-16, 1e-2, 125, log=True)]))
    return out


FIXTURES = {
    "logistic_pdf": make(1, lambda: _pdf_regions("logistic_pdf"), _pdf, D.pdf, "pdf"),
    "logistic_logpdf": make(1, lambda: _pdf_regions("logistic_logpdf"), lambda x: mpmath.log(_pdf(x)), D.logpdf, "logpdf"),
    "logistic_cdf": make(1, lambda: _cdf_regions("logistic_cdf"), _cdf, D.cdf, "cdf"),
    "logistic_sf": make(1, lambda: _cdf_regions("logistic_sf", -1), lambda x: _cdf(-mpf(x)), D.sf, "sf"),
    "logistic_logcdf": make(1, lambda: _cdf_regions("logistic_logcdf"), lambda x: -mpmath.log1p(mpmath.exp(-mpf(x))), D.logcdf, "logcdf"),
    "logistic_logsf": make(1, lambda: _cdf_regions("logistic_logsf", -1), lambda x: -mpmath.log1p(mpmath.exp(mpf(x))), D.logsf, "logsf"),
    "logistic_ppf": make(1, lambda: _pp("logistic_ppf"), _ppf, D.ppf, "ppf"),
    "logistic_isf": make(1, lambda: _pp("logistic_isf"), lambda q: -_ppf(q), D.isf, "isf"),
}
