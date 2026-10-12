"""Exponential distribution fixtures (scipy.stats.expon, standard form)."""

import mpmath
from mpmath import mpf
import scipy.stats as st

from cfam import R, xs, probs, make, log1mexp

D = st.expon


def _pdf_regions(name):
    r = R(name)
    return [("tiny", xs(r, 1e-300, 1e-3, log=True)), ("core", xs(r, 1e-3, 2)),
            ("mid", xs(r, 2, 50)), ("tail", xs(r, 50, 745))]


def _cdf_regions(name):
    r = R(name)
    return [("tiny", xs(r, 1e-300, 1e-3, log=True)), ("core", xs(r, 1e-3, 3)),
            ("right", xs(r, 3, 40)), ("far_right", xs(r, 40, 745))]


FIXTURES = {
    "expon_pdf": make(1, lambda: _pdf_regions("expon_pdf"), lambda x: mpmath.exp(-mpf(x)), D.pdf, "pdf"),
    "expon_logpdf": make(1, lambda: _pdf_regions("expon_logpdf"), lambda x: -mpf(x), D.logpdf, "logpdf"),
    "expon_cdf": make(1, lambda: _cdf_regions("expon_cdf"), lambda x: -mpmath.expm1(-mpf(x)), D.cdf, "cdf"),
    "expon_sf": make(1, lambda: _cdf_regions("expon_sf"), lambda x: mpmath.exp(-mpf(x)), D.sf, "sf"),
    "expon_logcdf": make(1, lambda: _cdf_regions("expon_logcdf"), lambda x: log1mexp(mpf(x)), D.logcdf, "logcdf"),
    "expon_logsf": make(1, lambda: _cdf_regions("expon_logsf"), lambda x: -mpf(x), D.logsf, "logsf"),
    "expon_ppf": make(1, lambda: probs(R("expon_ppf")), lambda p: -mpmath.log1p(-mpf(p)), D.ppf, "ppf"),
    "expon_isf": make(1, lambda: probs(R("expon_isf")), lambda q: -mpmath.log(mpf(q)), D.isf, "isf"),
}
