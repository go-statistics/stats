"""Gumbel (right) distribution fixtures (scipy.stats.gumbel_r, standard form)."""

import mpmath
from mpmath import mpf
import scipy.stats as st

from cfam import R, xs, probs, make, log1mexp

D = st.gumbel_r


def _pdf_regions(name):
    r = R(name)
    return [("core", xs(r, -2, 3)), ("left", xs(r, -6.6, -2)), ("right", xs(r, 3, 40)),
            ("far_right", xs(r, 40, 745)), ("tiny", xs(r, 1e-300, 1e-3, log=True))]


def _cdf_regions(name):
    r = R(name)
    return [("core", xs(r, -2, 3)), ("left", xs(r, -6.6, -2)), ("right", xs(r, 3, 40)),
            ("far_right", xs(r, 40, 745)), ("tiny", xs(r, 1e-300, 1e-3, log=True))]


def _e(x):
    return mpmath.exp(-mpf(x))


FIXTURES = {
    "gumbel_r_pdf": make(1, lambda: _pdf_regions("gumbel_r_pdf"), lambda x: mpmath.exp(-mpf(x) - _e(x)), D.pdf, "pdf"),
    "gumbel_r_logpdf": make(1, lambda: _pdf_regions("gumbel_r_logpdf") + [("far_left", xs(R("g"), -709, -6.6))],
                            lambda x: -mpf(x) - _e(x), D.logpdf, "logpdf"),
    "gumbel_r_cdf": make(1, lambda: _cdf_regions("gumbel_r_cdf"), lambda x: mpmath.exp(-_e(x)), D.cdf, "cdf"),
    "gumbel_r_sf": make(1, lambda: _cdf_regions("gumbel_r_sf"), lambda x: -mpmath.expm1(-_e(x)), D.sf, "sf"),
    "gumbel_r_logcdf": make(1, lambda: _cdf_regions("gumbel_r_logcdf") + [("far_left", xs(R("g"), -709, -6.6))],
                            lambda x: -_e(x), D.logcdf, "logcdf"),
    "gumbel_r_logsf": make(1, lambda: _cdf_regions("gumbel_r_logsf"), lambda x: log1mexp(_e(x)), D.logsf, "logsf"),
    "gumbel_r_ppf": make(1, lambda: probs(R("gumbel_r_ppf")), lambda p: -mpmath.log(-mpmath.log(mpf(p))), D.ppf, "ppf"),
    "gumbel_r_isf": make(1, lambda: probs(R("gumbel_r_isf")), lambda q: -mpmath.log(-mpmath.log1p(-mpf(q))), D.isf, "isf"),
}
