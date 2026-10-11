"""Rayleigh distribution fixtures (scipy.stats.rayleigh, standard form)."""

import mpmath
from mpmath import mpf
import scipy.stats as st

from cfam import R, xs, probs, make, log1mexp

D = st.rayleigh


def _regions(name):
    r = R(name)
    return [("tiny", xs(r, 1e-300, 1e-3, log=True)), ("core", xs(r, 1e-3, 3)),
            ("right", xs(r, 3, 9)), ("tail", xs(r, 9, 38.6))]


def _h(x):
    x = mpf(x)
    return x * x / 2


FIXTURES = {
    "rayleigh_pdf": make(1, lambda: _regions("rayleigh_pdf"), lambda x: mpf(x) * mpmath.exp(-_h(x)), D.pdf, "pdf"),
    "rayleigh_logpdf": make(1, lambda: _regions("rayleigh_logpdf") + [("huge", xs(R("rl"), 40, 1e150, log=True))],
                            lambda x: mpmath.log(mpf(x)) - _h(x), D.logpdf, "logpdf"),
    "rayleigh_cdf": make(1, lambda: _regions("rayleigh_cdf"), lambda x: -mpmath.expm1(-_h(x)), D.cdf, "cdf"),
    "rayleigh_sf": make(1, lambda: _regions("rayleigh_sf"), lambda x: mpmath.exp(-_h(x)), D.sf, "sf"),
    "rayleigh_logcdf": make(1, lambda: _regions("rayleigh_logcdf"), lambda x: log1mexp(_h(x)), D.logcdf, "logcdf"),
    "rayleigh_logsf": make(1, lambda: _regions("rayleigh_logsf") + [("huge", xs(R("rs"), 40, 1e150, log=True))],
                           lambda x: -_h(x), D.logsf, "logsf"),
    "rayleigh_ppf": make(1, lambda: probs(R("rayleigh_ppf")), lambda p: mpmath.sqrt(-2 * mpmath.log1p(-mpf(p))), D.ppf, "ppf"),
    "rayleigh_isf": make(1, lambda: probs(R("rayleigh_isf")), lambda q: mpmath.sqrt(-2 * mpmath.log(mpf(q))), D.isf, "isf"),
}
