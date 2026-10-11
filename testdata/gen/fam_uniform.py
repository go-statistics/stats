"""Uniform distribution fixtures (scipy.stats.uniform, standard form)."""

import mpmath
from mpmath import mpf
import scipy.stats as st

from cfam import R, xs, probs, make

D = st.uniform


def _regions(name):
    r = R(name)
    return [("tiny", xs(r, 1e-300, 1e-3, log=True)), ("core", xs(r, 1e-3, 1 - 1e-3)),
            ("near_one", [1.0 - q for q in xs(r, 1.2e-16, 1e-3, log=True)])]


FIXTURES = {
    "uniform_pdf": make(1, lambda: _regions("uniform_pdf"), lambda x: mpf(1), D.pdf, "pdf"),
    "uniform_logpdf": make(1, lambda: _regions("uniform_logpdf"), lambda x: mpf(0), D.logpdf, "logpdf"),
    "uniform_cdf": make(1, lambda: _regions("uniform_cdf"), lambda x: mpf(x), D.cdf, "cdf"),
    "uniform_sf": make(1, lambda: _regions("uniform_sf"), lambda x: 1 - mpf(x), D.sf, "sf"),
    "uniform_logcdf": make(1, lambda: _regions("uniform_logcdf"), lambda x: mpmath.log(mpf(x)), D.logcdf, "logcdf"),
    "uniform_logsf": make(1, lambda: _regions("uniform_logsf"), lambda x: mpmath.log1p(-mpf(x)), D.logsf, "logsf"),
    "uniform_ppf": make(1, lambda: probs(R("uniform_ppf")), lambda p: mpf(p), D.ppf, "ppf"),
    "uniform_isf": make(1, lambda: probs(R("uniform_isf")), lambda q: 1 - mpf(q), D.isf, "isf"),
}
