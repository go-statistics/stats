"""Half-normal distribution fixtures (scipy.stats.halfnorm, standard form)."""

import mpmath
from mpmath import mpf
import scipy.stats as st

from cfam import R, xs, probs, make

D = st.halfnorm


def _regions(name):
    r = R(name)
    return [("tiny", xs(r, 1e-300, 1e-3, log=True)), ("core", xs(r, 1e-3, 3)),
            ("right", xs(r, 3, 9)), ("tail", xs(r, 9, 38.4))]


def _cdf(x):
    return mpmath.erf(mpf(x) / mpmath.sqrt(2))


def _sf(x):
    return mpmath.erfc(mpf(x) / mpmath.sqrt(2))


def _ppf(p):
    return mpmath.sqrt(2) * mpmath.erfinv(mpf(p))


def _isf(q):
    # erfinv(1 - q) loses q in mpmath's own subtraction; invert erfc instead
    q = mpf(q)
    x = mpf(float(st.halfnorm.isf(float(q))))
    for _ in range(200):
        f = _sf(x) - q
        d = -mpmath.sqrt(2 / mpmath.pi) * mpmath.exp(-x * x / 2)
        nx = x - f / d
        if abs(nx - x) <= abs(x) * mpf(2) ** (-mpmath.mp.prec + 4):
            return nx
        x = nx
    return x


def _ppf_any(p):
    p = mpf(p)
    if p > 0.5:
        return _isf(1 - p)
    return _ppf(p)


FIXTURES = {
    "halfnorm_pdf": make(1, lambda: _regions("halfnorm_pdf"), lambda x: mpmath.sqrt(2 / mpmath.pi) * mpmath.exp(-mpf(x) ** 2 / 2), D.pdf, "pdf"),
    "halfnorm_logpdf": make(1, lambda: _regions("halfnorm_logpdf") + [("huge", xs(R("hl"), 40, 1e150, log=True))],
                            lambda x: mpmath.log(mpmath.sqrt(2 / mpmath.pi)) - mpf(x) ** 2 / 2, D.logpdf, "logpdf"),
    "halfnorm_cdf": make(1, lambda: _regions("halfnorm_cdf"), _cdf, D.cdf, "cdf"),
    "halfnorm_sf": make(1, lambda: _regions("halfnorm_sf"), _sf, D.sf, "sf"),
    "halfnorm_logcdf": make(1, lambda: _regions("halfnorm_logcdf"), lambda x: mpmath.log1p(-_sf(x)) if x > 1 else mpmath.log(_cdf(x)), D.logcdf, "logcdf"),
    "halfnorm_logsf": make(1, lambda: _regions("halfnorm_logsf"), lambda x: mpmath.log(_sf(x)) if x > 1 else mpmath.log1p(-_cdf(x)), D.logsf, "logsf"),
    "halfnorm_ppf": make(1, lambda: probs(R("halfnorm_ppf")), _ppf_any, D.ppf, "ppf"),
    "halfnorm_isf": make(1, lambda: probs(R("halfnorm_isf")), _isf, D.isf, "isf"),
}
