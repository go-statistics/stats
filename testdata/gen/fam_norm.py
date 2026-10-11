"""Normal distribution fixtures (scipy.stats.norm, standard form)."""

import mpmath
from mpmath import mpf
import scipy.stats as st

import common
from dists import invert, logu, uni, probs_lower, probs_core, probs_upper

N = 300


def _pdf(x):
    x = mpf(x)
    return mpmath.exp(-x * x / 2) / mpmath.sqrt(2 * mpmath.pi)


def _logpdf(x):
    x = mpf(x)
    return -x * x / 2 - mpmath.log(mpmath.sqrt(2 * mpmath.pi))


def _cdf(x):
    return mpmath.erfc(-mpf(x) / mpmath.sqrt(2)) / 2


def _sf(x):
    return mpmath.erfc(mpf(x) / mpmath.sqrt(2)) / 2


def _ppf(p):
    x0 = float(st.norm.ppf(p))
    return invert(lambda x: mpmath.log(_cdf(x)), lambda x: _pdf(x) / _cdf(x), p, x0,
                  mpf("-inf"), mpf("inf"))


def _isf(q):
    x0 = float(st.norm.isf(q))
    return invert(lambda x: mpmath.log(_sf(x)), lambda x: -_pdf(x) / _sf(x), q, x0,
                  mpf("-inf"), mpf("inf"), decreasing=True)


def xs_pdf(name):
    r = common.rng(name)
    out = [("core", (x,)) for x in uni(r, -1, 1, N)]
    out += [("mid", (x,)) for x in uni(r, -5, 5, N)]
    out += [("tail", (x,)) for x in uni(r, -37.5, 37.5, N)]
    out += [("underflow", (x,)) for x in uni(r, 37.5, 38.7, N // 3)]
    return out


def pdf():
    return 1, xs_pdf("norm_pdf"), _pdf, lambda x: st.norm.pdf(x), "pdf"


def logpdf():
    r = common.rng("norm_logpdf")
    cases = xs_pdf("norm_logpdf")
    cases += [("zero_crossing", (x,)) for x in uni(r, 1.30, 1.42, N)]
    cases += [("huge", (x,)) for x in logu(r, 40, 1e150, N)]
    return 1, cases, _logpdf, lambda x: st.norm.logpdf(x), "logpdf"


def _cdf_cases(name):
    r = common.rng(name)
    out = [("core", (x,)) for x in uni(r, -3, 3, N)]
    out += [("left", (x,)) for x in uni(r, -38, -3, N)]
    out += [("right", (x,)) for x in uni(r, 3, 9, N)]
    return out


def cdf():
    return 1, _cdf_cases("norm_cdf"), _cdf, lambda x: st.norm.cdf(x), "cdf"


def sf():
    return 1, [(reg.replace("left", "L").replace("right", "left").replace("L", "right"), (-a[0],))
               for reg, a in _cdf_cases("norm_sf")], _sf, lambda x: st.norm.sf(x), "sf"


def logcdf():
    r = common.rng("norm_logcdf")
    cases = _cdf_cases("norm_logcdf")
    cases += [("far_left", (x,)) for x in logu(r, 38, 1e10, N, sign=-1)]
    return 1, cases, lambda x: mpmath.log(_cdf(x)), lambda x: st.norm.logcdf(x), "logcdf"


def logsf():
    r = common.rng("norm_logsf")
    cases = [(reg, (-a[0],)) for reg, a in _cdf_cases("norm_logsf")]
    cases += [("far_right", (x,)) for x in logu(r, 38, 1e10, N)]
    return 1, cases, lambda x: mpmath.log(_sf(x)), lambda x: st.norm.logsf(x), "logsf"


def _pcases(name):
    r = common.rng(name)
    out = [("lower", (p,)) for p in probs_lower(r, N)]
    out += [("core", (p,)) for p in probs_core(r, N)]
    out += [("upper", (p,)) for p in probs_upper(r, N)]
    return out


def ppf():
    return 1, _pcases("norm_ppf"), _ppf, lambda p: st.norm.ppf(p), "ppf"


def isf():
    return 1, _pcases("norm_isf"), _isf, lambda q: st.norm.isf(q), "isf"


FIXTURES = {
    "norm_pdf": pdf, "norm_logpdf": logpdf, "norm_cdf": cdf, "norm_sf": sf,
    "norm_logcdf": logcdf, "norm_logsf": logsf, "norm_ppf": ppf, "norm_isf": isf,
}
