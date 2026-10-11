"""Von Mises distribution fixtures (scipy.stats.vonmises, standard form,
first period [-π, π]); arguments (x, kappa).

The oracle for P(X <= x), x <= 0, is the density at x times
∫_0^{x+π} exp(-Δ(τ)) dτ, Δ(τ) = 2κ sin(τ/2) sin(τ/2 - x) >= 0, the
integral by mpmath's tanh-sinh quadrature on panels doubling away from
the peak at τ = 0 (the integrand falls off on the scale
1/(κ|sin x| + √κ)); the upper tail is the mirror image. The quantile is
the exact inverse of that function (dists.invert)."""

import math

import mpmath
from mpmath import mpf
import scipy.stats as st

from cfam import R, make
from dists import invert, logu, uni

D = st.vonmises
SHAPES = [0.0, 0.5, 2.0, 10.0, 60.0, 1e3, 1e5]
N = 150


def _logc(k):
    """log of the density's factor 1/(2π I0(k)), with exp(k) taken out:
    the density is exp(k (cos x - 1) + _logc(k))."""
    k = mpf(k)
    return -mpmath.log(2 * mpmath.pi * mpmath.besseli(0, k) * mpmath.exp(-k))


def _logpdf(x, k):
    x, k = mpf(x), mpf(k)
    return k * (mpmath.cos(x) - 1) + _logc(k)


def _pdf(x, k):
    return mpmath.exp(_logpdf(x, k))


def _lower(x, k):
    """P(X <= x) for -π <= x <= 0."""
    x, k = mpf(x), mpf(k)
    L = x + mpmath.pi
    if L <= 0:
        return mpf(0)
    if k == 0:
        return L / (2 * mpmath.pi)
    f = lambda t: mpmath.exp(-2 * k * mpmath.sin(t / 2) * mpmath.sin(t / 2 - x))
    s = 1 / (k * abs(mpmath.sin(x)) + mpmath.sqrt(k) + 1)
    pts = [mpf(0)]
    while pts[-1] + s < L:
        pts.append(pts[-1] + s)
        s *= 2
    pts.append(L)
    return _pdf(x, k) * mpmath.quad(f, pts)


def _cdf(x, k):
    x = mpf(x)
    if x <= 0:
        return _lower(x, k)
    return 1 - _lower(-x, k)


def _sf(x, k):
    return _cdf(-mpf(x), k)


def _logcdf(x, k):
    x = mpf(x)
    if x <= 0:
        return mpmath.log(_lower(x, k))
    return mpmath.log1p(-_lower(-x, k))


def _logsf(x, k):
    return _logcdf(-mpf(x), k)


def _x(r, region, n=N):
    out = []
    for i in range(n):
        k = SHAPES[i % len(SHAPES)]
        if region == "edge":
            x = -math.pi + logu(r, 1e-12, 1e-2, 1)[0]
        elif region == "left":
            x = r.uniform(-math.pi, -1)
        elif region == "core":
            x = r.uniform(-1, 1)
        else:  # right
            x = r.uniform(1, math.pi)
        out.append((x, k))
    return out


def _regions(name):
    r = R(name)
    return [(reg, _x(r, reg)) for reg in ("edge", "left", "core", "right")]


def _mirror(regions):
    return [(reg, [(-x, k) for x, k in pts]) for reg, pts in regions]


def _ppf(p, k):
    x0 = float(D.ppf(p, k))
    if not (-math.pi < x0 < math.pi):
        x0 = 0.0
    return invert(lambda x: _logcdf(x, k), lambda x: _pdf(x, k) / _cdf(x, k), p, x0,
                  -mpmath.pi, mpmath.pi)


def _isf(q, k):
    return -_ppf(q, k)


def _p(name, n=N):
    r = R(name)
    out = []
    for reg, ps in (("lower", logu(r, 1e-300, 1e-3, n)), ("core", uni(r, 1e-3, 1 - 1e-3, n)),
                    ("upper", [1.0 - q for q in logu(r, 1.2e-16, 1e-3, n)])):
        out.append((reg, [(p, SHAPES[i % len(SHAPES)]) for i, p in enumerate(ps)]))
    return out


FIXTURES = {
    "vonmises_pdf": make(2, lambda: _regions("vonmises_pdf"), _pdf, D.pdf, "pdf"),
    "vonmises_logpdf": make(2, lambda: _regions("vonmises_logpdf"), _logpdf, D.logpdf, "logpdf"),
    "vonmises_cdf": make(2, lambda: _regions("vonmises_cdf"), _cdf, D.cdf, "cdf"),
    "vonmises_sf": make(2, lambda: _mirror(_regions("vonmises_sf")), _sf, D.sf, "sf"),
    "vonmises_logcdf": make(2, lambda: _regions("vonmises_logcdf"), _logcdf, D.logcdf, "logcdf"),
    "vonmises_logsf": make(2, lambda: _mirror(_regions("vonmises_logsf")), _logsf, D.logsf, "logsf"),
    "vonmises_ppf": make(2, lambda: _p("vonmises_ppf"), _ppf, D.ppf, "ppf"),
    "vonmises_isf": make(2, lambda: _p("vonmises_isf"), _isf, D.isf, "isf"),
}
