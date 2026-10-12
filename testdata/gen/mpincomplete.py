"""Incomplete gamma and beta fixtures: gammainc, gammaincc, gammaincinv,
gammainccinv, betainc, betaincc, betaincinv, betainccinv.

The oracles do not call mp.gammainc/mp.betainc blindly: mpmath's
hypergeometric routes give up for large parameters ("converges too slowly")
and, for some tiny-parameter beta cases, return a wrong value that is the
same at both precisions (an infinite upper tail, for instance), which the
two-precision self-check cannot catch. Instead:

* the SMALLER tail is always computed directly and the other one as its
  complement (no cancellation), except where the smaller tail's series
  converges too slowly; then the complement is formed with as many extra
  bits as the cancellation costs;
* moderate parameters use positive-term series (P: x^a e^-x/Gamma(a+1) *
  sum x^n/(a+1)_n, DLMF 8.7.1; I_x: x^a (1-x)^b/(a B) * 2F1(a+b,1;a+1;x),
  DLMF 8.17.8) or, for the gamma upper tail, mpmath's own upper gammainc;
* large parameters integrate the tail numerically (tanh-sinh, mp.quad) in a
  variable measured from the end point, with geometric breakpoints at the
  integrand's own scale, and the normalising constant formed in logs with
  enough extra precision for the log-gamma cancellation.

Inverses are the exact root, at working precision, of the oracle forward
function at the float64 probability: a safeguarded Newton iteration
(bisection fallback inside a sign-change bracket) in t = log x (or log(1-x)
for the upper beta tail), the variable in which the tail is close to linear.
"""

import math

import mpmath
import scipy.special as sc
from mpmath import mp, mpf

import common
from common import loguniform, rng, uniform

N = 1000
NQ = 300  # regions whose oracle integrates numerically (about 1 s/point)

nan, inf = math.nan, math.inf


def _tiny_guard(*vals):
    """Extra bits when a parameter is tiny: the tail then moves like
    x^a, its logarithm by a*log x, and locating the root to 2^-prec needs
    log I to about a*2^-prec absolutely."""
    m = min(float(v) for v in vals)
    return max(0, -int(math.log2(m))) if m < 1 else 0


def _guard(*vals):
    """Extra bits needed for log-gamma / exponent cancellation."""
    m = max([abs(mpf(v)) for v in vals] + [mpf(2)])
    return 64 + int(mp.log(m, 2))


# ------------------------------------------------------------ gamma oracles

def _gam_prefix_log(a, x):
    """log(x^a e^-x / Gamma(a)) for mpf a, x > 0."""
    return a * mp.log(x) - x - mp.loggamma(a)


def _gam_lower_series(a, x):
    """P(a, x) by the positive series; good for x below about a (or small)."""
    s = term = mpf(1)
    eps = mpf(2) ** (-mp.prec - 8)
    n = 0
    while True:
        n += 1
        term *= x / (a + n)
        s += term
        if term < eps * s:
            break
        if n > 2000000:
            raise ValueError("lower series too slow")
    return mp.exp(_gam_prefix_log(a, x) - mp.log(a)) * s


def _quad_tail(logf_rel, scale, lim, below):
    """Integrate exp(logf_rel(s)) for s in [0, lim] (lim may be inf) where
    logf_rel(0) = 0 and the integrand decreases away from s = 0 at about
    the given scale. Breakpoints are geometric in the scale."""
    pts = [mpf(0)]
    k = 0
    cut = -(mp.prec * 0.7 + 60)
    while True:
        s = scale * mpf(4) ** k
        if lim is not None and s >= lim:
            pts.append(lim)
            break
        pts.append(s)
        if logf_rel(s) < cut:
            if lim is None:
                pts.append(mp.inf)
            break
        k += 1
    return mp.quad(lambda s: mp.exp(logf_rel(s)), pts)


def _gam_tail_quad(a, x, upper):
    if a > 2.0 ** 100 or x > 2.0 ** 100:
        # a - 1 and the like must stay exact: carry the whole computation
        with mp.extraprec(_guard(a, x)):
            return _gam_tail_quad_(a, x, upper)
    return _gam_tail_quad_(a, x, upper)


def _gam_tail_quad_(a, x, upper):
    """Q(a,x) (upper) or P(a,x) (lower) by integrating t^(a-1) e^-t from x
    outwards; the constant x^(a-1) e^-x / Gamma(a) is taken out."""
    am1 = a - 1
    with mp.extraprec(_guard(a, x)):
        c = am1 * mp.log(x) - x - mp.loggamma(a)
    if c < -3000:
        return mpf(0)
    d = abs(am1 / x - 1)
    sd = mp.sqrt(a) if a > 1 else mpf(1)
    h = min(1 / d, sd) if d > 0 else sd
    if upper:
        f = lambda s: am1 * mp.log1p(s / x) - s
        v = _quad_tail(f, h, None, False)
    else:
        h = min(h, x)
        f = lambda s: am1 * mp.log1p(-s / x) + s if s < x else mp.ninf
        v = _quad_tail(f, h, x, True)
    return mp.exp(c) * v


def gam_P_Q(a, x):
    """(P, Q) of mpf a > 0, x > 0, one of them computed directly as the
    smaller tail and the other as its complement."""
    if a > 1e4:
        if x < a:
            p = _gam_tail_quad(a, x, False)
            return p, 1 - p
        q = _gam_tail_quad(a, x, True)
        return 1 - q, q
    if x < a or x < 1:
        with mp.extraprec(_guard(a, x)):
            p = _gam_lower_series(a, x)
        if p > 0.75:
            # the upper tail is small: get it with the cancellation paid for
            q = _complement(lambda: _gam_lower_series(a, x), p)
            return p, q
        return p, 1 - p
    with mp.extraprec(_guard(a, x)):
        try:
            q = mp.gammainc(a, x, mp.inf, regularized=True)
        except (mpmath.libmp.NoConvergence, ValueError, ZeroDivisionError):
            q = _gam_tail_quad(a, x, True)
    return 1 - q, q


def _complement(fn, v):
    """1 - fn() with enough extra precision for the cancellation, given a
    first estimate v of fn()."""
    c = 1 - v
    bits = 64 if c == 0 else max(0, -int(mp.log(abs(c), 2))) + 64
    for _ in range(6):
        with mp.extraprec(bits):
            w = fn()
            c = 1 - w
        if c != 0 and -mp.log(abs(c), 2) + 40 < bits:
            return +c
        bits *= 2
        if bits > 20000:
            break
    raise ValueError("complement")


def gammainc_mp(a, x):
    if math.isnan(a) or math.isnan(x) or a < 0 or x < 0:
        return nan
    if a == 0:
        return nan if x == 0 else 1.0
    if math.isinf(a):
        return nan if math.isinf(x) else 0.0
    if x == 0:
        return 0.0
    if math.isinf(x):
        return 1.0
    return gam_P_Q(mpf(a), mpf(x))[0]


def gammaincc_mp(a, x):
    v = gammainc_mp(a, x)
    if isinstance(v, float):
        return v if math.isnan(v) else 1.0 - v
    return gam_P_Q(mpf(a), mpf(x))[1]


def _gam_dlog(a, x, tailv):
    """d log(P)/d log(x) = x * x^(a-1) e^-x / Gamma(a) / P."""
    return mp.exp(_gam_prefix_log(a, x)) / tailv


def _solve(F, t0, increasing, step0=1):
    """Root of F(t) (F returns (value, derivative)), F monotone. Newton from
    t0, kept inside the sign-change bracket as soon as one is known
    (bisection when a step leaves it, doubling steps while it is open).
    A first pass at 96 bits gets close cheaply; the final pass runs at the
    working precision and needs only a couple of iterations."""
    prec = mp.prec
    t = mpf(t0)
    p1 = max(96, prec - 150)
    for wp, tolbits in ((p1, p1 - 36), (prec, prec - 12)):
        with mp.workprec(wp):
            t = _solve1(F, +t, increasing, step0, tolbits)
    return t


def _solve1(F, t, increasing, step0, tolbits):
    lo, hi = None, None
    step = mpf(step0)
    tol = mpf(2) ** (-tolbits)
    for _ in range(600):
        v, dv = F(t)
        if v == 0:
            return t
        below = (v < 0) == increasing
        if below:
            lo = t
        else:
            hi = t
        nt = None
        if dv != 0 and mp.isfinite(dv) and mp.isfinite(v):
            nt = t - v / dv
            if abs(nt - t) <= tol * max(1, abs(t)):
                return nt
        ok = nt is not None and (lo is None or nt > lo) and (hi is None or nt < hi)
        if not ok:
            if lo is not None and hi is not None:
                nt = (lo + hi) / 2
            elif lo is None:
                nt = hi - step
                step *= 2
            else:
                nt = lo + step
                step *= 2
        if abs(nt - t) <= tol * max(1, abs(nt)):
            return nt
        if lo is not None and hi is not None and abs(hi - lo) <= tol * max(1, abs(lo)):
            return nt
        t = nt
    raise ValueError("no convergence")


def _gam_inv(a, target, lower, guess):
    """x with P(a,x) = target (lower) or Q(a,x) = target, target mpf in (0,1)."""
    a = mpf(a)
    ltarget = mp.log(target)

    def F(t):
        x = mp.exp(t)
        if x == 0:
            # x underflowed even mpf: P ~ x^a / Gamma(a+1)
            lp = a * t - mp.loggamma(a + 1)
            return (lp - ltarget, a) if lower else (-ltarget, mpf(0))
        p, q = gam_P_Q(a, x)
        v = p if lower else q
        if v == 0:
            return (mp.ninf if lower else mp.ninf), mpf(0)
        d = _gam_dlog(a, x, v)
        return (mp.log(v) - ltarget, d) if lower else (mp.log(v) - ltarget, -d)

    with mp.extraprec(_guard(a)):
        v800 = F(mpf(-800))[0]
    if (v800 >= 0) == lower:
        return 0.0  # the root underflows float64
    if guess is not None and 0 < guess < inf:
        t0 = mp.log(guess)
    else:
        t0 = mp.log(a) if a > 1 else (mp.log(target) + mp.loggamma(a + 1)) / a if lower else mpf(0)
    step0 = 1 / mp.sqrt(a) if a > 1 else 1
    with mp.extraprec(_guard(a) + _tiny_guard(a)):
        t = _solve(F, t0, lower, step0)
    if t < -800:
        return 0.0  # x underflows float64 whatever its last digits
    return mp.exp(t)


def gammaincinv_mp(a, p, guess=None, upper=False):
    if math.isnan(a) or math.isnan(p) or a < 0 or p < 0 or p > 1:
        return nan
    lo_end, hi_end = (inf, 0.0) if upper else (0.0, inf)
    if p == 0:
        return lo_end
    if p == 1:
        return hi_end
    if a == 0 or math.isinf(a):
        return nan
    if a > 2.0 ** 200 and 1e-300 <= p <= 1 - 1e-16:
        # x = a + s sqrt(a) + O(s^2) with |s| < 40 here (the normal
        # approximation's quantile, DLMF 8.12.18): |x/a - 1| < 2^-90, so the
        # float64 nearest x is a itself
        return mpf(a)
    pm = mpf(p)
    if pm <= 0.5:
        return _gam_inv(a, pm, not upper, guess)
    return _gam_inv(a, 1 - pm, upper, guess)


def gammainccinv_mp(a, q, guess=None):
    return gammaincinv_mp(a, q, guess, upper=True)


# ------------------------------------------------------------- beta oracles

def _logbeta(a, b):
    with mp.extraprec(_guard(a, b)):
        return mp.loggamma(a) + mp.loggamma(b) - mp.loggamma(mp.fadd(a, b, exact=True))


def _ib_series(a, b, x, y):
    """I_x(a,b), y = 1-x exactly: x^a y^b / (a B) * 2F1(a+b,1;a+1;x)."""
    s = term = mpf(1)
    eps = mpf(2) ** (-mp.prec - 8)
    ab = a + b
    n = 0
    while True:
        term *= (ab + n) / (a + 1 + n) * x
        n += 1
        s += term
        if term < eps * s:
            break
        if n > 400000:
            raise ValueError("series too slow")
    with mp.extraprec(_guard(a, b)):
        lp = a * mp.log(x) + b * mp.log(y) - _logbeta(a, b) - mp.log(a)
        return mp.exp(lp) * s


def _ib_quad(a, b, x, y):
    if a > 2.0 ** 100 or b > 2.0 ** 100:
        with mp.extraprec(_guard(a, b)):
            return _ib_quad_(a, b, x, y)
    return _ib_quad_(a, b, x, y)


def _ib_quad_(a, b, x, y):
    """I_x(a,b) for x below the mode region: integrate t^(a-1)(1-t)^(b-1)
    from x down to 0 in s = x - t, with 1-t = y + s exact."""
    am1, bm1 = a - 1, b - 1
    with mp.extraprec(_guard(a, b)):
        c = am1 * mp.log(x) + bm1 * mp.log(y) - _logbeta(a, b)
    if c < -3000:
        return mpf(0)
    d = abs(am1 / x - bm1 / y)
    sd = mp.sqrt(a * b / (a + b) ** 3)
    h = min(1 / d, sd) if d > 0 else sd
    h = min(h, x)

    def f(s):
        if s >= x:
            return mp.ninf
        return am1 * mp.log1p(-s / x) + bm1 * mp.log1p(s / y)

    return mp.exp(c) * _quad_tail(f, h, x, True)


def _ib_fast(a, b, x):
    # the series' term ratio starts at (a+b)x/(a+1) and tends to x: when
    # both are well below 1 it converges fast whatever a and b are
    return (a + b) * x / (a + 1) < 0.95 and x < 0.95


def ib_I_Ic(a, b, x):
    """(I_x(a,b), 1 - I_x(a,b)) for mpf a, b > 0, x in (0,1). One tail is
    computed directly and the other as its complement, with the
    cancellation paid for in extra bits when the direct one is close to 1
    (a tiny parameter puts nearly all the mass at an end point, far from
    the mean). The direct tail is the one on x's side of the mean when its
    series converges fast, else the other one when that one does, else the
    series (moderate a+b) or the quadrature on x's side."""
    # y must be exactly 1 - x: inside the root finder x is not a float64,
    # and an inexact y shifts y^b by b*2^-prec, which a tail of 1e-100
    # computed as a complement would not survive
    y = mp.fsub(1, x, exact=True) if x > mpf(2) ** -3000 else 1 - x
    lower = x <= a / (a + b)
    fast = _ib_fast(a, b, x) if lower else _ib_fast(b, a, y)
    if not fast:
        if (_ib_fast(b, a, y) if lower else _ib_fast(a, b, x)):
            lower, fast = not lower, True
    method = _ib_series if fast or a + b <= 3000 else _ib_quad
    fn = (lambda: method(a, b, x, y)) if lower else (lambda: method(b, a, y, x))
    v = fn()
    w = 1 - v if v <= 0.75 else _complement(fn, v)
    return (v, w) if lower else (w, v)


def _beta_edges(a, b, x):
    if math.isnan(a) or math.isnan(b) or math.isnan(x) or a < 0 or b < 0 or x < 0 or x > 1:
        return nan
    if (a == 0 and b == 0) or (math.isinf(a) and math.isinf(b)):
        return nan
    if a == 0 or math.isinf(b):
        return 0.0 if x == 0 else 1.0
    if b == 0 or math.isinf(a):
        return 1.0 if x == 1 else 0.0
    if x == 0:
        return 0.0
    if x == 1:
        return 1.0
    return None


def betainc_mp(a, b, x):
    e = _beta_edges(a, b, x)
    if e is not None:
        return e
    return ib_I_Ic(mpf(a), mpf(b), mpf(x))[0]


def betaincc_mp(a, b, x):
    e = _beta_edges(a, b, x)
    if e is not None:
        return e if math.isnan(e) else 1.0 - e
    return ib_I_Ic(mpf(a), mpf(b), mpf(x))[1]


def _ib_side(pa, pb, P, Q, lb, guess):
    """u in (0, 1/2] with I_u(pa,pb) = P (and 1 - I_u = Q, P + Q = 1),
    matched on whichever of P, Q is smaller, solved in t = log u."""
    use_lower = P <= Q
    target = P if use_lower else Q
    ltarget = mp.log(target)

    def F(t):
        u = mp.exp(t)
        if u == 0:
            return (mp.ninf, mpf(0)) if use_lower else (ltarget, mpf(0))
        if u >= 1:
            return (-ltarget, mpf(1)) if use_lower else (mp.inf, mpf(1))
        i, ic = ib_I_Ic(pa, pb, u)
        with mp.extraprec(_guard(pa, pb)):
            dens = mp.exp(pa * t + (pb - 1) * mp.log1p(-u) - lb)
        if use_lower:
            if i == 0:
                return mp.ninf, mpf(0)
            return mp.log(i) - ltarget, dens / i
        return ltarget - mp.log(ic), dens / ic

    with mp.extraprec(_guard(pa, pb)):
        if F(mpf(-800))[0] >= 0:
            return mpf(0)  # the root underflows float64
    if guess is not None and 0 < guess <= 0.5:
        t0 = mp.log(guess)
    else:
        t0 = min((mp.log(P) + lb + mp.log(pa)) / pa, mp.log(0.25))
    sd = mp.sqrt(pa * pb / (pa + pb) ** 3)
    step0 = min(1, sd / mp.exp(t0)) if t0 > -700 else 1
    with mp.extraprec(_guard(pa, pb) + _tiny_guard(pa, pb)):
        t = _solve(F, t0, True, step0)
    return mp.exp(t)


def _ib_inv(a, b, P, Q, guess):
    """x with I_x(a,b) = P, 1 - I_x(a,b) = Q (P + Q = 1 exactly). The root
    is located on one side of 1/2 first and found in the variable that is
    small there (x, or 1 - x), so that neither end loses digits."""
    a, b = mpf(a), mpf(b)
    if a == b and P == Q:
        return 0.5  # symmetry
    lb = _logbeta(a, b)
    ihalf = ib_I_Ic(a, b, mpf(0.5))[0]
    if P <= ihalf:
        g = guess if guess is not None and 0 < guess <= 0.5 else None
        u = _ib_side(a, b, P, Q, lb, g)
        return 0.0 if u == 0 else u
    g = 1 - guess if guess is not None and 0.5 <= guess < 1 else None
    u = _ib_side(b, a, Q, P, lb, g)
    return 1.0 if u == 0 else 1 - u


def betaincinv_mp(a, b, p, guess=None, upper=False):
    if math.isnan(a) or math.isnan(b) or math.isnan(p) or a <= 0 or b <= 0 or p < 0 or p > 1:
        return nan
    if p == 0:
        return 1.0 if upper else 0.0
    if p == 1:
        return 0.0 if upper else 1.0
    if math.isinf(a) or math.isinf(b):
        return nan
    pm = mpf(p)
    if upper:
        return _ib_inv(a, b, 1 - pm, pm, guess)
    return _ib_inv(a, b, pm, 1 - pm, guess)


def betainccinv_mp(a, b, q, guess=None):
    return betaincinv_mp(a, b, q, guess, upper=True)


# ------------------------------------------------------------------ cases

GA = [nan, -1.0, -inf, 0.0, 5e-324, 1e-300, 0.5, 1.0, 2.0, 1e300, inf]
GX = [nan, -1.0, -inf, 0.0, 5e-324, 1e-300, 0.5, 1.0, 2.0, 1e300, inf]


def gam_cases(name):
    r = rng(name)
    c = [("special", (a, x)) for a in GA for x in GX]
    la = lambda n, lo, hi: loguniform(r, lo, hi, n)
    c += [("a<1e-3", (a, x)) for a, x in zip(la(N, 1e-300, 1e-3), la(N, 1e-300, 1e3))]
    c += [("a in [1e-3,1)", (a, x)) for a, x in zip(la(N, 1e-3, 1), la(N, 1e-20, 60))]
    for a in la(N, 1, 30):
        c.append(("a in [1,30)", (a, a * math.exp(r.uniform(-4, 1.6)))))
    ints = [float(k) for k in range(1, 31)] + [k + 0.5 for k in range(0, 30)]
    c += [("integer, half-integer a", (r.choice(ints), x)) for x in la(N, 0.05, 120)]
    for a in la(N, 30, 1e4):
        c.append(("a in [30,1e4), x near a", (a, max(1e-3, a + r.uniform(-12, 12) * math.sqrt(a)))))
    for a in la(NQ, 1e4, 1e15):
        c.append(("a in [1e4,1e15], x near a", (a, a + r.uniform(-37, 37) * math.sqrt(a))))
    c += [("x tiny (P to subnormal)", (a, x)) for a, x in zip(la(N, 1e-3, 200), la(N, 1e-300, 1e-8))]
    for a in la(N, 1e-3, 200):
        c.append(("x large (Q to subnormal)", (a, a + r.uniform(5, 780))))
    c += [("x huge", (a, x)) for a, x in zip(la(100, 1e-3, 1e4), la(100, 1e3, 1e300))]
    return c


def gaminv_cases(name, guessfn):
    r = rng(name)
    c = [("special", (a, p)) for a in GA for p in [nan, -0.5, 0.0, 1e-300, 0.25, 0.5, 0.75, 1.0, 1.5, inf]]
    la = lambda n, lo, hi: loguniform(r, lo, hi, n)
    c += [("a<1e-3", (a, r.random())) for a in la(N // 2, 1e-300, 1e-3)]
    # p uniform mostly gives an x that underflows: also aim at representable x
    fwd = sc.gammaincc if name == "gammainccinv" else sc.gammainc
    for a, xt in zip(la(N // 2, 1e-300, 1e-3), la(N // 2, 1e-300, 30)):
        p = float(fwd(a, xt))
        if 0 < p < 1:
            c.append(("a<1e-3", (a, p)))
    c += [("a in [1e-3,1)", (a, r.random())) for a in la(N, 1e-3, 1)]
    c += [("a in [1,100)", (a, r.random())) for a in la(N, 1, 100)]
    c += [("a in [100,1e4)", (a, r.random())) for a in la(N, 100, 1e4)]
    c += [("a in [1e4,1e15]", (a, r.random())) for a in la(NQ // 2, 1e4, 1e15)]
    c += [("p tiny", (a, p)) for a, p in zip(la(N, 1e-3, 1e3), la(N, 1e-300, 1e-3))]
    c += [("p near 1", (a, 1 - p)) for a, p in zip(la(N, 1e-3, 1e3), la(N, 1.2e-16, 1e-3))]
    return c


def _robust(mpfn):
    """An oracle that cannot settle a point (no bracket, no convergence)
    must not abort the whole fixture nor invent a value: the row is dropped
    and counted with the timeouts."""
    def f(*args, **kw):
        try:
            return mpfn(*args, **kw)
        except (ValueError, TypeError, ZeroDivisionError, mpmath.libmp.NoConvergence) as e:
            import sys
            print("oracle failed at %r: %r" % (args, e), file=sys.stderr)
            raise common.Timeout()
    return f


def _with_guess(mpfn, spfn):
    """Seed the root finder with SciPy's answer when it is usable (it only
    speeds the search: the root is found inside a verified bracket)."""
    def f(*args):
        try:
            g = float(spfn(*args))
        except Exception:
            g = None
        if g is not None and (math.isnan(g) or math.isinf(g)):
            g = None
        return mpfn(*args, guess=g)
    return _robust(f)


BA = [nan, -1.0, 0.0, 1e-300, 0.5, 2.0, 1e300, inf]
BX = [nan, -0.5, 0.0, 5e-324, 0.25, 0.5, 1.0, 1.5]


def beta_cases(name):
    r = rng(name)
    c = [("special", (a, b, x)) for a in BA for b in BA for x in BX]
    la = lambda n, lo, hi: loguniform(r, lo, hi, n)
    u = lambda n: [r.random() for _ in range(n)]

    def xs(n):
        out = []
        for _ in range(n):
            k = r.random()
            if k < 0.6:
                out.append(r.random())
            elif k < 0.8:
                out.append(math.exp(r.uniform(math.log(1e-30), math.log(0.5))))
            else:
                out.append(1 - math.exp(r.uniform(math.log(1e-16), math.log(0.5))))
        return out

    c += [("a,b<1e-3", (a, b, x)) for a, b, x in zip(la(N, 1e-300, 1e-3), la(N, 1e-300, 1e-3), xs(N))]
    c += [("a<1e-3<=b", (a, b, x)) for a, b, x in zip(la(N, 1e-300, 1e-3), la(N, 1e-3, 1e3), xs(N))]
    c += [("b<1e-3<=a", (a, b, x)) for a, b, x in zip(la(N, 1e-3, 1e3), la(N, 1e-300, 1e-3), xs(N))]
    c += [("a,b in [1e-3,1)", (a, b, x)) for a, b, x in zip(la(N, 1e-3, 1), la(N, 1e-3, 1), xs(N))]
    c += [("a,b in [1,100)", (a, b, x)) for a, b, x in zip(la(N, 1, 100), la(N, 1, 100), xs(N))]

    def near_mean(region, n, lo, hi, spread):
        for a, b in zip(la(n, lo, hi), la(n, lo, hi)):
            m = a / (a + b)
            sd = math.sqrt(a * b / ((a + b) ** 2 * (a + b + 1)))
            for _ in range(20):
                x = m + r.uniform(-spread, spread) * sd
                if 0 < x < 1:
                    c.append((region, (a, b, x)))
                    break

    near_mean("a,b in [100,1e4), x near mean", N, 100, 1e4, 12)
    near_mean("a,b in [1e4,1e15], x near mean", NQ, 1e4, 1e15, 37)
    for a, b in zip(la(N, 1e-3, 10), la(N, 1e3, 1e15)):
        x = min(a / b * math.exp(r.uniform(math.log(1e-3), math.log(1e3))), 0.999)
        if r.random() < 0.5:
            c.append(("a small, b large", (a, b, x)))
        else:
            c.append(("a small, b large", (b, a, 1 - x)))
    c += [("x tiny (I to subnormal)", (a, b, x)) for a, b, x in zip(la(N, 1, 100), la(N, 1e-3, 100), la(N, 1e-300, 1e-6))]
    c += [("x near 1 (I^c to subnormal)", (a, b, 1 - x)) for a, b, x in zip(la(N, 1e-3, 100), la(N, 1, 100), la(N, 1.2e-16, 1e-6))]
    return c


def betainv_cases(name):
    r = rng(name)
    c = [("special", (a, b, p)) for a in BA for b in BA for p in [nan, -0.5, 0.0, 1e-300, 0.25, 0.5, 1.0, 1.5]]
    la = lambda n, lo, hi: loguniform(r, lo, hi, n)
    u = lambda n: [r.random() for _ in range(n)]
    fwd = sc.betaincc if name == "betainccinv" else sc.betainc

    def targeted(region, n, alo, ahi, blo, bhi, swap):
        # p uniform mostly gives an x that rounds to 0 or 1 when a parameter
        # is tiny: also aim at a representable x
        for a, b, p in zip(la(n // 2, alo, ahi), la(n // 2, blo, bhi), u(n // 2)):
            c.append((region, (b, a, p) if swap and r.random() < 0.5 else (a, b, p)))
        for a, b in zip(la(n // 2, alo, ahi), la(n // 2, blo, bhi)):
            if swap and r.random() < 0.5:
                a, b = b, a
            xt = math.exp(r.uniform(math.log(1e-300), 0))
            if r.random() < 0.5:
                xt = 1 - xt
            p = float(fwd(a, b, xt))
            if 0 < p < 1:
                c.append((region, (a, b, p)))

    targeted("a,b<1e-3", N, 1e-300, 1e-3, 1e-300, 1e-3, False)
    targeted("a<1e-3<=b or b<1e-3<=a", N, 1e-300, 1e-3, 1e-3, 1e3, True)
    c += [("a,b in [1e-3,1)", (a, b, p)) for a, b, p in zip(la(N, 1e-3, 1), la(N, 1e-3, 1), u(N))]
    c += [("a,b in [1,100)", (a, b, p)) for a, b, p in zip(la(N, 1, 100), la(N, 1, 100), u(N))]
    c += [("a,b in [100,3000)", (a, b, p)) for a, b, p in zip(la(N, 100, 3000), la(N, 100, 1000), u(N))]
    c += [("a,b in [3000,1e15]", (a, b, p)) for a, b, p in zip(la(NQ // 2, 3000, 1e15), la(NQ // 2, 3000, 1e15), u(NQ // 2))]
    c += [("a small, b large", (a, b, p) if r.random() < 0.5 else (b, a, p))
          for a, b, p in zip(la(NQ, 1e-3, 10), la(NQ, 3000, 1e15), u(NQ))]
    c += [("p tiny", (a, b, p)) for a, b, p in zip(la(N, 0.1, 100), la(N, 0.1, 100), la(N, 1e-300, 1e-3))]
    c += [("p near 1", (a, b, 1 - p)) for a, b, p in zip(la(N, 0.1, 100), la(N, 0.1, 100), la(N, 1.2e-16, 1e-3))]
    return c


FIXTURES = {
    "gammainc": lambda: (2, gam_cases("gammainc"), _robust(gammainc_mp), sc.gammainc, "P(a,x), regularised lower incomplete gamma"),
    "gammaincc": lambda: (2, gam_cases("gammaincc"), _robust(gammaincc_mp), sc.gammaincc, "Q(a,x), regularised upper incomplete gamma"),
    "gammaincinv": lambda: (2, gaminv_cases("gammaincinv", None), _with_guess(gammaincinv_mp, sc.gammaincinv),
                            sc.gammaincinv, "x with P(a,x) = p"),
    "gammainccinv": lambda: (2, gaminv_cases("gammainccinv", None), _with_guess(gammainccinv_mp, sc.gammainccinv),
                             sc.gammainccinv, "x with Q(a,x) = q"),
    "betainc": lambda: (3, beta_cases("betainc"), _robust(betainc_mp), sc.betainc, "I_x(a,b), regularised incomplete beta"),
    "betaincc": lambda: (3, beta_cases("betaincc"), _robust(betaincc_mp), sc.betaincc, "1 - I_x(a,b)"),
    "betaincinv": lambda: (3, betainv_cases("betaincinv"), _with_guess(betaincinv_mp, sc.betaincinv),
                           sc.betaincinv, "x with I_x(a,b) = p"),
    "betainccinv": lambda: (3, betainv_cases("betainccinv"), _with_guess(betainccinv_mp, sc.betainccinv),
                            sc.betainccinv, "x with 1 - I_x(a,b) = q"),
}
