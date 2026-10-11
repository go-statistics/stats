"""SciPy parity specs: the discrete distributions.

(zipf instances keep A >= 2.5: SciPy's generic quantile sums the masses up to
the quantile, beyond memory for ppf(1 - 1e-6) at A = 1.5.)

SciPy computes the entropy of binom, poisson, nbinom and zipf by summing
the masses from the median outwards, at most 1000 of them and only until a
chunk of 32 sums below 3.2e-8: for wide or heavy-tailed distributions that
truncates the sum well above 1e-11 (zipf: 1e-7 relative). The entropy is
therefore held to mpmath's exact sum (mp_moments) for every instance.

Also held to mpmath: the third and fourth moments of the distributions
concentrated at one point (binom with n = 0 or p = 0, poisson with mu = 0,
randint with high = low + 1), where SciPy's skewness or kurtosis is NaN or
infinite (0/0, 1/0) and its _moment_from_stats turns that into a NaN moment
although the moment is the point's power; and zipf's variance, skewness and
kurtosis, which SciPy forms by differences of raw moments that cancel (six
digits lost at a = 20).

The nbinom instance n = 1000 uses p = 0.45, not 0.5: at p = 1/2, cdf(999) is
exactly 1/2, SciPy's rounded value falls below it and its median is 1000;
the exact median (and ours) is 999. Likewise geom uses p = 1.5e-3, not
1e-3, and randint high = 999983, not 1e6: SciPy's isf(q) is ppf(1 - q), and
where sf(k) rounds to exactly q (sf(1) = 1 - 0.001 = 0.999 for geom) the
rounding of 1 - q moves its answer one up from the smallest k with
sf(k) <= q, which is ours.
"""

import math

import mpmath
from mpmath import mpf

nan = float("nan")
inf = float("inf")


def _entropy_sum(logpmf, mode, lo, hi):
    """-sum p log p over [lo, hi] from the mode outwards, to 2^-200."""
    tot = mpf(0)
    eps = mpf(2) ** -200
    k = mode
    while k <= hi:
        lp = logpmf(k)
        t = -mpmath.exp(lp) * lp
        tot += t
        if lp < -1 and abs(t) < eps * tot:
            break
        k += 1
    k = mode - 1
    while k >= lo:
        lp = logpmf(k)
        t = -mpmath.exp(lp) * lp
        tot += t
        if lp < -1 and abs(t) < eps * tot:
            break
        k -= 1
    return tot


def _lbin(n, k):
    return mpmath.loggamma(n + 1) - mpmath.loggamma(k + 1) - mpmath.loggamma(n - k + 1)


def _point(v):
    """Moments of the distribution concentrated at v."""
    v = mpf(v)
    return {"entropy": mpf(0), **{"moment(%d)" % n: v ** n for n in range(1, 5)}}


def _binom_entropy(inst):
    n, p = mpf(inst["N"]), mpf(inst["P"])
    if p == 0 or n == 0:
        return _point(inst["Loc"])
    if p == 1:
        return {"entropy": mpf(0)}
    lp = lambda k: _lbin(n, k) + k * mpmath.log(p) + (n - k) * mpmath.log1p(-p)
    return {"entropy": _entropy_sum(lp, int(math.floor((float(n) + 1) * float(p))) if p < 1 else int(n), 0, int(n))}


def _bernoulli_entropy(inst):
    return _binom_entropy({"N": 1, "P": inst["P"], "Loc": inst["Loc"]})


def _poisson_entropy(inst):
    mu = mpf(inst["Mu"])
    if mu == 0:
        return _point(inst["Loc"])
    lp = lambda k: k * mpmath.log(mu) - mu - mpmath.loggamma(k + 1)
    return {"entropy": _entropy_sum(lp, int(math.floor(inst["Mu"])), 0, math.inf)}


def _nbinom_entropy(inst):
    n, p = mpf(inst["N"]), mpf(inst["P"])
    lp = lambda k: (mpmath.loggamma(n + k) - mpmath.loggamma(k + 1) - mpmath.loggamma(n)
                    + n * mpmath.log(p) + k * mpmath.log1p(-p))
    mode = int(math.floor((inst["N"] - 1) * (1 - inst["P"]) / inst["P"])) if inst["N"] > 1 else 0
    return {"entropy": _entropy_sum(lp, mode, 0, math.inf)}


def _hypergeom_entropy(inst):
    M, n, N = inst["Total"], inst["Good"], inst["Draws"]
    lo, hi = max(0, N - (M - n)), min(n, N)
    if lo == hi:
        return {"entropy": mpf(0)}
    lp = lambda k: _lbin(mpf(n), k) + _lbin(mpf(M - n), N - k) - _lbin(mpf(M), N)
    mode = min(hi, max(lo, (n + 1) * (N + 1) // (M + 2)))
    return {"entropy": _entropy_sum(lp, mode, lo, hi)}


def _zipf_entropy(inst):
    a = mpf(inst["A"])
    z = mpmath.zeta(a)
    dz = mpmath.zeta(a, 1, 1)
    out = {"entropy": mpmath.log(z) - a * dz / z}
    m = [mpmath.zeta(a - j) / z if a > j + 1 else None for j in range(5)]
    if m[2] is not None:
        mu = m[1] + inst["Loc"]
        v = m[2] - m[1] ** 2
        out.update(mean=mu, var=v, std=mpmath.sqrt(v))
    if m[3] is not None:
        mu3 = m[3] - 3 * m[1] * m[2] + 2 * m[1] ** 3
        out["skew"] = mu3 / v ** 1.5
    if m[4] is not None:
        mu4 = m[4] - 4 * m[1] * m[3] + 6 * m[1] ** 2 * m[2] - 3 * m[1] ** 4
        out["kurtosis"] = mu4 / v ** 2 - 3
    return out


def _randint_moments(inst):
    if inst["High"] - inst["Low"] == 1:
        return _point(inst["Low"] + inst["Loc"])
    return {}


SPECS = [
    dict(scipy="bernoulli", go="Bernoulli", kind="discrete", shapes=["P"],
         instances=[{"P": 0.3}, {"P": 0.999, "Loc": 2.0}, {"P": 0.0}, {"P": 1.0}, {"P": 1e-4}],
         invalid=[{"P": -0.1}, {"P": 1.5}, {"P": nan}, {"P": 0.5, "Loc": nan}],
         mp_moments=_bernoulli_entropy),
    dict(scipy="binom", go="Binomial", kind="discrete", shapes=["N", "P"],
         instances=[{"N": 10, "P": 0.3}, {"N": 1000, "P": 0.02, "Loc": -3.0}, {"N": 1, "P": 0.5},
                    {"N": 200000, "P": 0.5}, {"N": 0, "P": 0.4}, {"N": 50, "P": 0.999}, {"N": 7, "P": 0.0}],
         invalid=[{"N": -1, "P": 0.5}, {"N": 5, "P": 1.2}, {"N": 5, "P": nan}],
         mp_moments=_binom_entropy),
    dict(scipy="poisson", go="Poisson", kind="discrete", shapes=["Mu"],
         instances=[{"Mu": 3.5}, {"Mu": 0.01, "Loc": 1.0}, {"Mu": 1e4}, {"Mu": 0.0}, {"Mu": 250.0}],
         invalid=[{"Mu": -1.0}, {"Mu": nan}],
         mp_moments=_poisson_entropy),
    dict(scipy="geom", go="Geometric", kind="discrete", shapes=["P"],
         instances=[{"P": 0.3}, {"P": 1.5e-3, "Loc": -1.0}, {"P": 0.95}, {"P": 0.5}],
         invalid=[{"P": 0.0}, {"P": 1.5}, {"P": nan}]),
    dict(scipy="nbinom", go="NegativeBinomial", kind="discrete", shapes=["N", "P"],
         instances=[{"N": 5.0, "P": 0.4}, {"N": 0.5, "P": 0.1}, {"N": 30.5, "P": 0.97, "Loc": 2.0},
                    {"N": 1000.0, "P": 0.45}, {"N": 1.0, "P": 0.25}],
         invalid=[{"N": 0.0, "P": 0.5}, {"N": 2.0, "P": 0.0}, {"N": 2.0, "P": 1.5}],
         mp_moments=_nbinom_entropy),
    dict(scipy="hypergeom", go="Hypergeometric", kind="discrete", shapes=["Total", "Good", "Draws"],
         instances=[{"Total": 20, "Good": 7, "Draws": 12}, {"Total": 500, "Good": 100, "Draws": 60, "Loc": 1.0},
                    {"Total": 1000, "Good": 999, "Draws": 500}, {"Total": 100000, "Good": 40000, "Draws": 30000},
                    {"Total": 50, "Good": 30, "Draws": 45}, {"Total": 30, "Good": 25, "Draws": 3}],
         invalid=[{"Total": 5, "Good": 6, "Draws": 2}, {"Total": 5, "Good": 2, "Draws": 6},
                  {"Total": 0, "Good": 0, "Draws": 0}, {"Total": 5, "Good": -1, "Draws": 2}],
         mp_moments=_hypergeom_entropy),
    dict(scipy="randint", go="DiscreteUniform", kind="discrete", shapes=["Low", "High"],
         instances=[{"Low": 0, "High": 10}, {"Low": -5, "High": 3, "Loc": 2.0}, {"Low": 7, "High": 8},
                    {"Low": 0, "High": 999983}],
         invalid=[{"Low": 3, "High": 3}, {"Low": 4, "High": 2}],
         mp_moments=_randint_moments),
    dict(scipy="zipf", go="Zipf", kind="discrete", shapes=["A"],
         instances=[{"A": 3.5, "Loc": -1.0}, {"A": 6.0}, {"A": 20.0}, {"A": 2.5}],
         invalid=[{"A": 1.0}, {"A": 0.5}, {"A": nan}],
         mp_moments=_zipf_entropy),
]
