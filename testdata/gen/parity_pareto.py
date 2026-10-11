"""SciPy parity specs: Pareto distribution."""

import math

import mpmath
import numpy as np

_r = np.random.default_rng(110)
nan = float("nan")


def _d(b, scale, n, loc=0.0):
    return [float(v) for v in (_r.pareto(b, n) + 1) * scale + loc]


def _raw(b, k):
    """E[Y^k]: b/(b-k), +inf for k <= 2 and nan for k = 3, 4 where it does
    not exist (the convention of SciPy's moments: an infinite mean or
    variance, an undefined skewness or kurtosis)."""
    if k == 0:
        return mpmath.mpf(1)
    if b > k:
        return b / (b - k)
    return mpmath.inf if k <= 2 else mpmath.nan


def _moments(inst):
    """Exact moments. SciPy's are numerical integrals for the higher ones
    (7000.0000004 for E[X³] = 7000) and finite nonsense where a moment
    does not exist (-8 for E[X²] at b = 1.5)."""
    b = mpmath.mpf(inst["B"])
    loc, sc = mpmath.mpf(inst["Loc"]), mpmath.mpf(inst["Scale"])
    out = {}
    for n in range(5):
        if n == 0:
            v = mpmath.mpf(1)
        elif loc == 0:
            v = sc ** n * _raw(b, n)
        else:
            v = sum(mpmath.binomial(n, k) * (sc / loc) ** k * _raw(b, k) for k in range(n + 1)) * loc ** n
        out["moment(%d)" % n] = v
    return out


SPECS = [
    dict(scipy="pareto", go="Pareto", kind="continuous", shapes=["B"],
         instances=[{"B": 5.0}, {"B": 1.5, "Loc": 1.0, "Scale": 3.0}, {"B": 2.5, "Loc": -2.0, "Scale": 0.1},
                    {"B": 0.8}, {"B": 3.5, "Scale": 10.0}, {"B": 50.0}],
         invalid=[{"B": 0.0}, {"B": -1.0}, {"B": float("inf")}, {"B": 1.0, "Scale": 0.0}],
         mp_moments=_moments,
         fit_kind="mle",
         fit=[{"data": _d(2.0, 3.0, 200), "fixed": {"Loc": 0.0}},
              {"data": _d(1.5, 1.0, 300, loc=2.0)},
              {"data": _d(3.0, 2.0, 100), "fixed": {"Scale": 2.0}},
              {"data": _d(0.8, 1.0, 150), "fixed": {"Loc": 0.0, "Scale": 1.0}},
              {"data": _d(4.0, 1.0, 150, loc=-1.0), "fixed": {"B": 4.0}},
              {"data": _d(4.0, 1.0, 150, loc=-1.0), "fixed": {"B": 4.0, "Loc": -1.0}}]),
]
