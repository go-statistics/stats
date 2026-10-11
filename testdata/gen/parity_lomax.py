"""SciPy parity specs: Lomax distribution."""

import mpmath
import numpy as np

_r = np.random.default_rng(111)
nan = float("nan")


def _raw(c, k):
    """E[Y^k] = k!/((c-1)...(c-k)); +inf for k <= 2 and nan for k = 3, 4
    where it does not exist (SciPy's convention for the moments)."""
    if k == 0:
        return mpmath.mpf(1)
    if c > k:
        r = mpmath.mpf(1)
        for j in range(1, k + 1):
            r *= j / (c - j)
        return r
    return mpmath.inf if k <= 2 else mpmath.nan


def _moments(inst):
    """Exact moments (SciPy's higher ones are numerical, or nonsense where
    they do not exist)."""
    c = mpmath.mpf(inst["C"])
    loc, sc = mpmath.mpf(inst["Loc"]), mpmath.mpf(inst["Scale"])
    out = {}
    for n in range(1, 5):
        if loc == 0:
            v = sc ** n * _raw(c, n)
        else:
            v = sum(mpmath.binomial(n, k) * (sc / loc) ** k * _raw(c, k) for k in range(n + 1)) * loc ** n
        out["moment(%d)" % n] = v
    return out


def _d(c, scale, n, loc=0.0):
    return [float(v) for v in _r.pareto(c, n) * scale + loc]


SPECS = [
    dict(scipy="lomax", go="Lomax", kind="continuous", shapes=["C"],
         instances=[{"C": 5.0}, {"C": 1.5, "Loc": 1.0, "Scale": 3.0}, {"C": 2.5, "Loc": -2.0, "Scale": 0.1},
                    {"C": 0.8}, {"C": 3.5, "Scale": 10.0}, {"C": 50.0}],
         invalid=[{"C": 0.0}, {"C": -1.0}, {"C": float("inf")}, {"C": 1.0, "Scale": 0.0}],
         mp_moments=_moments,
         fit_kind="mle",
         fit=[{"data": _d(3.0, 2.0, 200), "fixed": {"Loc": 0.0}},
              {"data": _d(2.0, 1.0, 300, loc=1.0)},
              {"data": _d(4.0, 2.0, 100), "fixed": {"Loc": 0.0, "Scale": 2.0}}]),
]
