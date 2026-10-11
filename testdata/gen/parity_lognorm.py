"""SciPy parity specs: log-normal distribution."""

import mpmath
import numpy as np

_r = np.random.default_rng(104)
nan = float("nan")


def _moments(inst):
    """Exact moments: SciPy's exp(s²) - 1 cancels for small s."""
    s = mpmath.mpf(inst["S"])
    e = mpmath.expm1(s * s)
    p = 1 + e
    sc = mpmath.mpf(inst["Scale"])
    mu = mpmath.sqrt(p)
    return {"mean": mu * sc + inst["Loc"], "var": p * e * sc * sc, "std": mpmath.sqrt(p * e) * sc,
            "skew": mpmath.sqrt(e) * (2 + p), "kurtosis": p ** 4 + 2 * p ** 3 + 3 * p ** 2 - 6}


def _d(s, scale, n, loc=0.0):
    return [float(v) for v in np.exp(_r.normal(0.0, s, n)) * scale + loc]


SPECS = [
    dict(scipy="lognorm", go="LogNormal", kind="continuous", shapes=["S"],
         instances=[{"S": 1.0}, {"S": 0.5, "Loc": 1.0, "Scale": 3.0}, {"S": 2.5, "Loc": -2.0, "Scale": 0.1},
                    {"S": 0.05}],
         invalid=[{"S": 0.0}, {"S": -1.0}, {"S": float("inf")}, {"S": 1.0, "Scale": 0.0}],
         mp_moments=_moments,
         fit_kind="mle",
         fit=[{"data": _d(0.7, 2.0, 100), "fixed": {"Loc": 0.0}},
              {"data": _d(1.5, 0.3, 300), "fixed": {"Loc": -1.0}},
              {"data": _d(0.4, 5.0, 200, loc=3.0)},
              {"data": _d(0.9, 1.0, 150, loc=-1.0)},
              {"data": _d(0.8, 2.0, 60), "fixed": {"Loc": 0.0, "Scale": 2.0}},
              {"data": _d(0.6, 1.0, 120), "fixed": {"S": 0.6}},
              {"data": _d(0.6, 1.0, 120), "fixed": {"Loc": 0.0, "S": 0.6}}]),
]
