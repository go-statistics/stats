"""SciPy parity specs: Weibull (minimum) distribution."""

import numpy as np

_r = np.random.default_rng(109)
nan = float("nan")


def _d(c, scale, n, loc=0.0):
    return [float(v) for v in _r.weibull(c, n) * scale + loc]


SPECS = [
    dict(scipy="weibull_min", go="Weibull", kind="continuous", shapes=["C"],
         instances=[{"C": 1.5}, {"C": 0.5, "Loc": 1.0, "Scale": 3.0}, {"C": 5.0, "Loc": -2.0, "Scale": 0.1},
                    {"C": 1.0}, {"C": 2.0, "Scale": 10.0}],
         invalid=[{"C": 0.0}, {"C": -1.0}, {"C": float("inf")}, {"C": 1.0, "Scale": 0.0}],
         fit_kind="mle",
         fit=[{"data": _d(2.0, 3.0, 200), "fixed": {"Loc": 0.0}},
              {"data": _d(1.5, 1.0, 300, loc=2.0)},
              {"data": _d(3.0, 2.0, 100), "fixed": {"Loc": 0.0, "C": 3.0}},
              {"data": _d(0.8, 1.0, 150), "fixed": {"Loc": 0.0, "Scale": 1.0}}]),
]
