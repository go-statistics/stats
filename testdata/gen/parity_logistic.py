"""SciPy parity specs: logistic distribution."""

import numpy as np

_r = np.random.default_rng(107)
nan = float("nan")

SPECS = [
    dict(scipy="logistic", go="Logistic", kind="continuous", shapes=[],
         instances=[{}, {"Loc": -2.0, "Scale": 0.5}, {"Loc": 1e3, "Scale": 37.0}],
         invalid=[{"Scale": 0.0}, {"Scale": -1.0}, {"Loc": nan}],
         fit_kind="mle",
         fit=[{"data": list(_r.logistic(1.0, 2.0, 50))},
              {"data": list(_r.logistic(-3.0, 0.1, 1000))},
              {"data": list(_r.logistic(0.0, 1.0, 20)), "fixed": {"Loc": 0.0}},
              {"data": list(_r.logistic(5.0, 3.0, 20)), "fixed": {"Scale": 3.0}}]),
]
