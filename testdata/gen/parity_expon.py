"""SciPy parity specs: exponential distribution."""

import numpy as np

_r = np.random.default_rng(101)
nan = float("nan")

SPECS = [
    dict(scipy="expon", go="Exponential", kind="continuous", shapes=[],
         instances=[{}, {"Loc": -2.0, "Scale": 0.5}, {"Loc": 1e3, "Scale": 37.0}],
         invalid=[{"Scale": 0.0}, {"Scale": -1.0}, {"Loc": nan}],
         fit_kind="exact",
         fit=[{"data": list(_r.exponential(2.0, 50) + 1.0)},
              {"data": list(_r.exponential(0.1, 1000) - 3.0)},
              {"data": list(_r.exponential(1.0, 20)), "fixed": {"Loc": 0.0}},
              {"data": list(_r.exponential(3.0, 20) + 5.0), "fixed": {"Scale": 3.0}}]),
]
