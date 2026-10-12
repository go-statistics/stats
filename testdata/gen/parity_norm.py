"""SciPy parity specs: normal distribution."""

import numpy as np

_r = np.random.default_rng(7)

SPECS = [
    dict(scipy="norm", go="Normal", kind="continuous", shapes=[],
         instances=[{}, {"Loc": -3.5, "Scale": 0.25}, {"Loc": 1e3, "Scale": 37.0}],
         invalid=[{"Scale": 0.0}, {"Scale": -1.0}, {"Loc": float("nan")}],
         fit_kind="exact",
         fit=[{"data": list(_r.normal(2.0, 3.0, 50))},
              {"data": list(_r.normal(-1.0, 0.1, 1000))},
              {"data": list(_r.normal(0.0, 1.0, 20)), "fixed": {"Loc": 0.0}},
              {"data": list(_r.normal(5.0, 2.0, 20)), "fixed": {"Scale": 2.0}}]),
]
