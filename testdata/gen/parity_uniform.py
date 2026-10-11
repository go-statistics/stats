"""SciPy parity specs: continuous uniform distribution."""

import numpy as np

_r = np.random.default_rng(102)
nan = float("nan")

SPECS = [
    dict(scipy="uniform", go="Uniform", kind="continuous", shapes=[],
         instances=[{}, {"Loc": -2.0, "Scale": 0.5}, {"Loc": 1e3, "Scale": 37.0}],
         invalid=[{"Scale": 0.0}, {"Scale": -1.0}, {"Loc": nan}],
         fit_kind="exact",
         fit=[{"data": list(_r.uniform(1.0, 3.0, 50))},
              {"data": list(_r.uniform(-3.0, -2.9, 1000))},
              {"data": list(_r.uniform(0.0, 1.0, 20)), "fixed": {"Loc": 0.0}},
              {"data": list(_r.uniform(5.0, 8.0, 20)), "fixed": {"Scale": 3.5}}]),
]
