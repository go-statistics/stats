"""SciPy parity specs: Laplace distribution."""

import numpy as np

_r = np.random.default_rng(106)
nan = float("nan")

SPECS = [
    dict(scipy="laplace", go="Laplace", kind="continuous", shapes=[],
         instances=[{}, {"Loc": -2.0, "Scale": 0.5}, {"Loc": 1e3, "Scale": 37.0}],
         invalid=[{"Scale": 0.0}, {"Scale": -1.0}, {"Loc": nan}],
         fit_kind="exact",
         fit=[{"data": list(_r.laplace(1.0, 2.0, 51))},
              {"data": list(_r.laplace(-3.0, 0.1, 1000))},
              {"data": list(_r.laplace(0.0, 1.0, 20)), "fixed": {"Loc": 0.0}},
              {"data": list(_r.laplace(5.0, 3.0, 21)), "fixed": {"Scale": 3.0}}]),
]
