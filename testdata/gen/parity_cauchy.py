"""SciPy parity specs: Cauchy distribution."""

import numpy as np

_r = np.random.default_rng(105)
nan = float("nan")

SPECS = [
    dict(scipy="cauchy", go="Cauchy", kind="continuous", shapes=[],
         instances=[{}, {"Loc": -2.0, "Scale": 0.5}, {"Loc": 1e3, "Scale": 37.0}],
         invalid=[{"Scale": 0.0}, {"Scale": -1.0}, {"Loc": nan}],
         fit_kind="mle",
         fit=[{"data": list(_r.standard_cauchy(100) * 2.0 + 1.0)},
              {"data": list(_r.standard_cauchy(500) * 0.1 - 3.0)},
              {"data": list(_r.standard_cauchy(50)), "fixed": {"Loc": 0.0}},
              {"data": list(_r.standard_cauchy(50) * 3.0 + 5.0), "fixed": {"Scale": 3.0}}]),
]
