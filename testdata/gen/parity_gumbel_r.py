"""SciPy parity specs: Gumbel (right) distribution."""

import numpy as np

_r = np.random.default_rng(108)
nan = float("nan")

SPECS = [
    dict(scipy="gumbel_r", go="Gumbel", kind="continuous", shapes=[],
         instances=[{}, {"Loc": -2.0, "Scale": 0.5}, {"Loc": 1e3, "Scale": 37.0}],
         invalid=[{"Scale": 0.0}, {"Scale": -1.0}, {"Loc": nan}],
         fit_kind="mle",
         fit=[{"data": list(_r.gumbel(1.0, 2.0, 50))},
              {"data": list(_r.gumbel(-3.0, 0.1, 1000))},
              {"data": list(_r.gumbel(0.0, 1.0, 20)), "fixed": {"Loc": 0.0}},
              {"data": list(_r.gumbel(5.0, 3.0, 20)), "fixed": {"Scale": 3.0}}]),
]
