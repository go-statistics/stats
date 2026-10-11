"""SciPy parity specs: half-normal distribution."""

import numpy as np

_r = np.random.default_rng(113)
nan = float("nan")

SPECS = [
    dict(scipy="halfnorm", go="HalfNormal", kind="continuous", shapes=[],
         instances=[{}, {"Loc": -2.0, "Scale": 0.5}, {"Loc": 1e3, "Scale": 37.0}],
         invalid=[{"Scale": 0.0}, {"Scale": -1.0}, {"Loc": nan}],
         fit_kind="exact",
         fit=[{"data": list(np.abs(_r.normal(0, 2.0, 50)) + 1.0)},
              {"data": list(np.abs(_r.normal(0, 0.1, 1000)) - 3.0)},
              {"data": list(np.abs(_r.normal(0, 1.0, 20))), "fixed": {"Loc": 0.0}},
              {"data": list(np.abs(_r.normal(0, 3.0, 20)) + 5.0), "fixed": {"Scale": 3.0}}]),
]
