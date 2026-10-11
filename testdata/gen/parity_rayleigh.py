"""SciPy parity specs: Rayleigh distribution."""

import numpy as np

_r = np.random.default_rng(112)
nan = float("nan")

SPECS = [
    dict(scipy="rayleigh", go="Rayleigh", kind="continuous", shapes=[],
         instances=[{}, {"Loc": -2.0, "Scale": 0.5}, {"Loc": 1e3, "Scale": 37.0}],
         invalid=[{"Scale": 0.0}, {"Scale": -1.0}, {"Loc": nan}],
         fit_kind="mle",
         fit=[{"data": list(_r.rayleigh(2.0, 50) + 1.0)},
              {"data": list(_r.rayleigh(0.1, 1000) - 3.0)},
              {"data": list(_r.rayleigh(1.0, 20)), "fixed": {"Loc": 0.0}},
              {"data": list(_r.rayleigh(3.0, 200) + 5.0), "fixed": {"Scale": 3.0}}]),
]
