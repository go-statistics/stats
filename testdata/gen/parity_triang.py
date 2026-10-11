"""SciPy parity specs: triangular distribution."""

import numpy as np

_r = np.random.default_rng(114)
nan = float("nan")


def _d(c, n, loc=0.0, scale=1.0):
    return [float(v) for v in _r.triangular(0, c, 1, n) * scale + loc]


SPECS = [
    dict(scipy="triang", go="Triangular", kind="continuous", shapes=["C"],
         instances=[{"C": 0.3}, {"C": 0.0, "Loc": 1.0, "Scale": 3.0}, {"C": 1.0, "Loc": -2.0, "Scale": 0.1},
                    {"C": 0.5}, {"C": 0.9, "Scale": 10.0}],
         invalid=[{"C": -0.1}, {"C": 1.5}, {"C": nan}, {"C": 0.5, "Scale": 0.0}],
         fit_kind="mle",
         fit=[{"data": _d(0.3, 200), "fixed": {"Loc": 0.0, "Scale": 1.0}},
              {"data": _d(0.6, 300, loc=2.0, scale=3.0)},
              {"data": _d(0.2, 100, scale=2.0), "fixed": {"C": 0.2}}]),
]
