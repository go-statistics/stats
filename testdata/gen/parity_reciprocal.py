"""SciPy parity specs: reciprocal (log-uniform) distribution, under both
of SciPy's names."""

import mpmath
import numpy as np
import scipy.stats as st

_r = np.random.default_rng(103)
nan = float("nan")


def _lu(a, b, n, loc=0.0):
    return [float(v) + loc for v in st.loguniform.rvs(a, b, size=n, random_state=_r)]


def _moments(inst):
    """Exact moments: SciPy forms the central moments from raw ones and
    loses digits when B/A is close to 1."""
    a, b = mpmath.mpf(inst["A"]), mpmath.mpf(inst["B"])
    lg = mpmath.log(b / a)
    m = [None] + [(b ** k - a ** k) / (k * lg) for k in range(1, 5)]
    mu2 = m[2] - m[1] ** 2
    mu3 = m[3] - 3 * m[1] * m[2] + 2 * m[1] ** 3
    mu4 = m[4] - 4 * m[1] * m[3] + 6 * m[1] ** 2 * m[2] - 3 * m[1] ** 4
    sc = mpmath.mpf(inst["Scale"])
    return {"mean": m[1] * sc + inst["Loc"], "var": mu2 * sc * sc, "std": mpmath.sqrt(mu2) * sc,
            "skew": mu3 / mu2 ** 1.5, "kurtosis": mu4 / mu2 ** 2 - 3}


def spec(name):
    return dict(scipy=name, go="Reciprocal", kind="continuous", shapes=["A", "B"],
                instances=[{"A": 1e-3, "B": 1e3}, {"A": 0.5, "B": 2.0, "Loc": 1.0, "Scale": 3.0},
                           {"A": 1.0, "B": 1e10}, {"A": 2.0, "B": 2.5, "Loc": -4.0, "Scale": 0.5}],
                invalid=[{"A": 0.0, "B": 1.0}, {"A": 2.0, "B": 1.0}, {"A": -1.0, "B": 1.0}, {"A": 1.0, "B": 1.0}],
                mp_moments=_moments,
                fit_kind="mle",
                fit=[{"data": _lu(0.01, 10.0, 100), "fixed": {"Loc": 0.0}},
                     {"data": _lu(1.0, 50.0, 200), "fixed": {"Loc": 0.0, "A": 0.5}},
                     {"data": _lu(0.1, 5.0, 80, loc=2.0)}])


SPECS = [spec("reciprocal"), spec("loguniform")]
