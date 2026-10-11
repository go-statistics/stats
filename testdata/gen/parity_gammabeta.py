"""SciPy parity specs: the gamma and beta families, t and F."""

import mpmath
import numpy as np

_r = np.random.default_rng(11)
nan = float("nan")
inf = float("inf")


def _chi_moments(inst):
    """Exact chi moments: SciPy's formulas cancel catastrophically for large df."""
    df = mpmath.mpf(inst["Df"])
    k = df / 2
    mu = mpmath.sqrt(2) * mpmath.gamma(k + 0.5) / mpmath.gamma(k)
    mu2 = df - mu ** 2
    g1 = (2 * mu ** 3 + mu * (1 - 2 * df)) / mu2 ** 1.5
    g2 = (2 * df * (1 - df) - 6 * mu ** 4 + 4 * mu ** 2 * (2 * df - 1)) / mu2 ** 2
    sc = mpmath.mpf(inst["Scale"])
    return {"mean": mu * sc + inst["Loc"], "var": mu2 * sc * sc, "std": mpmath.sqrt(mu2) * sc,
            "skew": g1, "kurtosis": g2}


def _d(gen, *a, n=200):
    return [float(v) for v in gen(*a, n)]


SPECS = [
    dict(scipy="gamma", go="Gamma", kind="continuous", shapes=["A"],
         instances=[{"A": 0.5}, {"A": 2.5, "Loc": 1.0, "Scale": 3.0}, {"A": 47.0, "Scale": 0.1}, {"A": 1.0}],
         invalid=[{"A": 0.0}, {"A": -1.0}, {"A": 1.0, "Scale": 0.0}, {"A": inf}],
         fit_kind="mle",
         fit=[{"data": _d(_r.gamma, 2.0, 1.5)},
              {"data": _d(_r.gamma, 0.7, 3.0), "fixed": {"Loc": 0.0}},
              {"data": _d(_r.gamma, 5.0, 1.0), "fixed": {"Loc": 0.0, "A": 5.0}},
              {"data": _d(_r.gamma, 3.0, 2.0), "fixed": {"Loc": 0.0, "Scale": 2.0}}]),
    dict(scipy="chi2", go="ChiSquared", kind="continuous", shapes=["Df"],
         instances=[{"Df": 1.0}, {"Df": 3.0, "Loc": -1.0, "Scale": 2.0}, {"Df": 250.0}, {"Df": 2.0}],
         invalid=[{"Df": 0.0}, {"Df": -2.0}],
         fit_kind="mle",
         fit=[{"data": _d(_r.chisquare, 4.0), "fixed": {"Loc": 0.0, "Scale": 1.0}},
              {"data": _d(_r.chisquare, 7.0), "fixed": {"Loc": 0.0}}]),
    dict(scipy="chi", go="Chi", kind="continuous", shapes=["Df"],
         instances=[{"Df": 1.0}, {"Df": 3.0, "Loc": 0.5, "Scale": 2.0}, {"Df": 400.0}],
         invalid=[{"Df": 0.0}],
         mp_moments=_chi_moments,
         fit_kind="mle",
         fit=[{"data": [float(v) for v in np.sqrt(_r.chisquare(3.0, 200))], "fixed": {"Loc": 0.0, "Scale": 1.0}},
              {"data": [float(v) for v in np.sqrt(_r.chisquare(5.0, 200))], "fixed": {"Loc": 0.0}}]),
    dict(scipy="invgamma", go="InverseGamma", kind="continuous", shapes=["A"],
         instances=[{"A": 0.8}, {"A": 4.5, "Loc": 1.0, "Scale": 2.0}, {"A": 300.0}],
         invalid=[{"A": 0.0}],
         fit_kind="mle",
         fit=[{"data": [float(1 / v) for v in _r.gamma(3.0, 1.0, 200)], "fixed": {"Loc": 0.0}}]),
    dict(scipy="beta", go="Beta", kind="continuous", shapes=["A", "B"],
         instances=[{"A": 0.5, "B": 0.5}, {"A": 2.0, "B": 5.0, "Loc": -1.0, "Scale": 3.0}, {"A": 1.0, "B": 3.0},
                    {"A": 300.0, "B": 40.0}, {"A": 0.2, "B": 80.0}],
         invalid=[{"A": 0.0, "B": 1.0}, {"A": 1.0, "B": -1.0}],
         fit_kind="mle",
         fit=[{"data": _d(_r.beta, 2.0, 5.0), "fixed": {"Loc": 0.0, "Scale": 1.0}},
              {"data": _d(_r.beta, 0.6, 0.9), "fixed": {"Loc": 0.0, "Scale": 1.0}},
              {"data": _d(_r.beta, 3.0, 2.0), "fixed": {"Loc": 0.0, "Scale": 1.0, "A": 3.0}},
              {"data": _d(_r.beta, 3.0, 2.0), "fixed": {"Loc": 0.0, "Scale": 1.0, "B": 2.0}}]),
    dict(scipy="t", go="StudentT", kind="continuous", shapes=["Df"],
         instances=[{"Df": 1.0}, {"Df": 2.5, "Loc": 1.0, "Scale": 2.0}, {"Df": 30.0}, {"Df": 1e4},
                    {"Df": 4.5}, {"Df": inf}],
         invalid=[{"Df": 0.0}, {"Df": -1.0}],
         fit_kind="mle",
         fit=[{"data": _d(_r.standard_t, 5.0)},
              {"data": _d(_r.standard_t, 3.0), "fixed": {"Loc": 0.0, "Scale": 1.0}}]),
    dict(scipy="f", go="F", kind="continuous", shapes=["Dfn", "Dfd"],
         instances=[{"Dfn": 1.0, "Dfd": 1.0}, {"Dfn": 2.0, "Dfd": 9.0, "Loc": 0.5, "Scale": 2.0},
                    {"Dfn": 5.0, "Dfd": 20.0}, {"Dfn": 300.0, "Dfd": 500.0}, {"Dfn": 2.0, "Dfd": 3.0}],
         invalid=[{"Dfn": 0.0, "Dfd": 1.0}, {"Dfn": 1.0, "Dfd": 0.0}],
         fit_kind="mle",
         fit=[{"data": _d(_r.f, 5.0, 12.0), "fixed": {"Loc": 0.0, "Scale": 1.0}}]),
]
