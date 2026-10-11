"""SciPy parity specs: von Mises distribution (scipy.stats.vonmises)."""

import mpmath
import numpy as np
import scipy.stats as st

_r = np.random.default_rng(115)
nan = float("nan")


def _moments(inst):
    """Exact moments over [Loc - π, Loc + π] (SciPy integrates them
    numerically: its mean at Loc = 0 is 7e-18, its skewness 1e-15)."""
    k = mpmath.mpf(inst["Kappa"])
    loc = mpmath.mpf(inst["Loc"])
    w = lambda t: mpmath.exp(k * (mpmath.cos(t) - 1))
    pts = [-mpmath.pi, 0, mpmath.pi]
    z = mpmath.quad(w, pts)
    # odd moments vanish by symmetry (exactly, not to quadrature noise)
    m = [mpmath.quad(lambda t: t ** n * w(t), pts) / z if n % 2 == 0 else mpmath.mpf(0) for n in range(5)]
    var = m[2]
    out = {"mean": loc, "var": var, "std": mpmath.sqrt(var), "skew": mpmath.mpf(0),
           "kurtosis": m[4] / var ** 2 - 3, "median": loc,
           "entropy": -mpmath.quad(lambda t: w(t) / z * mpmath.log(w(t) / z), pts)}
    for n in range(5):
        out["moment(%d)" % n] = sum(mpmath.binomial(n, j) * loc ** (n - j) * m[j] for j in range(n + 1))
    return out


def _d(kappa, n, loc=0.0):
    return [float(v) for v in st.vonmises.rvs(kappa, loc=loc, size=n, random_state=_r)]


SPECS = [
    dict(scipy="vonmises", go="VonMises", kind="continuous", shapes=["Kappa"],
         instances=[{"Kappa": 1.0}, {"Kappa": 0.5, "Loc": 1.0}, {"Kappa": 10.0, "Loc": -2.0},
                    {"Kappa": 0.0}, {"Kappa": 40.0}, {"Kappa": 4.0, "Loc": 0.75}],
         invalid=[{"Kappa": -1.0}, {"Kappa": float("inf")}, {"Kappa": 1.0, "Scale": 0.0}],
         mp_moments=_moments,
         fit_kind="mle",
         fit=[{"data": _d(2.0, 200, loc=0.5)},
              {"data": _d(20.0, 100, loc=-1.0)},
              {"data": _d(1.0, 150), "fixed": {"Loc": 0.0}},
              {"data": _d(5.0, 120, loc=2.0), "fixed": {"Kappa": 5.0}},
              {"data": _d(0.05, 50)}]),
]
