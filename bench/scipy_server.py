"""SciPy side of the speed comparison (driven by perf_test.go).

Reads one command per line on stdin, "<case> <n>", runs the SciPy call of
that case on data generated exactly as perf_test.go generates it, and prints
the elapsed wall time in seconds (allocation of the result included, as on
the Go side). "quit" ends. Data: x_i = lo + (hi - lo) * frac(i * 0.6180339887498949)
(the same IEEE operations in both languages), or the seeded NumPy generator
where a case says so (go-ndarray's random reproduces it).
"""

import sys
import time

import warnings

import numpy as np
import scipy.stats as st

GOLD = 0.6180339887498949


def points(n, lo, hi):
    i = np.arange(n, dtype=np.float64)
    return lo + (hi - lo) * np.mod(i * GOLD, 1.0)


def normal(n, seed):
    return np.random.default_rng(seed).standard_normal(n)


# case -> (setup(n) -> args, call(*args)); must match perfCases in perf_test.go
CASES = {
    "norm.pdf": (lambda n: (points(n, -10, 10),), lambda x: st.norm.pdf(x)),
    "norm.cdf": (lambda n: (points(n, -10, 10),), lambda x: st.norm.cdf(x)),
    "norm.ppf": (lambda n: (points(n, 1e-12, 1 - 1e-12),), lambda p: st.norm.ppf(p)),
    "beta.pdf": (lambda n: (points(n, 1e-6, 1 - 1e-6),), lambda x: st.beta.pdf(x, 2.5, 7.0)),
    "beta.cdf": (lambda n: (points(n, 1e-6, 1 - 1e-6),), lambda x: st.beta.cdf(x, 2.5, 7.0)),
    "beta.ppf": (lambda n: (points(n, 1e-9, 1 - 1e-9),), lambda p: st.beta.ppf(p, 2.5, 7.0)),
    "gamma.pdf": (lambda n: (points(n, 1e-3, 30),), lambda x: st.gamma.pdf(x, 4.5)),
    "gamma.cdf": (lambda n: (points(n, 1e-3, 30),), lambda x: st.gamma.cdf(x, 4.5)),
    "gamma.ppf": (lambda n: (points(n, 1e-9, 1 - 1e-9),), lambda p: st.gamma.ppf(p, 4.5)),
    "t.cdf": (lambda n: (points(n, -20, 20),), lambda x: st.t.cdf(x, 7.5)),
    "t.ppf": (lambda n: (points(n, 1e-9, 1 - 1e-9),), lambda p: st.t.ppf(p, 7.5)),
    "chi2.sf": (lambda n: (points(n, 1e-3, 100),), lambda x: st.chi2.sf(x, 12.0)),
    "norm.rvs": (lambda n: (n,), lambda n: st.norm.rvs(size=n, random_state=np.random.default_rng(1))),
    "gamma.rvs": (lambda n: (n,), lambda n: st.gamma.rvs(2.5, size=n, random_state=np.random.default_rng(1))),
    "ttest_ind": (lambda n: (normal(n, 1), normal(n, 2) + 0.01), lambda a, b: st.ttest_ind(a, b)),
    "ttest_1samp": (lambda n: (normal(n, 1),), lambda a: st.ttest_1samp(a, 0.0)),
    "spearmanr": (lambda n: (normal(n, 1), normal(n, 2)), lambda a, b: st.spearmanr(a, b)),
    "pearsonr": (lambda n: (normal(n, 1), normal(n, 2)), lambda a, b: st.pearsonr(a, b)),
    "kde.evaluate": (lambda n: (normal(n, 1), points(n, -4, 4)), lambda d, x: st.gaussian_kde(d)(x)),
    "describe": (lambda n: (normal(n, 1),), lambda a: st.describe(a)),
    "percentile": (lambda n: (normal(n, 1),), lambda a: np.percentile(a, [5, 25, 50, 75, 95])),
}


def positive(n, seed):
    return np.abs(normal(n, seed)) + 0.5


def counts(n):
    return np.floor(points(n, 50, 150))


MV_MEAN = np.array([0, 1, -1, 0.5, 2.0])
MV_COV = np.array([
    [2, 0.5, 0.3, 0.1, 0],
    [0.5, 1.5, 0.2, 0, 0.1],
    [0.3, 0.2, 1, 0.4, 0.2],
    [0.1, 0, 0.4, 2.5, 0.3],
    [0, 0.1, 0.2, 0.3, 1.2],
])
ALPHA = np.array([1.5, 2, 3.5, 0.8, 4])
MN_P = np.array([0.1, 0.2, 0.3, 0.15, 0.25])


def mvn(n):
    d = st.multivariate_normal(MV_MEAN, MV_COV)
    return d, d.rvs(size=n, random_state=np.random.default_rng(8))


def kde2d(n):
    return (np.vstack([normal(n, 1), normal(n, 2)]), np.vstack([points(n, -4, 4), points(n, -3, 3)]))


LIM = (-1, 1)
N1 = lambda n: (normal(n, 1),)
N2 = lambda n: (normal(n, 1), normal(n, 2))
N3 = lambda n: (normal(n, 1), normal(n, 2), normal(n, 3))
CASES.update({
    "mean": (N1, np.mean),
    "var": (N1, np.var),
    "std": (N1, np.std),
    "median": (N1, np.median),
    "quantile": (N1, lambda a: np.quantile(a, 0.3)),
    "average": (lambda n: (normal(n, 1), positive(n, 2)), lambda a, w: np.average(a, weights=w)),
    "moment": (N1, lambda a: st.moment(a, 3)),
    "skew": (N1, st.skew),
    "kurtosis": (N1, st.kurtosis),
    "sem": (N1, st.sem),
    "variation": (lambda n: (positive(n, 1),), st.variation),
    "zscore": (N1, st.zscore),
    "zmap": (N2, st.zmap),
    "gmean": (lambda n: (positive(n, 1),), st.gmean),
    "hmean": (lambda n: (positive(n, 1),), st.hmean),
    "pmean": (lambda n: (positive(n, 1),), lambda a: st.pmean(a, 2.5)),
    "iqr": (N1, st.iqr),
    "mad": (N1, st.median_abs_deviation),
    "mode": (lambda n: (np.round(normal(n, 1) * 10),), st.mode),
    "tmean": (N1, lambda a: st.tmean(a, LIM)),
    "tvar": (N1, lambda a: st.tvar(a, LIM)),
    "tstd": (N1, lambda a: st.tstd(a, LIM)),
    "tsem": (N1, lambda a: st.tsem(a, LIM)),
    "tmin": (N1, lambda a: st.tmin(a, -1)),
    "tmax": (N1, lambda a: st.tmax(a, 1)),
    "trim_mean": (N1, lambda a: st.trim_mean(a, 0.1)),
    "trimboth": (N1, lambda a: st.trimboth(a, 0.1)),
    "trim1": (N1, lambda a: st.trim1(a, 0.1, "right")),
    "rankdata": (N1, st.rankdata),
    "histogram": (N1, lambda a: np.histogram(a, 100)),
    "histogram_bin_edges": (N1, lambda a: np.histogram_bin_edges(a, "auto")),
    "cumfreq": (N1, lambda a: st.cumfreq(a, 100)),
    "relfreq": (N1, lambda a: st.relfreq(a, 100)),
    "binned_statistic": (N2, lambda a, b: st.binned_statistic(a, b, "mean", 50)),
    "entropy": (lambda n: (positive(n, 1),), st.entropy),
    "differential_entropy": (N1, st.differential_entropy),
    "ttest_rel": (N2, st.ttest_rel),
    "linregress": (N2, st.linregress),
    "kendalltau": (N2, st.kendalltau),
    "mannwhitneyu": (N2, st.mannwhitneyu),
    "wilcoxon": (N2, st.wilcoxon),
    "kruskal": (N3, st.kruskal),
    "f_oneway": (N3, st.f_oneway),
    "bartlett": (N3, st.bartlett),
    "levene": (N3, st.levene),
    "ks_1samp": (N1, lambda a: st.ks_1samp(a, st.norm.cdf)),
    "ks_2samp": (N2, st.ks_2samp),
    "shapiro": (N1, st.shapiro),
    "anderson": (N1, lambda a: st.anderson(a, "norm")),
    "skewtest": (N1, st.skewtest),
    "kurtosistest": (N1, st.kurtosistest),
    "normaltest": (N1, st.normaltest),
    "jarque_bera": (N1, st.jarque_bera),
    "chisquare": (lambda n: (counts(n),), st.chisquare),
    "power_divergence": (lambda n: (counts(n),), lambda a: st.power_divergence(a, lambda_="cressie-read")),
    "chi2_contingency": (lambda n: (counts(n).reshape(100, n // 100),), st.chi2_contingency),
    "fisher_exact": (lambda n: (), lambda: st.fisher_exact([[1200, 800], [900, 1100]])),
    "bootstrap": (N1, lambda a: st.bootstrap((a,), np.mean, n_resamples=9999, rng=np.random.default_rng(3))),
    "kde.logpdf": (lambda n: (normal(n, 1), points(n, -4, 4)), lambda d, x: st.gaussian_kde(d).logpdf(x)),
    "kde.evaluate2d": (kde2d, lambda d, x: st.gaussian_kde(d)(x)),
    "kde.resample": (lambda n: (st.gaussian_kde(normal(10_000, 1)), n), lambda k, n: k.resample(n, seed=np.random.default_rng(4))),
    "kde.integrate_box_1d": (lambda n: (st.gaussian_kde(normal(n, 1)),), lambda k: k.integrate_box_1d(-1, 0.5)),
    "mvn.logpdf": (mvn, lambda d, x: d.logpdf(x)),
    "mvn.rvs": (lambda n: (mvn(1)[0], n), lambda d, n: d.rvs(size=n, random_state=np.random.default_rng(5))),
    "mvn.cdf": (lambda n: (mvn(1)[0],), lambda d: d.cdf([0.5, 0.2, 1, -0.3, 0.8])),
    "mvn.fit": (lambda n: (mvn(n)[1],), st.multivariate_normal.fit),
    "dirichlet.logpdf": (lambda n: (st.dirichlet.rvs(ALPHA, size=n, random_state=np.random.default_rng(6)).T,), lambda x: st.dirichlet.logpdf(x, ALPHA)),
    "dirichlet.rvs": (lambda n: (n,), lambda n: st.dirichlet.rvs(ALPHA, size=n, random_state=np.random.default_rng(6))),
    "multinomial.logpmf": (lambda n: (st.multinomial.rvs(20, MN_P, size=n, random_state=np.random.default_rng(6)),), lambda x: st.multinomial.logpmf(x, 20, MN_P)),
    "multinomial.rvs": (lambda n: (n,), lambda n: st.multinomial.rvs(20, MN_P, size=n, random_state=np.random.default_rng(6))),
})


def shapes(s):
    if s == "-":
        return ()
    v = [float(t) for t in s.split(",")]
    return tuple(int(t) if t == int(t) else t for t in v)


def generic(cmd):
    """The "@" commands: (cache key, setup, call) of a generic distribution case."""
    kind, name = cmd[0], cmd[1]
    dist = getattr(st, name)
    n = int(cmd[-1])
    if kind == "@dist":
        method, lo, hi, disc, args = cmd[2], float(cmd[3]), float(cmd[4]), cmd[5] == "1", shapes(cmd[6])
        def setup():
            x = points(n, lo, hi)
            return (np.floor(x) if disc else x,)
        f = getattr(dist, method)
        return setup, lambda x: f(x, *args)
    args = shapes(cmd[2])
    if kind == "@rvs":
        return (lambda: ()), lambda: dist.rvs(*args, size=n, random_state=np.random.default_rng(1))
    if kind == "@entropy":
        def ent():
            for _ in range(100):
                dist(*args).entropy()
        return (lambda: ()), ent
    if kind == "@fit":
        return (lambda: (dist.rvs(*args, size=n, random_state=np.random.default_rng(7)),)), dist.fit
    raise ValueError(kind)


def main():
    warnings.simplefilter("ignore")
    cache = {}
    for line in sys.stdin:
        cmd = line.split()
        if not cmd or cmd[0] == "quit":
            break
        key = tuple(cmd)
        if cmd[0].startswith("@"):
            setup, call = generic(cmd)
            if key not in cache:
                cache[key] = setup()
        else:
            name, n = cmd[0], int(cmd[1])
            setup, call = CASES[name]
            if key not in cache:
                cache[key] = setup(n)
        args = cache[key]
        with np.errstate(all="ignore"):
            t0 = time.perf_counter()
            try:
                call(*args)
                dt = time.perf_counter() - t0
            except Exception:  # SciPy cannot compute this case: NaN
                dt = float("nan")
        print(repr(dt), flush=True)


if __name__ == "__main__":
    main()
