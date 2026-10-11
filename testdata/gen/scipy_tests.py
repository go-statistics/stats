"""SciPy fixtures for the hypothesis tests: testdata/scipy/tests_<group>.json.gz.

    uv run --python cpython-3.12-macos-x86_64-none --with mpmath==1.4.1 \
        --with scipy==1.18.1 --with numpy==2.5.4 \
        python testdata/gen/scipy_tests.py [group ...]

Each file holds {"cases": [...]}, a case being

    fn      the SciPy function (or null distribution) called
    data    the samples, as indices into the file's "datasets" (lists of
            floats, "nan", "inf", "-inf" as strings)
    opts    the options, with SciPy's names
    out     SciPy's outputs by field name (floats as above), absent when
            SciPy raised
    error   true when SciPy raised ValueError (or TypeError/IndexError
            for a single group)
    crash   true when SciPy failed with ZeroDivisionError (the Go function
            returns NaN there)

The Go harness (tests_scipy_test.go) replays every case through the Go
function and compares each field.
"""

import gzip
import json
import math
import os
import sys
import warnings

import numpy as np
import scipy
import scipy.special as sc
import scipy.stats as st

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.normpath(os.path.join(HERE, "..", "scipy"))
ALTS = ["two-sided", "less", "greater"]
LEVELS = [0.95, 0.9, 0.5, 0.999]


def f(x):
    x = float(x)
    if math.isnan(x):
        return "nan"
    if math.isinf(x):
        return "inf" if x > 0 else "-inf"
    return x


def fl(a):
    return [f(v) for v in np.asarray(a, dtype=float).ravel()]


def call(fn, *args, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return fn(*args, **kw), None
        except (ValueError, TypeError, IndexError) as e:
            # TypeError, IndexError: SciPy's way of refusing fewer than two
            # groups in f_oneway and kruskal.
            return None, e
        except ZeroDivisionError as e:
            return None, e


class Cases:
    def __init__(self):
        self.cases = []
        self.datasets = []
        self.index = {}

    def ds(self, d):
        key = np.asarray(d, dtype=float).tobytes()
        if key not in self.index:
            self.index[key] = len(self.datasets)
            self.datasets.append(fl(d))
        return self.index[key]

    def add(self, fn, data, opts, compute):
        """compute() returns a dict of outputs, or raises ValueError."""
        res, err = call(compute)
        c = {"fn": fn, "data": [self.ds(d) for d in data], "opts": opts}
        if isinstance(err, ZeroDivisionError):
            c["crash"] = True   # SciPy fails with ZeroDivisionError
        elif err is not None:
            c["error"] = True
        else:
            c["out"] = {k: (fl(v) if isinstance(v, (list, np.ndarray)) and np.ndim(v) > 0 else f(v))
                        for k, v in res.items()}
        self.cases.append(c)


def rng(seed):
    return np.random.default_rng(seed)


def samples_1():
    """One-sample datasets: (name, array)."""
    r = rng(1)
    out = []
    for n in [2, 3, 4, 5, 7, 8, 10, 13, 20, 50, 200, 1000]:
        out.append(r.normal(0.2, 1.5, n))
    out.append(r.standard_t(3, 30) * 10 + 100)
    out.append(r.exponential(2, 25))
    out.append(np.round(r.normal(0, 2, 40)))      # ties
    out.append(r.integers(0, 4, 15).astype(float))  # heavy ties
    out.append(np.full(6, 3.25))                  # constant
    out.append(np.array([1.0, 2.0, np.nan, 4.0]))  # NaN
    out.append(np.array([1.0]))
    out.append(np.array([]))
    out.append(r.normal(1e8, 1, 20))              # large offset
    return out


def samples_2():
    """Two-sample datasets: (a, b) pairs (not necessarily equal length)."""
    r = rng(2)
    out = []
    for n1, n2 in [(2, 2), (2, 3), (3, 5), (5, 5), (4, 9), (8, 8), (9, 12),
                   (10, 10), (20, 30), (50, 50), (100, 80), (300, 400)]:
        out.append((r.normal(0, 1, n1), r.normal(0.5, 2, n2)))
    out.append((np.round(r.normal(0, 2, 30)), np.round(r.normal(1, 2, 25))))
    out.append((np.full(5, 1.0), np.full(4, 1.0)))
    out.append((np.full(5, 1.0), np.array([1.0, 2.0, 3.0])))
    out.append((np.array([1.0, np.nan, 3.0]), np.array([1.0, 2.0])))
    out.append((np.array([1.0]), np.array([2.0, 3.0, 5.0])))
    out.append((np.array([]), np.array([2.0, 3.0])))
    return out


def paired():
    r = rng(3)
    out = []
    for n in [2, 3, 4, 5, 6, 7, 8, 10, 12, 13, 14, 20, 30, 50, 51, 100, 500]:
        x = r.normal(0, 1, n)
        out.append((x, 0.6 * x + r.normal(0.3, 1, n)))
    x = np.round(r.normal(0, 3, 25))
    out.append((x, np.round(x + r.normal(0.5, 1, 25))))       # ties and zeros
    x = np.round(r.normal(0, 3, 10))
    out.append((x, np.round(x + r.normal(0.5, 1, 10))))       # small, ties
    out.append((np.arange(8.0), np.arange(8.0)[::-1]))        # perfect -1
    out.append((np.arange(8.0), 2 * np.arange(8.0) + 1))      # perfect +1
    out.append((np.full(6, 2.0), np.arange(6.0)))            # constant x
    out.append((np.array([1.0, 2, np.nan, 4]), np.array([2.0, 1, 3, 5])))
    out.append((np.array([1.0]), np.array([2.0])))
    out.append((np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0])))  # mismatch
    out.append((np.array([0.0, 1, 2, 3]), np.array([0.0, 1, 2, 3])))  # all zero diffs
    return out


def tt_out(res, cis=True):
    o = {"statistic": res.statistic, "pvalue": res.pvalue, "df": res.df}
    if cis:
        for lv in LEVELS:
            ci = res.confidence_interval(lv)
            o[f"ci_lo_{lv}"] = ci.low
            o[f"ci_hi_{lv}"] = ci.high
    return o


def gen_ttest():
    c = Cases()
    for x in samples_1():
        for alt in ALTS:
            for pm in [0.0, 0.3]:
                c.add("ttest_1samp", [x], {"popmean": pm, "alternative": alt},
                      lambda: tt_out(st.ttest_1samp(x, pm, alternative=alt)))
    for a, b in samples_2():
        for alt in ALTS:
            for ev in [True, False]:
                for trim in [0.0, 0.2]:
                    c.add("ttest_ind", [a, b], {"equal_var": ev, "trim": trim, "alternative": alt},
                          lambda: tt_out(st.ttest_ind(a, b, equal_var=ev, trim=trim, alternative=alt)))
    for a, b in paired():
        for alt in ALTS:
            c.add("ttest_rel", [a, b], {"alternative": alt},
                  lambda: tt_out(st.ttest_rel(a, b, alternative=alt)))
    return c


LAMBDAS = {"pearson": 1, "log-likelihood": 0, "freeman-tukey": -0.5,
           "mod-log-likelihood": -1, "neyman": -2, "cressie-read": 2 / 3, "1.5": 1.5}


def gen_chisq():
    c = Cases()
    r = rng(4)
    obs_sets = [
        (np.array([16.0, 18, 16, 14, 12, 12]), None),
        (np.array([16.0, 18, 16, 14, 12, 12]), np.array([16.0, 16, 16, 16, 16, 8])),
        (np.array([43.0, 52, 54, 40]), np.array([47.0, 47, 47, 48])),
        (np.array([10.0, 0, 5, 3]), None),
        (np.array([10.0, 0, 5, 3]), np.array([5.0, 4, 5, 4])),
        (r.integers(20, 200, 30).astype(float), None),
        (np.array([5.0]), None),
        (np.array([1.0, 2, 3]), np.array([1.0, 2, 4])),   # sums differ: error
        (np.array([1.0, np.nan, 3]), None),
        (np.array([1e6, 1e6 + 1, 2e6]), None),
    ]
    for obs, exp in obs_sets:
        data = [obs] if exp is None else [obs, exp]
        for ddof in [0, 1]:
            c.add("chisquare", data, {"ddof": ddof},
                  lambda: dict(zip(["statistic", "pvalue"], st.chisquare(obs, exp, ddof=ddof))))
            for name, lam in LAMBDAS.items():
                c.add("power_divergence", data, {"ddof": ddof, "lambda_": lam},
                      lambda: dict(zip(["statistic", "pvalue"],
                                       st.power_divergence(obs, exp, ddof=ddof, lambda_=lam))))
    tables = [
        np.array([[12.0, 3], [17, 16]]),
        np.array([[10.0, 10, 20], [20, 20, 20]]),
        np.array([[5.0, 2, 7, 1], [3, 9, 4, 6], [8, 1, 2, 10]]),
        np.array([[[3.0, 4], [5, 6]], [[7.0, 2], [4, 9]]]),
        np.array([[1.0, 0], [0, 1]]),
        np.array([[100.0, 101], [99, 100]]),
        np.array([[4.0, 4], [4, 4]]),
        np.array([[1.0, 2, 3]]),          # dof 0
        np.array([[0.0, 0], [3, 4]]),     # zero expected: error
        np.array([[-1.0, 2], [3, 4]]),    # negative: error
        r.integers(0, 50, (4, 6)).astype(float),
    ]
    for t in tables:
        for corr in [True, False]:
            for lam in [None, 0, 2 / 3, -2]:
                def comp():
                    res = st.contingency.chi2_contingency(t, correction=corr, lambda_=lam)
                    return {"statistic": res.statistic, "pvalue": res.pvalue, "dof": res.dof,
                            "expected": res.expected_freq}
                c.add("chi2_contingency", [t.ravel(), np.array(t.shape, dtype=float)],
                      {"correction": corr, "lambda_": "none" if lam is None else lam}, comp)
    ftables = [[[8, 2], [1, 5]], [[1, 9], [11, 3]], [[0, 5], [5, 0]], [[3, 3], [3, 3]],
               [[0, 0], [3, 4]], [[2, 0], [5, 7]], [[100, 50], [40, 120]],
               [[1000, 980], [990, 1010]], [[20, 1], [1, 20]], [[7, 12], [0, 9]],
               [[1, 2], [3, 4]], [[50, 0], [0, 50]], [[3000, 2900], [2950, 3100]],
               [[5, 15], [25, 10]], [[0, 1], [1, 0]], [[12, 5], [29, 2]]]
    for t in ftables:
        for alt in ALTS:
            c.add("fisher_exact", [np.array(t, dtype=float).ravel()], {"alternative": alt},
                  lambda: dict(zip(["statistic", "pvalue"], st.fisher_exact(t, alternative=alt))))
    return c


def gen_corr():
    c = Cases()
    for a, b in paired():
        for alt in ALTS:
            def pr():
                res = st.pearsonr(a, b, alternative=alt)
                o = {"statistic": res.statistic, "pvalue": res.pvalue}
                for lv in LEVELS:
                    ci = res.confidence_interval(lv)
                    o[f"ci_lo_{lv}"] = ci.low
                    o[f"ci_hi_{lv}"] = ci.high
                return o
            c.add("pearsonr", [a, b], {"alternative": alt}, pr)
            c.add("spearmanr", [a, b], {"alternative": alt},
                  lambda: dict(zip(["statistic", "pvalue"], st.spearmanr(a, b, alternative=alt))))
            for variant in ["b", "c"]:
                for method in ["auto", "exact", "asymptotic"]:
                    c.add("kendalltau", [a, b], {"alternative": alt, "variant": variant, "method": method},
                          lambda: dict(zip(["statistic", "pvalue"],
                                           st.kendalltau(a, b, alternative=alt, variant=variant, method=method))))

            def lr():
                res = st.linregress(a, b, alternative=alt)
                return {"slope": res.slope, "intercept": res.intercept, "rvalue": res.rvalue,
                        "pvalue": res.pvalue, "stderr": res.stderr,
                        "intercept_stderr": res.intercept_stderr}
            c.add("linregress", [a, b], {"alternative": alt}, lr)
    # Kendall exact on larger untied samples (n up to 33, and beyond with
    # few discordant pairs), and the n >= 171 recursion.
    r = rng(5)
    for n in [20, 33, 34, 60, 171, 180]:
        x = r.permutation(n).astype(float)
        y = x.copy()
        y[:2] = y[1::-1]   # one discordant pair
        z = x + r.normal(0, n / 3, n)
        for yy in [y, z]:
            for alt in ALTS:
                for method in ["auto", "exact"]:
                    c.add("kendalltau", [x, yy], {"alternative": alt, "variant": "b", "method": method},
                          lambda: dict(zip(["statistic", "pvalue"],
                                           st.kendalltau(x, yy, alternative=alt, method=method))))
    for x in samples_1():
        for m in ["average", "min", "max", "dense", "ordinal"]:
            c.add("rankdata", [x], {"method": m}, lambda: {"ranks": st.rankdata(x, method=m)})
    return c


def gen_ks():
    c = Cases()
    # The null distributions on grids crossing every branch of _kolmogn.
    for n in [1, 2, 3, 5, 10, 20, 50, 100, 140, 141, 200, 500, 1000, 5000, 100000, 100001, 2000000]:
        ds = sorted(set([0.5 / n, 0.7 / n, 1.0 / n, 1.5 / n, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3,
                         0.45, 0.5, 0.6, 0.9, 1 - 1.0 / n, 1 - 0.5 / n,
                         math.sqrt(0.5 / n), math.sqrt(0.754 / n), math.sqrt(2 / n),
                         math.sqrt(3.9 / n), math.sqrt(4.1 / n), math.sqrt(10 / n), math.sqrt(20 / n),
                         math.sqrt(100 / n), math.sqrt(400 / n), (1.3 / n) ** (2 / 3)]))
        ds = [d for d in ds if 0 < d < 1]
        for d in ds:
            c.add("kstwo_sf", [np.array([n, d])], {}, lambda: {"sf": st.kstwo.sf(d, n),
                                                              "cdf": st.kstwo.cdf(d, n)})
            c.add("smirnov", [np.array([n, d])], {}, lambda: {"sf": sc.smirnov(n, d)})
    for x in np.concatenate([np.linspace(0.01, 0.2, 5), np.linspace(0.25, 3, 23), [0.82, 0.8200001, 5, 6.5]]):
        c.add("kolmogorov", [np.array([x])], {}, lambda: {"sf": sc.kolmogorov(x)})
    r = rng(6)
    sets = [r.normal(0, 1, n) for n in [1, 2, 3, 5, 10, 25, 100, 141, 500, 3000]]
    sets.append(r.normal(0.3, 1.2, 60))
    sets.append(np.round(r.normal(0, 1, 40), 1))
    sets.append(np.array([0.1, np.nan, 0.3]))
    for x in sets:
        for alt in ALTS:
            for method in ["auto", "exact", "asymp", "approx"]:
                if method == "approx" and alt != "two-sided":
                    continue

                def k1():
                    res = st.ks_1samp(x, st.norm.cdf, alternative=alt, method=method)
                    return {"statistic": res.statistic, "pvalue": res.pvalue,
                            "statistic_location": res.statistic_location,
                            "statistic_sign": res.statistic_sign}
                c.add("ks_1samp", [x], {"alternative": alt, "method": method, "cdf": "norm"}, k1)

                def k1t():
                    res = st.ks_1samp(x, st.t(4, 0.1, 1.3).cdf, alternative=alt, method=method)
                    return {"statistic": res.statistic, "pvalue": res.pvalue,
                            "statistic_location": res.statistic_location,
                            "statistic_sign": res.statistic_sign}
                c.add("ks_1samp", [x], {"alternative": alt, "method": method, "cdf": "t4"}, k1t)
    pairs2 = [(r.normal(0, 1, n1), r.normal(0.4, 1, n2)) for n1, n2 in
              [(1, 1), (1, 3), (3, 3), (5, 7), (10, 10), (8, 12), (20, 15), (50, 50), (60, 90),
               (200, 300), (1000, 1000), (10001, 30)]]
    pairs2.append((np.round(r.normal(0, 1, 30)), np.round(r.normal(0.5, 1, 20))))
    pairs2.append((np.arange(10.0), np.arange(10.0) + 100))
    pairs2.append((np.array([1.0, np.nan]), np.array([1.0, 2.0])))
    for a, b in pairs2:
        for alt in ALTS:
            for method in ["auto", "exact", "asymp"]:
                def k2():
                    res = st.ks_2samp(a, b, alternative=alt, method=method)
                    return {"statistic": res.statistic, "pvalue": res.pvalue,
                            "statistic_location": res.statistic_location,
                            "statistic_sign": res.statistic_sign}
                c.add("ks_2samp", [a, b], {"alternative": alt, "method": method}, k2)
    return c


def rank_out(res, z=False):
    o = {"statistic": res.statistic, "pvalue": res.pvalue}
    if z:
        o["zstatistic"] = res.zstatistic
    return o


def gen_rank():
    c = Cases()
    for a, b in samples_2():
        for alt in ALTS:
            for method in ["auto", "exact", "asymptotic"]:
                for cont in [True, False]:
                    if method != "asymptotic" and not cont:
                        continue
                    c.add("mannwhitneyu", [a, b], {"alternative": alt, "method": method, "use_continuity": cont},
                          lambda: rank_out(st.mannwhitneyu(a, b, use_continuity=cont, alternative=alt,
                                                           method=method), method == "asymptotic"))
    r = rng(7)
    for n1, n2 in [(3, 40), (8, 60), (12, 15), (25, 30)]:
        a, b = r.normal(0, 1, n1), r.normal(0.8, 1, n2)
        for alt in ALTS:
            for method in ["auto", "exact"]:
                c.add("mannwhitneyu", [a, b], {"alternative": alt, "method": method, "use_continuity": True},
                      lambda: rank_out(st.mannwhitneyu(a, b, alternative=alt, method=method)))
    wil = list(paired())
    x = np.array([0.0, 0, 1.5, -2.0, 3.0, 3.0, -1.0, 2.5, 0.5, -0.5, 4.0, 1.0])  # zeros and ties, n=12
    wil.append((x, np.zeros_like(x)))
    x2 = np.round(r.normal(0.5, 2, 60))
    wil.append((x2, np.zeros_like(x2)))
    x3 = r.normal(0.5, 1, 40)
    wil.append((x3, np.zeros_like(x3)))
    for a, b in wil:
        for alt in ALTS:
            for zm in ["wilcox", "pratt", "zsplit"]:
                for method in ["auto", "exact", "asymptotic"]:
                    for corr in [False, True]:
                        if method != "asymptotic" and corr:
                            continue
                        c.add("wilcoxon", [a, b], {"alternative": alt, "zero_method": zm, "method": method,
                                                   "correction": corr},
                              lambda: rank_out(st.wilcoxon(a, b, zero_method=zm, correction=corr,
                                                           alternative=alt, method=method),
                                               method == "asymptotic"))
        c.add("wilcoxon1", [a - b if len(a) == len(b) else a], {},
              lambda: rank_out(st.wilcoxon(a - b if len(a) == len(b) else a)))
    groups = [
        [r.normal(0, 1, 5), r.normal(0.5, 1, 7), r.normal(1, 1, 6)],
        [np.round(r.normal(0, 2, 20)), np.round(r.normal(1, 2, 25))],
        [r.normal(0, 1, 30), r.normal(0, 2, 40), r.normal(0.3, 1, 35), r.normal(0, 1, 2)],
        [np.array([1.0, 1, 1]), np.array([1.0, 1])],
        [np.array([1.0, 2, np.nan]), np.array([1.0, 2])],
        [np.array([1.0, 2, 3])],
        [np.array([]), np.array([1.0, 2])],
        [np.array([1.0, 1, 1]), np.array([2.0, 2])],
        [np.array([5.0]), np.array([6.0]), np.array([7.5])],
    ]
    for g in groups:
        c.add("kruskal", g, {}, lambda: dict(zip(["statistic", "pvalue"], st.kruskal(*g))))
    return c


def gen_norm():
    c = Cases()
    r = rng(8)
    sets = samples_1()
    sets += [r.normal(0, 1, n) for n in [3, 4, 5, 6, 9, 11, 12, 25, 100, 499, 5001]]
    sets += [r.exponential(1, 30), r.uniform(0, 1, 40), r.logistic(2, 3, 50), r.gumbel(1, 2, 45),
             -r.gumbel(1, 2, 45), np.array([1.0, 2, 4]), np.array([1.0, 2, 2.5])]
    for x in sets:
        c.add("shapiro", [x], {}, lambda: dict(zip(["statistic", "pvalue"], st.shapiro(x))))
        for alt in ALTS:
            c.add("skewtest", [x], {"alternative": alt},
                  lambda: dict(zip(["statistic", "pvalue"], st.skewtest(x, alternative=alt))))
            c.add("kurtosistest", [x], {"alternative": alt},
                  lambda: dict(zip(["statistic", "pvalue"], st.kurtosistest(x, alternative=alt))))
        c.add("normaltest", [x], {}, lambda: dict(zip(["statistic", "pvalue"], st.normaltest(x))))
        c.add("jarque_bera", [x], {}, lambda: dict(zip(["statistic", "pvalue"], st.jarque_bera(x))))
        if len(x) < 2 or (len(x) > 0 and np.all(x == x[0])) or np.any(np.isnan(x)):
            continue
        for dist in ["norm", "expon", "logistic", "gumbel_l", "gumbel_r"]:
            if dist == "expon" and np.any(x <= 0):
                continue

            def ad():
                res = st.anderson(x, dist)
                p = st.anderson(x, dist, method="interpolate").pvalue
                return {"statistic": res.statistic, "pvalue": p, "critical_values": res.critical_values,
                        "significance_level": res.significance_level,
                        "loc": res.fit_result.params[0], "scale": res.fit_result.params[1]}
            c.add("anderson", [x], {"dist": dist}, ad)
    return c


def gen_anova():
    c = Cases()
    r = rng(9)
    groups = [
        [r.normal(0, 1, 5), r.normal(0.5, 1, 7), r.normal(1, 1, 6)],
        [r.normal(0, 1, 30), r.normal(0, 2, 40), r.normal(0.3, 1, 35)],
        [r.normal(10, 1, 3), r.normal(10, 5, 4)],
        [np.round(r.normal(0, 2, 20)), np.round(r.normal(1, 2, 25)), np.round(r.normal(1, 3, 15))],
        [np.array([1.0, 1, 1]), np.array([2.0, 2])],
        [np.array([1.0, 1, 1]), np.array([1.0, 1])],
        [np.array([5.0]), np.array([6.0]), np.array([7.5])],
        [np.array([5.0]), np.array([6.0, 7.0])],
        [np.array([1.0, 2, np.nan]), np.array([1.0, 2])],
        [np.array([]), np.array([1.0, 2])],
        [np.array([1.0, 2, 3])],
        [r.normal(1e6, 1, 50), r.normal(1e6, 1, 50)],
    ]
    for g in groups:
        for ev in [True, False]:
            c.add("f_oneway", g, {"equal_var": ev},
                  lambda: dict(zip(["statistic", "pvalue"], st.f_oneway(*g, equal_var=ev))))
        c.add("bartlett", g, {}, lambda: dict(zip(["statistic", "pvalue"], st.bartlett(*g))))
        for center in ["median", "mean", "trimmed"]:
            for cut in [0.05, 0.2]:
                if center != "trimmed" and cut != 0.05:
                    continue
                c.add("levene", g, {"center": center, "proportiontocut": cut},
                      lambda: dict(zip(["statistic", "pvalue"],
                                       st.levene(*g, center=center, proportiontocut=cut))))
    return c


GROUPS = {
    "ttest": gen_ttest,
    "chisq": gen_chisq,
    "corr": gen_corr,
    "ks": gen_ks,
    "rank": gen_rank,
    "norm": gen_norm,
    "anova": gen_anova,
}


def main():
    names = sys.argv[1:] or list(GROUPS)
    for name in names:
        g = GROUPS[name]()
        cases = g.cases
        path = os.path.join(OUT, f"tests_{name}.json.gz")
        with gzip.open(path, "wt", compresslevel=9) as fh:
            json.dump({"scipy": scipy.__version__, "numpy": np.__version__,
                       "datasets": g.datasets, "cases": cases}, fh, separators=(",", ":"))
        print(f"{path}: {len(cases)} cases")


if __name__ == "__main__":
    main()
