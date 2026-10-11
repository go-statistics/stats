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


def main():
    cache = {}
    for line in sys.stdin:
        cmd = line.split()
        if not cmd or cmd[0] == "quit":
            break
        name, n = cmd[0], int(cmd[1])
        setup, call = CASES[name]
        if (name, n) not in cache:
            cache[(name, n)] = setup(n)
        args = cache[(name, n)]
        with np.errstate(all="ignore"):
            t0 = time.perf_counter()
            call(*args)
            dt = time.perf_counter() - t0
        print(repr(dt), flush=True)


if __name__ == "__main__":
    main()
