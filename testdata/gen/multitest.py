"""Fixtures for the multiple-testing corrections: testdata/scipy/multitest.json.gz.

Run inside a linux/amd64 container (macOS arm64 wheels use FMA), from the
repository root:

    docker run --rm --platform linux/amd64 -v "$PWD":/w -w /w \
        ghcr.io/astral-sh/uv:0.9.5-python3.12-bookworm-slim \
        uv run --with statsmodels==0.15.0 --with scipy==1.18.1 \
        --with numpy==2.5.4 python testdata/gen/multitest.py

(The Go side, multitest_oracle_test.go, can regenerate it with
STATS_REGEN_MULTITEST=1 go test -run TestMultiTestRegen, which runs exactly
this command.)

The file holds {"versions": {...}, "cases": [...]}, a case being

    fn       "multipletests", "fdc" (scipy.stats.false_discovery_control) or
             "npsum" (numpy.sum(1.0 / numpy.arange(1, n + 1)))
    p        the p-values (multipletests, fdc)
    alpha, method, maxiter   the arguments
    reject, pcorr, sidak, bonf   the outputs (floats as repr strings are not
             needed: JSON floats round-trip exactly)
    ties     true when p holds equal values with n > 1 (the reference's
             argsort is unstable, so only pcorr is compared there)
    error    true when the reference raised

The rng seed is fixed; the p-values include 0, 1, ties and tiny values.
"""

import gzip
import json
import os
import warnings

import numpy as np
import scipy
import scipy.stats as st
import statsmodels
from statsmodels.stats.multitest import multipletests

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.normpath(os.path.join(HERE, "..", "scipy", "multitest.json.gz"))

METHODS = [
    "bonferroni", "sidak", "holm-sidak", "holm", "simes-hochberg", "hommel",
    "fdr_bh", "fdr_by", "fdr_tsbh", "fdr_tsbky",
]
ALIASES = ["b", "s", "hs", "h", "sh", "ho", "fdr_i", "fdr_n", "fdr_2sbh",
           "fdr_twostage", "FDR_BH", "Holm"]
SIZES = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 15, 16, 17, 30, 50, 100, 129, 257, 600]
ALPHAS = [0.05, 0.1, 0.01, 0.5, 1.0, 0.0, 0.2]


def gen_p(rng, n, kind):
    if kind == "uniform":
        p = rng.uniform(size=n)
    elif kind == "mixed":
        p = np.where(rng.uniform(size=n) < 0.3, rng.uniform(size=n) * 1e-3, rng.uniform(size=n))
    elif kind == "tiny":
        p = 10.0 ** rng.uniform(-300, -1, size=n)
    elif kind == "edges":
        p = rng.choice([0.0, 1.0, 0.5, 1e-17, 0.999999], size=n)
    else:  # ties
        p = np.round(rng.uniform(size=n), 1)
    return p


def run(fn, *a, **k):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with np.errstate(all="ignore"):
                return fn(*a, **k), False
    except Exception:
        return None, True


def main():
    rng = np.random.default_rng(20261101)
    cases = []
    for n in SIZES:
        for kind in ["uniform", "mixed", "tiny", "edges", "ties"]:
            if n == 0 and kind != "uniform":
                continue
            p = gen_p(rng, n, kind)
            ties = bool(n > 1 and len(np.unique(p)) < n)
            pl = [float(v) for v in p]
            alphas = ALPHAS if kind in ("uniform", "mixed") else ALPHAS[:3]
            for alpha in alphas:
                for method in METHODS + (ALIASES if kind == "uniform" else []):
                    if method in ("hommel", "ho") and n > 300:
                        continue
                    two = method in ("fdr_tsbh", "fdr_tsbky", "fdr_2sbh", "fdr_twostage")
                    for maxiter in ([1, 0, -1, 3] if two else [1]):
                        out, err = run(multipletests, p, alpha=alpha, method=method, maxiter=maxiter)
                        c = {"fn": "multipletests", "p": pl, "alpha": alpha,
                             "method": method, "maxiter": maxiter, "ties": ties, "error": err}
                        if not err:
                            rj, pc, sidak, bonf = out
                            c.update(reject=[bool(v) for v in rj], pcorr=[float(v) for v in pc],
                                     sidak=None if sidak != sidak else float(sidak),
                                     bonf=None if bonf != bonf else float(bonf))
                        cases.append(c)
            for method in ["bh", "by", "BH"]:
                out, err = run(st.false_discovery_control, p, method=method)
                c = {"fn": "fdc", "p": pl, "method": method, "ties": ties, "error": err}
                if not err:
                    c.update(pcorr=[float(v) for v in out])
                cases.append(c)
    # the harmonic sums of fdr_by
    for n in list(range(0, 300)) + [511, 512, 1000, 1023, 1024, 4097, 100000]:
        s = float(np.sum(1.0 / np.arange(1, n + 1))) if n else 0.0
        cases.append({"fn": "npsum", "n": n, "value": s})
    # errors: out-of-range and NaN p-values for scipy
    for bad in ([0.5, 1.5], [-0.1, 0.5], [0.5, float("nan")]):
        _, err = run(st.false_discovery_control, np.array(bad))
        cases.append({"fn": "fdc_bad", "p": [None if v != v else v for v in bad], "error": err})
    doc = {"versions": {"statsmodels": statsmodels.__version__, "scipy": scipy.__version__,
                        "numpy": np.__version__}, "cases": cases}
    with gzip.GzipFile(OUT, "wb", mtime=0) as f:
        f.write(json.dumps(doc, separators=(",", ":")).encode())
    print(len(cases), "cases", doc["versions"])


main()
