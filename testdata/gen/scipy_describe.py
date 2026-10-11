"""NumPy/SciPy fixtures for the descriptive statistics:
testdata/scipy/describe_cases.json.gz.

    uv run --python cpython-3.12-macos-x86_64-none --with mpmath==1.4.1 \
        --with scipy==1.18.1 --with numpy==2.5.4 \
        python testdata/gen/scipy_describe.py

The x86_64 build is the reference (NumPy's arm64 macOS wheel differs from
it in the last bit of some libm calls).

The datasets are drawn from numpy.random.default_rng(seed).random(...) and
transformed with + - * / and floor only, so that the Go test regenerates
them bit for bit with go-ndarray's random package (a SHA-256 of the bytes
checks it); the small special datasets are stored inline.

Each case names a function, a dataset, its arguments, and the result
NumPy/SciPy return ("error": true when they raise). "exact" marks results
NumPy computes itself (pairwise sums, selection, linspace): ours must equal
them to the bit. The others are compared to a relative tolerance and the
Go test reports the largest difference per function. For the moments, "mp"
holds the exact value (mpmath, 120 digits) of the same statistic, to tell
which of SciPy and us is closer where the two differ.
"""

import gzip
import hashlib
import json
import math
import os
import warnings

import mpmath as mp
import numpy as np
import scipy
import scipy.stats as st

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.normpath(os.path.join(HERE, "..", "scipy", "describe_cases.json.gz"))

mp.mp.dps = 120
warnings.simplefilter("ignore")


def f(x):
    x = float(x)
    if math.isnan(x):
        return "nan"
    if math.isinf(x):
        return "inf" if x > 0 else "-inf"
    return x


def fl(v):
    return [f(t) for t in np.asarray(v, dtype=float).ravel()]


# ---------------------------------------------------------------- datasets

def gen(spec):
    """The dataset of a spec; the Go test mirrors this function."""
    if "values" in spec:
        return np.array([float(t) for t in spec["values"]], dtype=float)
    n, seed, kind = spec["n"], spec["seed"], spec["kind"]
    rng = np.random.default_rng(seed)
    scale, shift = spec.get("scale", 1.0), spec.get("shift", 0.0)
    if kind == "unif":
        u = rng.random(n)
        x = u * scale + shift
    elif kind == "ih":  # Irwin-Hall(4) - 2: bell shaped
        u = rng.random((n, 4))
        s = u[:, 0] + u[:, 1]
        s = s + u[:, 2]
        s = s + u[:, 3]
        x = (s - 2.0) * scale + shift
    elif kind == "ties":
        u = rng.random(n)
        x = np.floor(u * spec["k"]) * scale + shift
    elif kind == "heavy":  # 1/(u + c), random sign
        u = rng.random((n, 2))
        x = 1.0 / (u[:, 0] + spec["c"])
        x = np.where(u[:, 1] < 0.5, -x, x) * scale + shift
    elif kind == "skewed":  # u^2 * u' (products only): right skewed, positive
        u = rng.random((n, 2))
        x = (u[:, 0] * u[:, 0]) * u[:, 1] * scale + shift
    elif kind == "const":
        x = np.full(n, spec["c"])
    else:
        raise ValueError(kind)
    x = np.array(x, dtype=float)
    for i in spec.get("nan_at", []):
        x[i] = np.nan
    return x


DATASETS = {}


def ds(name, **spec):
    DATASETS[name] = spec


# inline special datasets
ds("one", values=[3.5])
ds("two", values=[1.0, 2.0])
ds("three", values=[3.0, -1.0, 2.0])
ds("seven", values=[0.5, -2.0, 7.25, 3.0, 3.0, -0.0, 1e-3])
ds("signed_zero", values=[0.0, -0.0, -0.0, 0.0])
ds("with_inf", values=[1.0, "inf", 2.0, -3.0, 4.0, "-inf", 0.5])
ds("with_nan_small", values=[1.0, 2.0, "nan", 4.0, 3.0])
ds("all_nan", values=["nan", "nan"])
# random datasets: the sizes around the pairwise-sum block boundaries
for n in [1, 2, 3, 7, 8, 9, 127, 128, 129, 1000]:
    ds(f"ih{n}", kind="ih", n=n, seed=100 + n)
for n in [8, 9, 129, 1000]:
    ds(f"unif{n}", kind="unif", n=n, seed=200 + n, scale=10.0, shift=-3.0)
    ds(f"ties{n}", kind="ties", n=n, seed=300 + n, k=5, scale=0.5, shift=1.0)
ds("big", kind="ih", n=100000, seed=7, scale=3.0, shift=1.5)
ds("bigties", kind="ties", n=100000, seed=8, k=37, scale=0.25, shift=-2.0)
ds("bigpos", kind="skewed", n=100000, seed=9, scale=2.0, shift=0.0)
ds("const9", kind="const", n=9, seed=0, c=2.5)
ds("const130", kind="const", n=130, seed=0, c=-0.1)
ds("neg129", kind="unif", n=129, seed=11, scale=5.0, shift=-10.0)
ds("huge", kind="ih", n=129, seed=12, scale=1e300, shift=0.0)
ds("tiny", kind="ih", n=129, seed=13, scale=1e-300, shift=0.0)
ds("offset", kind="ih", n=1000, seed=14, scale=1e-3, shift=1e8)  # cancellation
ds("heavy1000", kind="heavy", n=1000, seed=15, c=1e-3)
ds("pos129", kind="skewed", n=129, seed=16, scale=3.0, shift=0.0)
ds("pos1000", kind="unif", n=1000, seed=17, scale=1.0, shift=0.001)
ds("nan129", kind="ih", n=129, seed=18, nan_at=[0, 17, 64, 128])
ds("nan1000", kind="unif", n=1000, seed=19, scale=2.0, nan_at=[5, 500, 501])
ds("weights8", kind="unif", n=8, seed=401)
ds("weights9", kind="unif", n=9, seed=402)
ds("weights129", kind="unif", n=129, seed=403)
ds("weights1000", kind="unif", n=1000, seed=404)
ds("weights100000", kind="unif", n=100000, seed=405)
ds("intweights1000", kind="ties", n=1000, seed=406, k=4)

DATA = {k: gen(v) for k, v in DATASETS.items()}
NUMERIC = [k for k in DATA if not k.startswith(("weights", "intweights"))]
FINITE = [k for k in NUMERIC if np.all(np.isfinite(DATA[k]))]
POSITIVE = [k for k in FINITE if np.all(DATA[k] > 0)]
WEIGHTS = {8: "weights8", 9: "weights9", 129: "weights129", 1000: "weights1000",
           100000: "weights100000"}

CASES = []


def case(fn, data, args, func, exact=False, mpval=None):
    try:
        r = func()
    except Exception as e:  # noqa: BLE001
        CASES.append({"fn": fn, "data": data, "args": args, "error": True,
                      "msg": type(e).__name__ + ": " + str(e)[:80]})
        return
    c = {"fn": fn, "data": data, "args": args, "exact": exact}
    if isinstance(r, dict):
        c["want"] = r
    elif np.ndim(r) == 0:
        c["want"] = f(r)
    else:
        c["want"] = fl(r)
    if mpval is not None:
        c["mp"] = mpval
    CASES.append(c)


# ------------------------------------------------------------- exact moments

def mp_stats(x):
    v = [mp.mpf(float(t)) for t in x]
    n = len(v)
    m = mp.fsum(v) / n
    d = [t - m for t in v]
    mom = {k: mp.fsum(t ** k for t in d) / n for k in range(2, 7)}
    out = {"mean": f(m), "m2": f(mom[2])}
    for k in range(2, 7):
        out[f"m{k}"] = f(mom[k])
    if mom[2] != 0:
        out["skew"] = f(mom[3] / mom[2] ** mp.mpf(1.5))
        out["kurt"] = f(mom[4] / mom[2] ** 2 - 3)
    return out


QS = [0.0, 1e-9, 0.01, 0.1, 0.25, 1 / 3, 0.5, 0.6, 0.75, 0.9, 0.99, 1 - 1e-12, 1.0]
PS = [0.0, 1.0, 12.5, 25.0, 33.3, 50.0, 66.6, 75.0, 99.9, 100.0]
METHODS = ["linear", "inverted_cdf", "averaged_inverted_cdf", "closest_observation",
           "interpolated_inverted_cdf", "hazen", "weibull", "median_unbiased",
           "normal_unbiased", "lower", "higher", "midpoint", "nearest"]


def numpy_cases(name):
    x = DATA[name]
    n = len(x)
    w = DATA[WEIGHTS[n]] if n in WEIGHTS else None
    case("mean", name, {}, lambda: np.mean(x), True)
    case("nanmean", name, {}, lambda: np.nanmean(x), True)
    for ddof in [0, 1, 2.5]:
        case("var", name, {"ddof": ddof}, lambda: np.var(x, ddof=ddof), True)
        case("std", name, {"ddof": ddof}, lambda: np.std(x, ddof=ddof), True)
        case("nanvar", name, {"ddof": ddof}, lambda: np.nanvar(x, ddof=ddof), True)
    case("median", name, {}, lambda: np.median(x), True)
    case("nanmedian", name, {}, lambda: np.nanmedian(x), True)
    if w is not None:
        case("average", name, {"weights": WEIGHTS[n]},
             lambda: list(np.average(x, weights=w, returned=True)), True)
        case("quantile_w", name, {"weights": WEIGHTS[n], "q": QS},
             lambda: np.quantile(x, QS, method="inverted_cdf", weights=w), True)
    if name == "ties1000" or name == "bigties":
        iw = DATA["intweights1000"] if n == 1000 else np.floor(DATA["weights100000"] * 4)
        case("quantile_w", name, {"weights": "int", "q": QS},
             lambda: np.quantile(x, QS, method="inverted_cdf", weights=iw), True)
    for m in METHODS:
        case("quantile", name, {"method": m, "q": QS},
             lambda: np.quantile(x, QS, method=m), True)
        case("percentile", name, {"method": m, "q": PS},
             lambda: np.percentile(x, PS, method=m), True)
        case("quantile1", name, {"method": m, "q": 0.37},
             lambda: np.quantile(x, 0.37, method=m), True)
        if np.isnan(x).any():
            case("nanquantile", name, {"method": m, "q": QS},
                 lambda: np.nanquantile(x, QS, method=m), True)
    # histograms
    hopts = [{"bins": 10}, {"bins": 7, "range": [-1.0, 2.0]}, {"bins": 1},
             {"bins": 33, "density": True},
             {"edges": [-3.0, -1.0, 0.0, 0.5, 2.0, 2.0, 10.0]},
             {"edges": [-3.0, -1.0, 0.0, 0.5, 2.0, 10.0], "density": True}]
    for rule in ["auto", "fd", "doane", "scott", "stone", "rice", "sturges", "sqrt"]:
        hopts.append({"rule": rule})
        hopts.append({"rule": rule, "range": [-0.5, 1.5]})
    if w is not None:
        hopts += [{"bins": 10, "weights": WEIGHTS[n]},
                  {"bins": 12, "weights": WEIGHTS[n], "density": True, "range": [-1.0, 1.0]},
                  {"edges": [-3.0, -1.0, 0.0, 0.5, 2.0, 10.0], "weights": WEIGHTS[n]}]
    for h in hopts:
        kw = {}
        if "edges" in h:
            kw["bins"] = np.array(h["edges"])
        elif "rule" in h:
            kw["bins"] = h["rule"]
        else:
            kw["bins"] = h["bins"]
        if "range" in h:
            kw["range"] = tuple(h["range"])
        if "weights" in h:
            kw["weights"] = DATA[h["weights"]]
        kwd = dict(kw)
        if h.get("density"):
            kwd["density"] = True
        case("histogram", name, h,
             lambda: dict(zip(["counts", "edges"], map(fl, np.histogram(x, **kwd)))), True)
        if "rule" in h:
            case("histogram_bin_edges", name, h,
                 lambda: np.histogram_bin_edges(x, **kw), True)


def scipy_cases(name):
    x = DATA[name]
    n = len(x)
    finite = name in FINITE
    mpv = mp_stats(x) if finite and n <= 1000 and n > 1 else None
    for ddof in [1, 0]:
        for bias in [True, False]:
            case("describe", name, {"ddof": ddof, "bias": bias},
                 lambda: dict(zip(["nobs", "min", "max", "mean", "variance", "skewness", "kurtosis"],
                                  [f(t) for t in (lambda r: [r.nobs, r.minmax[0], r.minmax[1], r.mean,
                                                             r.variance, r.skewness, r.kurtosis])(
                                      st.describe(x, ddof=ddof, bias=bias))])),
                 mpval=mpv)
    case("describe_omit", name, {}, lambda: dict(zip(
        ["nobs", "min", "max", "mean", "variance", "skewness", "kurtosis"],
        [f(t) for t in (lambda r: [r.nobs, r.minmax[0], r.minmax[1], r.mean, r.variance,
                                   r.skewness, r.kurtosis])(st.describe(x, nan_policy="omit"))])))
    for bias in [True, False]:
        case("skew", name, {"bias": bias}, lambda: st.skew(x, bias=bias), mpval=mpv)
        for fisher in [True, False]:
            case("kurtosis", name, {"bias": bias, "fisher": fisher},
                 lambda: st.kurtosis(x, bias=bias, fisher=fisher), mpval=mpv)
    case("skew_omit", name, {}, lambda: st.skew(x, nan_policy="omit"))
    for order in range(0, 7):
        case("moment", name, {"order": order}, lambda: st.moment(x, order=order), mpval=mpv)
    case("moment", name, {"order": 3, "center": 1.25}, lambda: st.moment(x, order=3, center=1.25))
    case("moment", name, {"order": 1, "center": -0.5}, lambda: st.moment(x, order=1, center=-0.5))
    case("mode", name, {}, lambda: (lambda r: {"mode": f(r.mode), "count": f(r.count)})(st.mode(x)))
    case("mode_omit", name, {}, lambda: (lambda r: {"mode": f(r.mode), "count": f(r.count)})(
        st.mode(x, nan_policy="omit")))
    for ddof in [0, 1]:
        case("sem", name, {"ddof": ddof}, lambda: st.sem(x, ddof=ddof))
        case("variation", name, {"ddof": ddof}, lambda: st.variation(x, ddof=ddof))
        if n <= 1000:
            case("zscore", name, {"ddof": ddof}, lambda: st.zscore(x, ddof=ddof))
            case("zscore_omit", name, {"ddof": ddof}, lambda: st.zscore(x, ddof=ddof, nan_policy="omit"))
            sc = np.array([-1.0, 0.0, 0.5, 2.25, 1e3])
            case("zmap", name, {"ddof": ddof}, lambda: st.zmap(sc, x, ddof=ddof))
    case("sem_omit", name, {}, lambda: st.sem(x, nan_policy="omit"))
    case("variation_omit", name, {}, lambda: st.variation(x, nan_policy="omit"))
    for m in METHODS:
        interp = m
        case("iqr", name, {"method": m}, lambda: st.iqr(x, interpolation=interp))
    case("iqr", name, {"rng": [90, 10], "scale": "normal"}, lambda: st.iqr(x, rng=(90, 10), scale="normal"))
    case("iqr", name, {"rng": [0, 100], "scale": 2.0}, lambda: st.iqr(x, rng=(0, 100), scale=2.0))
    case("iqr_omit", name, {}, lambda: st.iqr(x, nan_policy="omit"))
    case("mad", name, {}, lambda: st.median_abs_deviation(x))
    case("mad", name, {"scale": "normal"}, lambda: st.median_abs_deviation(x, scale="normal"))
    case("mad", name, {"center": "mean"}, lambda: st.median_abs_deviation(x, center=np.mean))
    case("mad_omit", name, {}, lambda: st.median_abs_deviation(x, nan_policy="omit"))
    # trimmed statistics: limits at fixed values and at quantiles of the data
    if finite and n > 0:
        lo, hi = float(np.quantile(x, 0.2, method="lower")), float(np.quantile(x, 0.8, method="higher"))
    else:
        lo, hi = -0.5, 0.5
    lims = [None, [lo, hi], [-0.5, 0.5]]
    for lim in lims:
        for inc in [[True, True], [False, True], [True, False]]:
            if lim is None and inc != [True, True]:
                continue
            a = {"limits": lim, "inclusive": inc}
            L = None if lim is None else tuple(lim)
            I = tuple(inc)
            case("tmean", name, a, lambda: st.tmean(x, L, I))
            for ddof in [1, 0]:
                a2 = dict(a, ddof=ddof)
                case("tvar", name, a2, lambda: st.tvar(x, L, I, ddof=ddof))
                case("tstd", name, a2, lambda: st.tstd(x, L, I, ddof=ddof))
                case("tsem", name, a2, lambda: st.tsem(x, L, I, ddof=ddof))
    for lim in [None, lo, 0.0]:
        for inc in [True, False]:
            case("tmin", name, {"limit": lim, "inclusive": inc}, lambda: st.tmin(x, lim, inclusive=inc))
            case("tmax", name, {"limit": lim, "inclusive": inc}, lambda: st.tmax(x, lim, inclusive=inc))
    for p in [0.0, 0.1, 0.25, 0.4, 0.5]:
        case("trim_mean", name, {"p": p}, lambda: st.trim_mean(x, p))
        if n <= 129:
            case("trimboth", name, {"p": p}, lambda: np.sort(st.trimboth(x, p)))
            for tail in ["right", "left"]:
                case("trim1", name, {"p": p, "tail": tail}, lambda: np.sort(st.trim1(x, p, tail=tail)))
    # means of positive data
    if name in POSITIVE or name in ("with_nan_small", "nan1000", "three", "signed_zero"):
        w = DATA[WEIGHTS[n]] if n in WEIGHTS else None
        for wn in ([None, WEIGHTS[n]] if w is not None else [None]):
            ww = None if wn is None else DATA[wn]
            case("gmean", name, {"weights": wn}, lambda: st.gmean(x, weights=ww))
            case("hmean", name, {"weights": wn}, lambda: st.hmean(x, weights=ww))
            for p in [-2.0, -1.0, 0.5, 1e-7, -2e-6, 1.5, 3.0, 0.0]:
                case("pmean", name, {"p": p, "weights": wn}, lambda: st.pmean(x, p, weights=ww))
        case("gmean_omit", name, {}, lambda: st.gmean(x, nan_policy="omit"))
        qk = x[::-1].copy()

        def mp_entropy(p, q=None, base=None):
            if not (np.all(np.isfinite(p)) and np.all(p >= 0) and np.sum(p) > 0):
                return None
            P = [mp.mpf(float(t)) for t in p]
            sp = mp.fsum(P)
            if q is None:
                r = -mp.fsum((t / sp) * mp.log(t / sp) for t in P if t > 0)
            else:
                Q = [mp.mpf(float(t)) for t in q]
                sq = mp.fsum(Q)
                if sq == 0 or any(b == 0 and a > 0 for a, b in zip(P, Q)):
                    return None
                r = mp.fsum((a / sp) * mp.log((a / sp) / (b / sq)) for a, b in zip(P, Q) if a > 0)
            if base is not None:
                r /= mp.log(base)
            return {"value": f(r)}
        case("entropy", name, {}, lambda: st.entropy(x), mpval=mp_entropy(x))
        case("entropy", name, {"base": 2.0}, lambda: st.entropy(x, base=2.0), mpval=mp_entropy(x, None, 2))
        case("entropy", name, {"qk": "reversed"}, lambda: st.entropy(x, qk), mpval=mp_entropy(x, qk))
        case("entropy", name, {"qk": "reversed", "base": 10.0}, lambda: st.entropy(x, qk, base=10.0),
             mpval=mp_entropy(x, qk, 10))
    if finite:
        for meth in ["auto", "vasicek", "van es", "ebrahimi", "correa"]:
            if meth == "correa" and n > 1000:
                continue
            case("differential_entropy", name, {"method": meth},
                 lambda: st.differential_entropy(x, method=meth))
            case("differential_entropy", name, {"method": meth, "window_length": 2, "base": 2.0},
                 lambda: st.differential_entropy(x, method=meth, window_length=2, base=2.0))
    if finite:
        w = DATA[WEIGHTS[n]] if n in WEIGHTS else None
        for nb in [4, 10]:
            for lim in [None, [-1.0, 2.0]]:
                for wn in ([None, WEIGHTS[n]] if w is not None else [None]):
                    ww = None if wn is None else DATA[wn]
                    a = {"numbins": nb, "limits": lim, "weights": wn}
                    L = None if lim is None else tuple(lim)
                    case("cumfreq", name, a, lambda: (lambda r: {
                        "freq": fl(r.cumcount), "lower": f(r.lowerlimit), "binsize": f(r.binsize),
                        "extra": f(r.extrapoints)})(st.cumfreq(x, nb, L, ww)))
                    case("relfreq", name, a, lambda: (lambda r: {
                        "freq": fl(r.frequency), "lower": f(r.lowerlimit), "binsize": f(r.binsize),
                        "extra": f(r.extrapoints)})(st.relfreq(x, nb, L, ww)))
    if finite and n >= 2:
        vals = x * 2.0 + 1.0
        for stat in ["mean", "std", "median", "count", "sum", "min", "max"]:
            for bins in [5, [-1.0, 0.0, 0.5, 3.0]]:
                case("binned_statistic", name, {"statistic": stat, "bins": bins}, lambda: (lambda r: {
                    "statistic": fl(r.statistic), "edges": fl(r.bin_edges),
                    "binnumber": fl(r.binnumber) if n <= 129 else []})(st.binned_statistic(x, vals, statistic=stat, bins=bins)))


def main():
    for name in NUMERIC:
        numpy_cases(name)
        scipy_cases(name)
    specs = {}
    for k, v in DATASETS.items():
        s = dict(v)
        if "values" in s:
            s["values"] = [f(float(t)) for t in s["values"]]
        s["sha256"] = hashlib.sha256(DATA[k].astype("<f8").tobytes()).hexdigest()
        specs[k] = s
    doc = {"numpy": np.__version__, "scipy": scipy.__version__, "datasets": specs, "cases": CASES}
    with gzip.open(OUT, "wt", compresslevel=9) as fh:
        json.dump(doc, fh, separators=(",", ":"))
    print(f"{len(CASES)} cases -> {OUT}")


if __name__ == "__main__":
    main()
