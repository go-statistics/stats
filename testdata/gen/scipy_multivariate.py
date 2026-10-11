"""SciPy parity fixtures for the multivariate distributions, the Gaussian
kernel density estimate and the bootstrap: testdata/scipy/mv_<name>.json.gz.

    uv run --python cpython-3.12-macos-x86_64-none --with mpmath==1.4.1 \
        --with scipy==1.18.1 --with numpy==2.5.4 \
        python testdata/gen/scipy_multivariate.py [mvn dirichlet multinomial kde bootstrap]

Every float is stored as its exact hexadecimal form (float.hex()). Where an
exact reference is cheap (multivariate normal log-densities, bivariate
normal probabilities, Dirichlet and multinomial log-probabilities) it is
computed with mpmath at the float64 arguments and stored next to SciPy's
value, so the tests can measure both against the truth.

Random draws use np.random.default_rng(seed); the x86_64 build of NumPy is
the reference (go-ndarray reproduces it bit for bit).
"""

import gzip
import json
import math
import os
import sys
import warnings

import mpmath as mp
import numpy as np
import scipy
import scipy.stats as st

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.normpath(os.path.join(HERE, "..", "scipy"))


def H(x):
    """Floats (nested lists, arrays) as exact hex strings."""
    if isinstance(x, np.ndarray) and x.ndim == 0:
        x = x[()]
    if isinstance(x, (list, tuple, np.ndarray)):
        return [H(v) for v in x]
    return float(x).hex()


def write(name, obj):
    obj = dict(obj, scipy_version=scipy.__version__, numpy_version=np.__version__)
    raw = json.dumps(obj, separators=(",", ":"), sort_keys=True).encode()
    path = os.path.join(OUT, "mv_%s.json.gz" % name)
    with open(path, "wb") as fh:
        with gzip.GzipFile(filename="", mode="wb", fileobj=fh, mtime=0, compresslevel=9) as gz:
            gz.write(raw)
    print("%-12s %7d bytes" % (name, os.path.getsize(path)), file=sys.stderr)


def raises(f):
    try:
        f()
    except Exception:
        return True
    return False


# ---------------------------------------------------------------- normal

def spd(rng, d, cond):
    """A random SPD matrix with condition number about cond."""
    q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    ev = np.geomspace(1.0, 1.0 / cond, d) if d > 1 else np.array([1.0])
    c = (q * ev) @ q.T
    return (c + c.T) / 2


def mp_logpdf(x, mean, cov):
    """Exact log-density at float64 arguments (cov taken as given)."""
    with mp.workdps(80):
        d = len(mean)
        C = mp.matrix([[mp.mpf(float(cov[i][j])) for j in range(d)] for i in range(d)])
        dev = mp.matrix([mp.mpf(float(x[i])) - mp.mpf(float(mean[i])) for i in range(d)])
        z = mp.lu_solve(C, dev)
        maha = sum(dev[i] * z[i] for i in range(d))
        lp = -(d * mp.log(2 * mp.pi) + mp.log(mp.det(C)) + maha) / 2
        return float(lp), float(mp.exp(lp))


def mp_entropy(cov):
    with mp.workdps(60):
        d = len(cov)
        C = mp.matrix([[mp.mpf(float(cov[i][j])) for j in range(d)] for i in range(d)])
        return float((d * (1 + mp.log(2 * mp.pi)) + mp.log(mp.det(C))) / 2)


def mp_bvn(h, k, r):
    """P(X <= h, Y <= k), standard bivariate normal, correlation r."""
    with mp.workdps(50):
        h, k, r = mp.mpf(h), mp.mpf(k), mp.mpf(r)
        s = mp.sqrt((1 - r) * (1 + r))
        if h > k:
            h, k = k, h
        f = lambda t: mp.npdf(h - t) * mp.ncdf((k - r * (h - t)) / s)
        sc = 1 / max(abs(h), 1)
        pts = [0] + [sc * 2 ** (j / 4) for j in range(-16, 40)] + [mp.inf]
        return float(mp.quad(f, pts, method="gauss-legendre"))


def gen_mvn():
    rng = np.random.default_rng(20261011)
    cases = []
    specs = []
    for d in range(1, 6):
        for cond in (1.5, 1e3, 1e8, 3e9):
            if d == 1 and cond != 1.5:
                continue
            specs.append((d, cond))
    for d, cond in specs:
        cov = spd(rng, d, cond) * rng.uniform(0.5, 3)
        mean = rng.normal(size=d) * 2
        cases.append({"mean": mean, "cov": cov, "allow_singular": False})
    # singular covariances
    for d, r in ((2, 1), (3, 2), (4, 2), (5, 3)):
        a = rng.standard_normal((d, r))
        cov = a @ a.T
        cases.append({"mean": rng.normal(size=d), "cov": (cov + cov.T) / 2, "allow_singular": True})
    out = []
    for ci, c in enumerate(cases):
        mean, cov = c["mean"], c["cov"]
        d = len(mean)
        dist = st.multivariate_normal(mean, cov, allow_singular=c["allow_singular"])
        sing = c["allow_singular"]
        pts = dist.rvs(size=12, random_state=np.random.default_rng(ci))
        pts = np.asarray(pts, dtype=float).reshape(12, d)
        far = mean + 30 * np.sqrt(np.diag(cov)) * np.sign(rng.standard_normal(d))
        extra = [mean, far, mean + 1e-3]
        if sing:
            off = mean + rng.standard_normal(d)  # almost surely off the support
            extra.append(off)
        pts = np.vstack([pts] + [np.asarray(e).reshape(1, d) for e in extra])
        row = {"mean": H(mean), "cov": H(cov.ravel()), "d": d, "allow_singular": sing,
               "points": H(pts.ravel()),
               "logpdf": H(np.atleast_1d(dist.logpdf(pts))),
               "pdf": H(np.atleast_1d(dist.pdf(pts))),
               "entropy": H(dist.entropy())}
        if not sing:
            mpv = [mp_logpdf(p, mean, cov) for p in pts]
            row["mp_logpdf"] = H([v[0] for v in mpv])
            row["mp_pdf"] = H([v[1] for v in mpv])
            row["mp_entropy"] = H(mp_entropy(cov))
        # cdf: every point, one generator for the whole call (as CDFPoints)
        npts = pts if d <= 2 else pts[:4]
        cdf = st.multivariate_normal.cdf(npts, mean, cov, allow_singular=sing,
                                         rng=np.random.default_rng(7))
        row["cdf_points"] = H(npts.ravel())
        row["cdf"] = H(np.atleast_1d(cdf))
        lower = mean - 1.0
        cdfl = st.multivariate_normal.cdf(npts, mean, cov, allow_singular=sing,
                                          lower_limit=lower, rng=np.random.default_rng(8))
        row["lower"] = H(lower)
        row["cdf_lower"] = H(np.atleast_1d(cdfl))
        rv = []
        for seed in (0, 42):
            r = st.multivariate_normal(mean, cov, allow_singular=sing).rvs(
                size=9, random_state=np.random.default_rng(seed))
            rv.append({"seed": seed, "size": 9, "hex": H(np.asarray(r).ravel())})
        row["rvs"] = rv
        out.append(row)

    # bivariate normal probabilities against mpmath
    bvn = []
    for r in (0.0, 0.3, -0.3, 0.5, -0.5, 0.8, -0.8, 0.95, -0.95, 0.999, -0.999):
        for h, k in ((0.0, 0.0), (1.0, -0.5), (-2.0, -3.0), (-5.0, -4.0), (-8.0, -8.5),
                     (-12.0, -11.0), (3.0, 2.5), (6.0, -1.0), (-1.0, 7.0), (-30.0, 2.0)):
            cov = np.array([[1.0, r], [r, 1.0]])
            sp = st.multivariate_normal.cdf([h, k], [0, 0], cov)
            bvn.append({"r": H(r), "h": H(h), "k": H(k), "scipy": H(sp), "mp": H(mp_bvn(h, k, r))})
    # scaled covariance and rectangles
    rect = []
    for cov, lo, hi in (([[2.0, 0.6], [0.6, 0.5]], [-1, -1], [1, 0.5]),
                        ([[2.0, -0.6], [-0.6, 0.5]], [3, 1], [6, 4]),
                        ([[1.0, 0.9], [0.9, 1.0]], [2, 2], [-1, -1]),
                        ([[1.0, 0.2], [0.2, 4.0]], [-np.inf, 0], [0, np.inf])):
        cov = np.array(cov)
        sp = st.multivariate_normal.cdf(hi, [0.0, 0.0], cov, lower_limit=lo)
        rect.append({"cov": H(cov.ravel()), "lower": H(lo), "upper": H(hi), "scipy": H(sp)})

    # lattice vectors
    from scipy.stats._qmvnt import _cbc_lattice
    lat = []
    for dim in (2, 3, 4, 6):
        for m in (424, 600, 849, 1200, 5000, 20011, 100003):
            q, nq = _cbc_lattice(dim, m)
            lat.append({"dim": dim, "m": m, "n": int(nq), "q": H(q)})

    # fits
    fits = []
    for d, n in ((1, 7), (2, 50), (3, 200), (5, 1000)):
        x = rng.standard_normal((n, d)) * rng.uniform(0.5, 4, d) + rng.normal(size=d)
        m, c = st.multivariate_normal.fit(x)
        fits.append({"x": H(x.ravel()), "n": n, "d": d, "mean": H(m), "cov": H(np.atleast_2d(c).ravel())})
        fm = np.arange(d, dtype=float)
        m, c = st.multivariate_normal.fit(x, fix_mean=fm)
        fits.append({"x": H(x.ravel()), "n": n, "d": d, "fix_mean": H(fm), "mean": H(m),
                     "cov": H(np.atleast_2d(c).ravel())})
        fc = np.eye(d) * 2.0
        m, c = st.multivariate_normal.fit(x, fix_cov=fc)
        fits.append({"x": H(x.ravel()), "n": n, "d": d, "fix_cov": H(fc.ravel()), "mean": H(m),
                     "cov": H(np.atleast_2d(c).ravel())})

    invalid = []
    for cov in ([[1.0, 2.0], [2.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]], [[-1.0]]):
        invalid.append({"cov": H(np.array(cov).ravel()), "d": len(cov),
                        "raises": raises(lambda: st.multivariate_normal([0.0] * len(cov), cov))})
    write("mvn", {"cases": out, "bvn": bvn, "rect": rect, "lattice": lat, "fits": fits,
                  "invalid": invalid})


# ------------------------------------------------------------- dirichlet

def mp_dir_logpdf(x, alpha):
    with mp.workdps(60):
        a = [mp.mpf(float(v)) for v in alpha]
        x = [mp.mpf(float(v)) for v in x]
        s = sum(mp.loggamma(v) for v in a) - mp.loggamma(sum(a))
        return float(-s + sum((ai - 1) * mp.log(xi) for ai, xi in zip(a, x) if ai != 1))


def mp_dir_entropy(alpha):
    with mp.workdps(60):
        a = [mp.mpf(float(v)) for v in alpha]
        a0 = sum(a)
        k = len(a)
        lnB = sum(mp.loggamma(v) for v in a) - mp.loggamma(a0)
        return float(lnB + (a0 - k) * mp.digamma(a0) - sum((v - 1) * mp.digamma(v) for v in a))


def gen_dirichlet():
    out = []
    for alpha in ([1.0, 1.0, 1.0], [0.5, 2.0, 3.5], [10.0, 20.0, 30.0, 40.0], [0.1, 0.2],
                  [1000.0, 2000.0, 5.0], [1.0, 0.7, 1.0], [2.5], [3.0, 1.0, 4.0, 1.5, 9.0]):
        alpha = np.array(alpha)
        k = len(alpha)
        x = st.dirichlet.rvs(alpha, size=8, random_state=np.random.default_rng(3))
        x = np.clip(x, 1e-300, None)
        x = x / x.sum(axis=1, keepdims=True)
        x = x[np.all(x > 0, axis=1)]
        pts = []
        for p in x:
            if raises(lambda: st.dirichlet.logpdf(p, alpha)):
                continue
            row = {"x": H(p), "logpdf": H(st.dirichlet.logpdf(p, alpha)),
                   "pdf": H(st.dirichlet.pdf(p, alpha)), "mp": H(mp_dir_logpdf(p, alpha))}
            if k > 1:
                row["logpdf_short"] = H(st.dirichlet.logpdf(p[:-1], alpha))
            pts.append(row)
        bad = []
        if k > 1:
            bad = [list(np.full(k, 1.0 / k) + 0.01), [-0.1, 1.1] + [0.0] * (k - 2)]
        rv = []
        for seed in (0, 42):
            r = st.dirichlet(alpha).rvs(size=6, random_state=np.random.default_rng(seed))
            rv.append({"seed": seed, "size": 6, "hex": H(np.asarray(r).ravel())})
        out.append({"alpha": H(alpha), "points": pts, "mean": H(st.dirichlet.mean(alpha)),
                    "var": H(np.atleast_1d(st.dirichlet.var(alpha))),
                    "cov": H(np.atleast_2d(st.dirichlet.cov(alpha)).ravel()),
                    "entropy": H(st.dirichlet.entropy(alpha)), "mp_entropy": H(mp_dir_entropy(alpha)),
                    "bad": [H(b) for b in bad],
                    "bad_raises": [raises(lambda: st.dirichlet.logpdf(np.array(b), alpha)) for b in bad],
                    "rvs": rv})
    write("dirichlet", {"cases": out})


# ----------------------------------------------------------- multinomial

def mp_multi_logpmf(x, n, p):
    with mp.workdps(60):
        s = mp.loggamma(n + 1)
        for xi, pi in zip(x, p):
            if xi == 0:
                continue
            s += xi * mp.log(mp.mpf(float(pi))) - mp.loggamma(xi + 1)
        return float(s)


def gen_multinomial():
    out = []
    for n, p in ((5, [0.2, 0.3, 0.5]), (0, [0.5, 0.5]), (20, [0.1, 0.0, 0.9]),
                 (1000, [0.25, 0.25, 0.5]), (37, [0.05, 0.15, 0.3, 0.2, 0.3]),
                 (600, [1.0]), (3, [1 / 3, 1 / 3, 1 / 3])):
        p = np.array(p)
        k = len(p)
        x = st.multinomial.rvs(n, p, size=6, random_state=np.random.default_rng(5))
        pts = [list(map(float, r)) for r in x]
        if k > 1:
            pts.append([float(n + 1)] + [0.0] * (k - 1))   # wrong total
            pts.append([-1.0, float(n + 1)] + [0.0] * (k - 2))  # negative
            pts.append([0.5, float(n) - 0.5] + [0.0] * (k - 2))  # fractional
        rows = []
        for q in pts:
            lp = st.multinomial.logpmf(q, n, p)
            row = {"x": H(q), "logpmf": H(lp), "pmf": H(st.multinomial.pmf(q, n, p))}
            if math.isfinite(lp):
                row["mp"] = H(mp_multi_logpmf([int(v) for v in q], n, p))
            rows.append(row)
        rv = []
        for seed in (0, 42):
            r = st.multinomial(n, p).rvs(size=6, random_state=np.random.default_rng(seed))
            rv.append({"seed": seed, "size": 6, "int": [int(v) for v in np.asarray(r).ravel()]})
        out.append({"n": n, "p": H(p), "points": rows, "mean": H(st.multinomial.mean(n, p)),
                    "cov": H(st.multinomial.cov(n, p).ravel()),
                    "entropy": H(st.multinomial.entropy(n, p)), "rvs": rv})
    invalid = [{"n": -1, "p": H([0.5, 0.5])}, {"n": 3, "p": H([0.5, 0.6])},
               {"n": 3, "p": H([1.5, -0.5])}]
    for c in invalid:
        c["nan"] = bool(np.isnan(st.multinomial.logpmf([1, 2], c["n"], [float.fromhex(v) for v in c["p"]])))
    write("multinomial", {"cases": out, "invalid": invalid})


# ------------------------------------------------------------------- kde

def gen_kde():
    rng = np.random.default_rng(99)
    cases = []
    data = {
        "n50": rng.normal(size=50),
        "n1000": rng.normal(size=1000),
        "bimodal": np.r_[rng.normal(-3, 0.5, 300), rng.normal(2, 1.5, 200)],
        "d2": rng.multivariate_normal([0, 1], [[1, 0.6], [0.6, 2]], size=200).T,
        "d3": rng.normal(size=(3, 150)) * np.array([[1], [3], [0.2]]),
    }
    wts = {"n50": rng.uniform(0.1, 2, 50), "d2": rng.uniform(0, 1, 200)}
    bws = [("scott", None), ("silverman", None), ("scalar", 0.3), ("half_scott", None)]
    for name, ds in data.items():
        for wname in (None, name if name in wts else None):
            for bw, val in bws:
                if wname is None and bw == "half_scott" and name != "n50":
                    continue
                w = wts[wname] if wname else None
                if bw == "scalar":
                    m = val
                elif bw == "half_scott":
                    m = lambda k: k.scotts_factor() / 2
                else:
                    m = bw
                kde = st.gaussian_kde(ds, bw_method=m, weights=w)
                d = kde.d
                pts = np.atleast_2d(ds)[:, :7] + 0.1
                lo = np.atleast_2d(ds).min(axis=1, keepdims=True)
                hi = np.atleast_2d(ds).max(axis=1, keepdims=True)
                grid = lo + (hi - lo) * np.linspace(-0.5, 1.5, 9)
                far = np.atleast_2d(ds).mean(axis=1, keepdims=True) + 60 * np.atleast_2d(ds).std(axis=1, keepdims=True)
                pts = np.hstack([pts, grid, far])
                row = {"name": name, "weights": H(w) if w is not None else None, "bw": bw,
                       "bw_value": val, "dataset": H(np.atleast_2d(ds).ravel()),
                       "d": d, "n": kde.n, "factor": H(kde.factor), "neff": H(kde.neff),
                       "covariance": H(kde.covariance.ravel()), "cho_cov": H(kde.cho_cov.ravel()),
                       "inv_cov": H(kde.inv_cov.ravel()), "log_det": H(kde.log_det),
                       "scotts": H(kde.scotts_factor()), "silverman": H(kde.silverman_factor()),
                       "points": H(pts.ravel()), "m": pts.shape[1],
                       "pdf": H(kde.evaluate(pts)), "logpdf": H(kde.logpdf(pts))}
                mean = np.atleast_2d(ds).mean(axis=1) + 0.3
                cov = np.eye(d) * 0.7
                row["ig_mean"] = H(mean)
                row["ig_cov"] = H(cov.ravel())
                row["integrate_gaussian"] = H(kde.integrate_gaussian(mean, cov))
                if d == 1:
                    row["box1d"] = [[H(a), H(b), H(kde.integrate_box_1d(a, b))]
                                    for a, b in ((-1.0, 0.5), (0.0, np.inf), (3.0, 5.0), (-np.inf, -2.5), (2.0, 1.0))]
                if d <= 2:
                    blo, bhi = mean - 1.0, mean + 0.5
                    row["box"] = [H(blo), H(bhi), H(kde.integrate_box(blo, bhi))]
                rs = []
                for seed in (0, 42):
                    r = kde.resample(size=6, seed=np.random.default_rng(seed))
                    rs.append({"seed": seed, "size": 6, "hex": H(r.ravel())})
                r = kde.resample(seed=np.random.default_rng(1))
                rs.append({"seed": 1, "size": 0, "shape": list(r.shape), "hex": H(r.ravel()[:20])})
                row["resample"] = rs
                cases.append(row)
    # integrate_kde between two estimates
    pairs = []
    a = st.gaussian_kde(data["n50"])
    b = st.gaussian_kde(data["bimodal"], weights=None)
    pairs.append({"a": "n50", "b": "bimodal", "value": H(a.integrate_kde(b))})
    a = st.gaussian_kde(data["d2"], weights=wts["d2"])
    b = st.gaussian_kde(data["d2"][:, :60] + 0.5)
    pairs.append({"a": "d2w", "b": "d2shift", "value": H(a.integrate_kde(b)), "b_data": H((data["d2"][:, :60] + 0.5).ravel())})
    # integrate_box in three dimensions (randomized)
    k3 = st.gaussian_kde(data["d3"])
    box3 = {"lo": H([-1.0, -2.0, -0.1]), "hi": H([1.0, 2.0, 0.2]),
            "value": H(k3.integrate_box(np.array([-1.0, -2.0, -0.1]), np.array([1.0, 2.0, 0.2]),
                                        rng=np.random.default_rng(4)))}
    # marginal
    marg = st.gaussian_kde(data["d3"]).marginal([0, 2])
    pts = np.array([[0.0, 0.5, -1.0], [0.0, 0.1, 0.3]])
    margo = {"dims": [0, 2], "pdf": H(marg.evaluate(pts)), "points": H(pts.ravel()), "factor": H(marg.factor)}
    write("kde", {"cases": cases, "pairs": pairs, "box3": box3, "marginal": margo})


# ------------------------------------------------------------- bootstrap

def gen_bootstrap():
    rng = np.random.default_rng(11)
    samples = {
        "n10": rng.normal(size=10),
        "n30": rng.exponential(size=30),
        "n200": rng.normal(1, 2, size=200),
        "n9b": rng.normal(0.5, 1, size=9),
        "n30b": rng.normal(0.2, 1, size=30),
        "const": np.ones(8),
    }
    mean = lambda x, axis=-1: np.mean(x, axis=axis)
    diff = lambda x, y, axis=-1: np.mean(x, axis=axis) - np.mean(y, axis=axis)
    median = lambda x: np.median(x)
    out = []
    specs = []
    for name in ("n10", "n30", "n200"):
        for method in ("BCa", "percentile", "basic"):
            for alt in ("two-sided", "less", "greater"):
                specs.append(dict(data=[name], stat="mean", method=method, alternative=alt))
    specs += [
        dict(data=["n10"], stat="mean", method="BCa", n_resamples=999, full=True),
        dict(data=["n30"], stat="mean", method="percentile", n_resamples=999, batch=100, full=True),
        dict(data=["n30"], stat="mean", method="BCa", confidence_level=0.9, batch=1000),
        dict(data=["n10", "n30b"], stat="diff", method="BCa", n_resamples=999, full=True),
        dict(data=["n10", "n30b"], stat="diff", method="basic"),
        dict(data=["n10", "n9b"], stat="diff", method="BCa", paired=False, n_resamples=999),
        dict(data=["n30", "n30b"], stat="diff", method="BCa", paired=True, n_resamples=999, full=True),
        dict(data=["n30", "n30b"], stat="diff", method="percentile", paired=True, alternative="less"),
        dict(data=["n30"], stat="median", method="BCa", n_resamples=999, full=True),
        dict(data=["n10"], stat="median", method="basic", confidence_level=0.8, n_resamples=500),
        dict(data=["const"], stat="mean", method="BCa", n_resamples=99),
        dict(data=["n10"], stat="mean", method="percentile", n_resamples=1, full=True),
    ]
    stats_ = {"mean": mean, "diff": diff, "median": median}
    for i, s in enumerate(specs):
        data = tuple(samples[n] for n in s["data"])
        kw = dict(n_resamples=s.get("n_resamples", 9999), method=s.get("method", "BCa"),
                  alternative=s.get("alternative", "two-sided"),
                  confidence_level=s.get("confidence_level", 0.95),
                  paired=s.get("paired", False), batch=s.get("batch"))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with np.errstate(all="ignore"):
                r = st.bootstrap(data, stats_[s["stat"]], rng=np.random.default_rng(1000 + i), **kw)
        bd = np.asarray(r.bootstrap_distribution)
        row = dict(s, seed=1000 + i, low=H(r.confidence_interval.low), high=H(r.confidence_interval.high),
                   se=H(r.standard_error), head=H(bd[:5]), sum=H(math.fsum(bd)), len=len(bd))
        row.update(n_resamples=kw["n_resamples"], batch=kw["batch"] or 0,
                   confidence_level=kw["confidence_level"], alternative=kw["alternative"])
        if s.get("full"):
            row["dist"] = H(bd)
        out.append(row)
    write("bootstrap", {"samples": {k: H(v) for k, v in samples.items()}, "cases": out})


GENS = {"mvn": gen_mvn, "dirichlet": gen_dirichlet, "multinomial": gen_multinomial,
        "kde": gen_kde, "bootstrap": gen_bootstrap}


def main():
    want = sys.argv[1:] or list(GENS)
    for w in want:
        with np.errstate(all="ignore"):
            GENS[w]()


if __name__ == "__main__":
    main()
