"""SciPy parity fixtures for the distributions: testdata/scipy/dist_<name>.json.gz.

    uv run --with mpmath==1.4.1 --with scipy==1.18.1 --with numpy==2.5.4 \
        python testdata/gen/parity.py [scipy-name ...]

The ULP fixtures (gen.py) measure accuracy against mpmath in standard form.
These fixtures check everything else against SciPy for parameterised
instances (shapes, loc, scale): the values in the bulk of the distribution,
the moments, entropy, median, interval and support, fit results, and the
random draws for fixed seeds (as exact float64 bit patterns).

Each parity_*.py module exposes SPECS, a list of dicts:

    scipy      name in scipy.stats
    go         Go type name
    kind       "continuous" | "discrete"
    shapes     Go field names of the shape parameters, in SciPy's order
    instances  list of {field: value} (Loc, Scale default 0, 1)
    invalid    list of {field: value} that SciPy rejects
    fit        list of {"data": [...], "fixed": {field: value}}  (optional)
    rvs_sizes  list of sizes to draw (default [None, 1000, [3, 4]])
"""

import glob
import gzip
import importlib
import json
import math
import os
import sys
import warnings

import numpy as np
import scipy
import scipy.stats as st

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.normpath(os.path.join(HERE, "..", "scipy"))
sys.path.insert(0, HERE)

PROBS = [1e-6, 0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 0.999, 1 - 1e-6]
SEEDS = [0, 42, 20261011]


def f(x):
    """A float64 as JSON: finite numbers as repr, specials as strings."""
    x = float(x)
    if math.isnan(x):
        return "nan"
    if math.isinf(x):
        return "inf" if x > 0 else "-inf"
    return x


def args_of(spec, inst):
    a = [inst[s] for s in spec["shapes"]]  # raw values (not f())
    kw = {"loc": inst.get("Loc", 0.0)}
    if spec["kind"] == "continuous":
        kw["scale"] = inst.get("Scale", 1.0)
    return a, kw


def full(spec, inst):
    d = dict(inst)
    d.setdefault("Loc", 0.0)
    if spec["kind"] == "continuous":
        d.setdefault("Scale", 1.0)
    return {k: f(v) for k, v in d.items()}


def draws(arr):
    arr = np.asarray(arr)
    if arr.dtype.kind in "iu":
        return {"shape": list(arr.shape), "int": [int(v) for v in arr.ravel()]}
    return {"shape": list(arr.shape), "hex": [float(v).hex() for v in arr.ravel()]}


def instance(spec, inst):
    dist = getattr(st, spec["scipy"])
    a, kw = args_of(spec, inst)
    fr = dist(*a, **kw)
    out = {"params": full(spec, inst)}
    m, v, s, k = fr.stats(moments="mvsk")
    out.update(mean=f(m), var=f(v), std=f(fr.std()), skew=f(s), kurtosis=f(k),
               median=f(fr.median()), entropy=f(fr.entropy()),
               moments=[f(fr.moment(n)) for n in range(0, 5)],
               interval=[f(x) for x in fr.interval(0.95)],
               support=[f(x) for x in fr.support()])
    pts = []
    for p in PROBS:
        x = float(fr.ppf(p))
        if not math.isfinite(x):
            continue
        row = {"x": f(x)}
        if spec["kind"] == "continuous":
            row.update(pdf=f(fr.pdf(x)), logpdf=f(fr.logpdf(x)))
        else:
            row.update(pmf=f(fr.pmf(x)), logpmf=f(fr.logpmf(x)))
        row.update(cdf=f(fr.cdf(x)), sf=f(fr.sf(x)), logcdf=f(fr.logcdf(x)),
                   logsf=f(fr.logsf(x)))
        pts.append(row)
    out["points"] = pts
    out["quantiles"] = [{"p": f(p), "ppf": f(fr.ppf(p)), "isf": f(fr.isf(p))} for p in PROBS]
    rv = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        for size in spec.get("rvs_sizes", [None, 1000, [3, 4]]):
            sz = tuple(size) if isinstance(size, list) else size
            r = fr.rvs(size=sz, random_state=rng)
            d = draws(r)
            d.update(seed=seed, size=size)
            rv.append(d)
    out["rvs"] = rv
    return out


def fit_case(spec, case):
    dist = getattr(st, spec["scipy"])
    data = np.asarray(case["data"], dtype=float)
    kw = {}
    names = spec["shapes"] + ["Loc"] + (["Scale"] if spec["kind"] == "continuous" else [])
    for k, v in case.get("fixed", {}).items():
        if k == "Loc":
            kw["floc"] = v
        elif k == "Scale":
            kw["fscale"] = v
        else:
            kw["f%d" % spec["shapes"].index(k)] = v
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        params = dist.fit(data, **kw)
        nll = dist.nnlf(params, data)
    return {"data": [f(x) for x in data], "fixed": case.get("fixed", {}),
            "params": dict(zip(names, [f(p) for p in params])), "nnlf": f(nll)}


def run(spec):
    out = {"scipy": spec["scipy"], "go": spec["go"], "kind": spec["kind"],
           "scipy_version": scipy.__version__, "numpy_version": np.__version__,
           "instances": [], "invalid": [full(spec, i) for i in spec.get("invalid", [])],
           "fit": [], "fit_kind": spec.get("fit_kind", "")}
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for inst in spec["instances"]:
            out["instances"].append(instance(spec, inst))
    for case in spec.get("fit", []):
        out["fit"].append(fit_case(spec, case))
    path = os.path.join(OUT, "dist_%s.json.gz" % spec["scipy"])
    raw = json.dumps(out, separators=(",", ":"), sort_keys=True).encode()
    with open(path, "wb") as fh:
        with gzip.GzipFile(filename="", mode="wb", fileobj=fh, mtime=0, compresslevel=9) as gz:
            gz.write(raw)
    print("%-12s %d instances, %d fits" % (spec["scipy"], len(out["instances"]), len(out["fit"])), file=sys.stderr)


def main():
    specs = []
    for p in sorted(glob.glob(os.path.join(HERE, "parity_*.py"))):
        mod = importlib.import_module(os.path.splitext(os.path.basename(p))[0])
        specs += mod.SPECS
    want = sys.argv[1:]
    for s in specs:
        if not want or s["scipy"] in want:
            run(s)


if __name__ == "__main__":
    main()
