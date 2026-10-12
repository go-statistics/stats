"""Regenerate the accuracy fixtures in testdata/fixtures.

    uv run --python cpython-3.12-macos-x86_64-none \
        --with mpmath==1.4.1 --with scipy==1.18.1 --with numpy==2.5.4 \
        python testdata/gen/gen.py [fixture-name ...]

Each fam_*.py module exposes FIXTURES = {name: builder}; a builder returns
(nargs, cases, mpfn, spfn, doc) for common.write. With no arguments every
fixture is regenerated; fixtures run in parallel worker processes. Sampling is
seeded by the fixture name, so a rerun with the same library versions
reproduces the files byte for byte.

The mpmath evaluations are slow under emulation, so the fixtures are made
in two passes: first natively (mpmath, and a provisional SciPy column),
then the SciPy column is recomputed with the x86_64 interpreter, the
reference build (NumPy and SciPy's macOS arm64 wheels round some libm calls
differently):

    uv run --with ... python testdata/gen/gen.py NAMES...
    uv run --python cpython-3.12-macos-x86_64-none --with ... \
        python testdata/gen/gen.py --scipy-column NAMES...
"""

import glob
import importlib
import multiprocessing
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def modules():
    for p in sorted(glob.glob(os.path.join(HERE, "fam_*.py"))):
        yield os.path.splitext(os.path.basename(p))[0]


def registry():
    reg = {}
    for m in modules():
        mod = importlib.import_module(m)
        for name in mod.FIXTURES:
            reg[name] = m
    return reg


def run(job):
    modname, name, scipy_only = job
    sys.path.insert(0, HERE)
    import common
    mod = importlib.import_module(modname)
    nargs, cases, mpfn, spfn, doc = mod.FIXTURES[name]()
    if scipy_only:
        common.rewrite_scipy(name, spfn)
    else:
        common.write(name, nargs, cases, mpfn, spfn, doc)
    return name


def main():
    reg = registry()
    args = sys.argv[1:]
    scipy_only = "--scipy-column" in args
    args = [a for a in args if a != "--scipy-column"]
    want = args or sorted(reg)
    for w in want:
        if w not in reg:
            sys.exit("unknown fixture %r; known: %s" % (w, ", ".join(sorted(reg))))
    jobs = [(reg[w], w, scipy_only) for w in want]
    workers = int(os.environ.get("GEN_WORKERS", "8"))
    with multiprocessing.Pool(workers) as pool:
        for _ in pool.imap_unordered(run, jobs):
            pass


if __name__ == "__main__":
    main()
