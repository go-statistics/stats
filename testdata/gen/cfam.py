"""Shared builders for the closed-form distribution fixtures (fam_*.py of
the closed-form families): every function there has an exact mpmath
expression, the quantiles included, so no fixture needs an iterative
inversion.

A family module declares, per function, its regions as (name, cases) with
cases a list of argument tuples (x or p first, then the shape parameters
in SciPy's order), and calls fixtures() to obtain FIXTURES.
"""

import mpmath
from mpmath import mpf

import common
from dists import logu, uni

N = 250


def R(name):
    return common.rng(name)


def xs(r, lo, hi, n=N, log=False, sign=1):
    if log:
        return logu(r, lo, hi, n, sign)
    return uni(r, lo, hi, n)


def probs(r, n=N):
    """The three probability regions every quantile fixture has."""
    lower = logu(r, 1e-300, 1e-3, n)
    core = uni(r, 1e-3, 1 - 1e-3, n)
    upper = [1.0 - q for q in logu(r, 1.2e-16, 1e-3, n)]
    return [("lower", lower), ("core", core), ("upper", upper)]


def with_shapes(points, shapes):
    """Pair each point with a shape tuple, cycling through shapes."""
    return [(p,) + tuple(shapes[i % len(shapes)]) for i, p in enumerate(points)]


def make(nargs, regions, mpfn, spfn, doc):
    """A fixture builder for gen.py from a list of (region, [args, ...])."""
    def build():
        cases = []
        for reg, args in regions():
            cases += [(reg, a if isinstance(a, tuple) else (a,)) for a in args]
        return nargs, cases, mpfn, spfn, doc
    return build


def M(x):
    return mpf(x)


def log1mexp(t):
    """log(1 - exp(-t)) in mpmath."""
    if t > 1:
        return mpmath.log1p(-mpmath.exp(-t))
    return mpmath.log(-mpmath.expm1(-t))
