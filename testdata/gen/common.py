"""Shared machinery for the accuracy fixtures.

Every fixture row holds the arguments, the exact value computed by mpmath at
high precision split into a double-double (hi + lo, hi = the correctly
rounded float64), and the value SciPy returns for the same float64 arguments.
The Go tests measure the error of this package in ULPs against hi + lo; the
same measurement applied to the SciPy column gives the comparison table in
docs/accuracy.md. SciPy is used only as a point of comparison, never as the
reference.

Oracle self-check: every value is computed twice, at PREC and at PREC2 bits.
A row whose two evaluations disagree beyond 2^-80 relative is dropped and
counted, so an mpmath evaluation that silently lost precision cannot become
the truth the tests are held to.
"""

import gzip
import io
import math
import os
import random
import signal
import sys

import mpmath
from mpmath import mp, mpf

PREC = 192
PREC2 = 320
TIMEOUT_S = 20

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.normpath(os.path.join(HERE, "..", "fixtures"))


class Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise Timeout()


def fmt(x):
    """Shortest round-trip text of a float64 (Go's ParseFloat reads it back
    bit-exactly, including nan/inf/-0.0)."""
    if math.isnan(x):
        return "nan"
    if math.isinf(x):
        return "inf" if x > 0 else "-inf"
    return repr(float(x))


def split(v):
    """mpf -> (hi, lo): hi is v rounded to nearest float64, lo = v - hi."""
    if isinstance(v, float):
        return v, 0.0
    if v is None:
        return math.nan, 0.0
    if mpmath.isnan(v):
        return math.nan, 0.0
    if mpmath.isinf(v):
        return (math.inf if v > 0 else -math.inf), 0.0
    if v == 0:
        return 0.0, 0.0
    try:
        hi = float(v)  # mpmath rounds to nearest, handles subnormals
    except OverflowError:
        hi = math.inf if v > 0 else -math.inf
    if math.isinf(hi):
        return hi, 0.0
    lo = float(v - mpf(hi))
    return hi, lo


def evaluate(fn, args):
    """Evaluate fn at PREC and PREC2; return the PREC2 value or None if the
    two disagree (oracle self-check failure) or evaluation timed out."""
    signal.signal(signal.SIGALRM, _alarm)
    vals = []
    for prec in (PREC, PREC2):
        signal.alarm(TIMEOUT_S)
        try:
            with mp.workprec(prec):
                v = fn(*args)
                if v is None:
                    return None  # the oracle found no root/bracket: dropped
                if isinstance(v, float):
                    vals.append(v)
                else:
                    vals.append(mpf(v) if not isinstance(v, mpf) else +v)
        except Timeout:
            return "timeout"
        except (mpmath.libmp.libhyper.NoConvergence, ZeroDivisionError, ValueError, TypeError, OverflowError):
            return None  # counted as dropped: the oracle could not decide
        finally:
            signal.alarm(0)
    a, b = vals
    if isinstance(a, float) or isinstance(b, float):
        if (math.isnan(a) and math.isnan(b)) or a == b:
            return b
        return None
    if mpmath.isnan(a) or mpmath.isnan(b) or mpmath.isinf(a) or mpmath.isinf(b):
        return b if (a == b or (mpmath.isnan(a) and mpmath.isnan(b))) else None
    with mp.workprec(PREC2):
        if b == 0:
            return b if a == 0 else None
        if abs((a - b) / b) > mpf(2) ** -80:
            return None
    return b


def write(name, nargs, cases, mpfn, spfn, doc=""):
    """cases: iterable of (region, args tuple of float64).
    mpfn: function of float64 args evaluated under mp precision; it receives
    Python floats and must return an mpf (or a float for exact specials).
    spfn: SciPy's function of the float64 args, or None if SciPy has none."""
    path = os.path.join(OUT, name + ".tsv.gz")
    dropped = 0
    timeouts = 0
    rows = []
    seen = set()
    for region, args in cases:
        args = tuple(float(a) for a in args)
        key = (region, tuple(fmt(a) for a in args))
        if key in seen:
            continue
        seen.add(key)
        v = evaluate(mpfn, args)
        if v == "timeout":
            timeouts += 1
            continue
        if v is None:
            dropped += 1
            continue
        hi, lo = split(v)
        if spfn is None:
            sp = "-"
        else:
            import numpy as np

            with np.errstate(all="ignore"):
                try:
                    sp = fmt(float(spfn(*args)))
                except Exception:
                    sp = "-"
        rows.append("\t".join([region] + [fmt(a) for a in args] + [fmt(hi), fmt(lo), sp]))
    f = io.StringIO()
    if True:
        f.write("# %s: %d argument(s); columns: region, args, mpmath hi, mpmath lo, scipy\n" % (name, nargs))
        if doc:
            f.write("# %s\n" % doc)
        f.write("# oracle: mpmath %s at %d and %d bits (rows that disagree dropped: %d, timeouts: %d)\n"
                % (mpmath.__version__, PREC, PREC2, dropped, timeouts))
        for r in rows:
            f.write(r + "\n")
    # mtime=0 and no file name: the output is reproducible byte for byte.
    with open(path, "wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=9) as gz:
            gz.write(f.getvalue().encode())
    print("%-14s %6d rows  dropped %d  timeouts %d" % (name, len(rows), dropped, timeouts), file=sys.stderr)


# ---------------------------------------------------------------- sampling

def rng(name):
    return random.Random("go-statistics/stats:" + name)


def loguniform(r, lo, hi, n, sign=1):
    a, b = math.log(lo), math.log(hi)
    return [sign * math.exp(r.uniform(a, b)) for _ in range(n)]


def uniform(r, lo, hi, n):
    return [r.uniform(lo, hi) for _ in range(n)]


SPECIALS = [math.nan, math.inf, -math.inf, 0.0, -0.0, 5e-324, -5e-324,
            2.2250738585072014e-308, 1.7976931348623157e308, -1.7976931348623157e308]


def ival(x):
    """Return x as mpf when finite; exact specials pass through."""
    return mpf(x)


def rewrite_scipy(name, spfn):
    """Recompute the SciPy column of an existing fixture (same arguments)
    with this interpreter's SciPy, and record which build it was."""
    import platform
    import numpy as np
    import scipy
    path = os.path.join(OUT, name + ".tsv.gz")
    with gzip.open(path, "rt") as fh:
        lines = fh.read().splitlines()
    out = []
    for line in lines:
        if line.startswith("# scipy:"):
            continue
        if line.startswith("#"):
            out.append(line)
            continue
        cols = line.split("\t")
        args = [float(c) for c in cols[1:-3]]
        with np.errstate(all="ignore"):
            try:
                sp = fmt(float(spfn(*args)))
            except Exception:
                sp = "-"
        cols[-1] = sp
        out.append("\t".join(cols))
    hdr = [l for l in out if l.startswith("#")]
    body = [l for l in out if not l.startswith("#")]
    hdr.append("# scipy: %s numpy %s on %s" % (scipy.__version__, np.__version__, platform.machine()))
    with open(path, "wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=9) as gz:
            gz.write(("\n".join(hdr + body) + "\n").encode())
    print("%-14s scipy column rewritten (%s)" % (name, platform.machine()), file=sys.stderr)
