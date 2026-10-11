"""Exact references for the SciPy test fixtures: testdata/oracle/tests_<group>.json.gz.

    uv run --with mpmath==1.4.1 python testdata/gen/oracle_tests.py [group ...]

(native Python is fine: no SciPy or NumPy is involved.)

For each case of testdata/scipy/tests_<group>.json.gz this computes the
exact value of the outputs (for the float64 inputs as given) with mpmath
at 50 digits, or with exact integer and rational arithmetic where the null
distribution is combinatorial, and rounds them to float64. The Go harness
(tests_scipy_test.go) then measures both the Go function and SciPy against
these values: where SciPy's value is itself inaccurate (cancellation, a
noisy root-finding equation, the float64 recursions of the Kolmogorov
distribution), the Go function is held to the exact value instead of to
SciPy's.

The definitions are SciPy's (the same formulas, approximations included:
Shapiro-Wilk's coefficients, the asymptotic p-values); only the arithmetic
is exact. Fields left out are those whose exact value is not defined
independently of an algorithm (the logistic fit of anderson stops fsolve at
xtol=1e-5) or would be too slow to compute (the Durbin matrix for large
n·d).

The output records the SHA-256 of the fixture it was computed from, so
that a regenerated fixture without regenerated references is detected.
"""

import gzip
import hashlib
import json
import math
import os
import sys
from fractions import Fraction
from math import comb

import mpmath as mp

mp.mp.dps = 50
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.normpath(os.path.join(HERE, "..", "scipy"))
OUT = os.path.normpath(os.path.join(HERE, "..", "oracle"))


def num(v):
    if isinstance(v, str):
        return float(v)
    return float(v)


def f(x):
    x = float(x)
    if math.isnan(x):
        return "nan"
    if math.isinf(x):
        return "inf" if x > 0 else "-inf"
    return x


def M(v):
    return mp.mpf(num(v))


# ---------------------------------------------------------------- helpers

def mean(a):
    return mp.fsum(a) / len(a)


def var(a, ddof):
    if len(a) - ddof <= 0:
        return mp.nan
    m = mean(a)
    return mp.fsum((v - m) ** 2 for v in a) / (len(a) - ddof)


def norm_sf(z):
    return mp.erfc(z / mp.sqrt(2)) / 2


def norm_cdf(z):
    return mp.erfc(-z / mp.sqrt(2)) / 2


def t_sf(t, df):
    if t >= 0:
        return mp.betainc(df / 2, mp.mpf(1) / 2, 0, df / (df + t * t), regularized=True) / 2
    return 1 - t_sf(-t, df)


def chi2_sf(x, df):
    if x <= 0:
        return mp.mpf(1)
    return mp.gammainc(mp.mpf(df) / 2, x / 2, mp.inf, regularized=True)


def f_sf(x, dfn, dfd):
    if x <= 0:
        return mp.mpf(1)
    return mp.betainc(mp.mpf(dfd) / 2, mp.mpf(dfn) / 2, 0, dfd / (dfd + dfn * x), regularized=True)


def pvalue_sym(t, sf, alt):
    if alt == "less":
        return sf(-t)
    if alt == "greater":
        return sf(t)
    return 2 * sf(abs(t))


def ranks(a):
    """Average ranks (exact half-integers as Fractions) and tie counts."""
    order = sorted(range(len(a)), key=lambda i: a[i])
    r = [None] * len(a)
    ties = []
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and a[order[j + 1]] == a[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = Fraction(i + j + 2, 2)
        ties.append(j - i + 1)
        i = j + 1
    return r, ties


def finite(*arrs):
    return all(not math.isnan(num(v)) for a in arrs for v in a)


# ------------------------------------------------------------- t-tests

def o_ttest_1samp(c, data):
    x = [M(v) for v in data[0]]
    if len(x) < 2 or not finite(data[0]):
        return None
    n = len(x)
    s = mp.sqrt(var(x, 1) / n)
    if s == 0:
        return None
    t = (mean(x) - M(c["opts"]["popmean"])) / s
    return {"statistic": t, "pvalue": pvalue_sym(t, lambda u: t_sf(u, n - 1), c["opts"]["alternative"])}


def o_ttest_rel(c, data):
    if len(data[0]) != len(data[1]):
        return None
    d = [num(a) - num(b) for a, b in zip(data[0], data[1])]  # SciPy forms a - b in float64
    return o_ttest_1samp({"opts": {"popmean": 0.0, "alternative": c["opts"]["alternative"]}}, [d])


def trim_stats(a, trim):
    s = sorted(a)
    n = len(s)
    g = int(n * trim)
    if g == 0:
        v = var(s, 1)
    else:
        w = list(s)
        for i in range(g):
            w[i] = s[g]
            w[n - 1 - i] = s[n - 1 - g]
        v = var(w, 2 * g + 1)
    m = mean(s[g:n - g])
    return v, m, n - 2 * g


def o_ttest_ind(c, data):
    a, b = [M(v) for v in data[0]], [M(v) for v in data[1]]
    if len(a) == 0 or len(b) == 0 or not finite(data[0], data[1]):
        return None
    o = c["opts"]
    if o["trim"] == 0:
        n1, n2 = len(a), len(b)
        v1 = var(a, 1) if n1 > 1 else mp.nan
        v2 = var(b, 1) if n2 > 1 else mp.nan
        m1, m2 = mean(a), mean(b)
    else:
        v1, m1, n1 = trim_stats(a, o["trim"])
        v2, m2, n2 = trim_stats(b, o["trim"])
    if o["equal_var"]:
        if n1 == 1:
            v1 = 0
        if n2 == 1:
            v2 = 0
        if mp.isnan(v1) or mp.isnan(v2):
            return None
        df = n1 + n2 - 2
        den = mp.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / df * (mp.mpf(1) / n1 + mp.mpf(1) / n2))
    else:
        if n1 < 2 or n2 < 2 or mp.isnan(v1) or mp.isnan(v2):
            return None
        vn1, vn2 = v1 / n1, v2 / n2
        if vn1 + vn2 == 0:
            return None
        df = (vn1 + vn2) ** 2 / (vn1 ** 2 / (n1 - 1) + vn2 ** 2 / (n2 - 1))
        den = mp.sqrt(vn1 + vn2)
    if den == 0:
        return None
    t = (m1 - m2) / den
    return {"statistic": t, "df": df, "pvalue": pvalue_sym(t, lambda u: t_sf(u, df), o["alternative"])}


# ------------------------------------------------------- chi-squared family

def pd_terms(obs, exp, lam):
    out = []
    for fo, e in zip(obs, exp):
        if lam == 1:
            out.append((fo - e) ** 2 / e)
        elif lam == 0:
            out.append(0 if fo == 0 else 2 * fo * mp.log(fo / e))
        elif lam == -1:
            out.append(mp.inf if fo == 0 else 2 * e * mp.log(e / fo))
        else:
            if fo == 0 and lam < 0:
                return None
            out.append(fo * ((fo / e) ** lam - 1) / (mp.mpf(lam) * (lam + 1) / 2))
    return out


def o_power(c, data, lam=None):
    obs = [M(v) for v in data[0]]
    if not finite(data[0]) or len(obs) == 0:
        return None
    if len(data) > 1:
        if not finite(data[1]):
            return None
        exp = [M(v) for v in data[1]]
    else:
        e = mean(obs)
        exp = [e] * len(obs)
    lam = c["opts"].get("lambda_", 1) if lam is None else lam
    terms = pd_terms(obs, exp, mp.mpf(num(lam)) if lam not in (0, 1, -1) else lam)
    if terms is None:
        return None
    stat = mp.fsum(terms)
    df = len(obs) - 1 - c["opts"]["ddof"]
    if df <= 0 or mp.isinf(stat):
        return {"statistic": stat}
    return {"statistic": stat, "pvalue": chi2_sf(stat, df)}


def o_chisquare(c, data):
    return o_power(c, data, 1)


def o_chi2_contingency(c, data):
    shape = [int(num(s)) for s in data[1]]
    obs = [Fraction(num(v)) for v in data[0]]
    if any(v < 0 for v in obs):
        return None
    d = len(shape)
    strides = []
    st = 1
    for k in reversed(shape):
        strides.insert(0, st)
        st *= k
    marg = [[Fraction(0)] * s for s in shape]
    for i, v in enumerate(obs):
        for k in range(d):
            marg[k][(i // strides[k]) % shape[k]] += v
    total = sum(obs)
    exp = []
    for i in range(len(obs)):
        p = Fraction(1)
        for k in range(d):
            p *= marg[k][(i // strides[k]) % shape[k]]
        exp.append(p / total ** (d - 1))
    if any(e == 0 for e in exp):
        return None
    dof = len(exp) - sum(shape) + d - 1
    if dof == 0:
        return None
    if dof == 1 and c["opts"]["correction"]:
        adj = []
        for o_, e in zip(obs, exp):
            diff = e - o_
            mag = min(Fraction(1, 2), abs(diff))
            adj.append(o_ + (mag if diff > 0 else -mag if diff < 0 else 0))
        obs = adj
    lam = c["opts"]["lambda_"]
    lam = 1 if lam == "none" else lam
    lam = lam if lam in (0, 1, -1) else mp.mpf(num(lam))
    terms = pd_terms([mp.mpf(v.numerator) / v.denominator for v in obs],
                     [mp.mpf(v.numerator) / v.denominator for v in exp], lam)
    if terms is None:
        return None
    stat = mp.fsum(terms)
    out = {"statistic": stat}
    if not mp.isinf(stat):
        out["pvalue"] = chi2_sf(stat, dof)
    return out


def hyp_pmf(k, M_, n, N):
    if k < max(0, N - (M_ - n)) or k > min(n, N):
        return Fraction(0)
    return Fraction(comb(n, k) * comb(M_ - n, N - k), comb(M_, N))


def hyp_cdf(k, M_, n, N):
    return sum((hyp_pmf(j, M_, n, N) for j in range(0, k + 1)), Fraction(0))


def o_fisher_exact(c, data):
    t = [int(num(v)) for v in data[0]]
    c00, c01, c10, c11 = t
    if 0 in (c00 + c10, c01 + c11, c00 + c01, c10 + c11):
        return None
    n1, n2, n = c00 + c01, c10 + c11, c00 + c10
    Mt = n1 + n2
    alt = c["opts"]["alternative"]
    if alt == "less":
        p = hyp_cdf(c00, Mt, n1, n)
    elif alt == "greater":
        p = hyp_cdf(c01, Mt, n1, c01 + c11)
    else:
        mode = int((n + 1) * (n1 + 1) / (n1 + n2 + 2))
        pexact = hyp_pmf(c00, Mt, n1, n)
        pmode = hyp_pmf(mode, Mt, n1, n)
        if abs(pexact - pmode) / max(pexact, pmode) <= Fraction(1, 10 ** 14):
            return {"pvalue": 1.0}
        gamma = 1 + Fraction(1, 10 ** 14)
        lo, hi = max(0, n - n2), min(n1, n)
        p = sum((q for q in (hyp_pmf(k, Mt, n1, n) for k in range(lo, hi + 1)) if q <= pexact * gamma),
                Fraction(0))
    return {"pvalue": min(p, 1)}


# -------------------------------------------------------- correlations

def o_pearsonr(c, data):
    if len(data[0]) != len(data[1]) or len(data[0]) < 3 or not finite(data[0], data[1]):
        return None
    x, y = [M(v) for v in data[0]], [M(v) for v in data[1]]
    mx, my = mean(x), mean(y)
    sxy = mp.fsum((a - mx) * (b - my) for a, b in zip(x, y))
    sxx = mp.fsum((a - mx) ** 2 for a in x)
    syy = mp.fsum((b - my) ** 2 for b in y)
    if sxx == 0 or syy == 0:
        return None
    r = sxy / mp.sqrt(sxx * syy)
    n = len(x)
    a = mp.mpf(n) / 2 - 1

    def sf(u):  # beta(a, a) on [-1, 1]
        z = (u + 1) / 2
        return mp.betainc(a, a, z, 1, regularized=True)
    return {"statistic": r, "pvalue": pvalue_sym(r, sf, c["opts"]["alternative"])}


def o_spearmanr(c, data):
    if len(data[0]) != len(data[1]) or len(data[0]) < 3 or not finite(data[0], data[1]):
        return None
    rx, _ = ranks([num(v) for v in data[0]])
    ry, _ = ranks([num(v) for v in data[1]])
    rx = [mp.mpf(v.numerator) / v.denominator for v in rx]
    ry = [mp.mpf(v.numerator) / v.denominator for v in ry]
    mx, my = mean(rx), mean(ry)
    sxy = mp.fsum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sxx = mp.fsum((a - mx) ** 2 for a in rx)
    syy = mp.fsum((b - my) ** 2 for b in ry)
    if sxx == 0 or syy == 0:
        return None
    rs = sxy / mp.sqrt(sxx * syy)
    dof = len(rx) - 2
    if abs(rs) == 1:
        return {"statistic": rs}
    t = rs * mp.sqrt(dof / ((rs + 1) * (1 - rs)))
    return {"statistic": rs, "pvalue": pvalue_sym(t, lambda u: t_sf(u, dof), c["opts"]["alternative"])}


def o_linregress(c, data):
    if len(data[0]) != len(data[1]) or len(data[0]) < 3 or not finite(data[0], data[1]):
        return None
    x, y = [M(v) for v in data[0]], [M(v) for v in data[1]]
    n = len(x)
    mx, my = mean(x), mean(y)
    ssxm = mp.fsum((a - mx) ** 2 for a in x) / n
    ssym = mp.fsum((b - my) ** 2 for b in y) / n
    ssxym = mp.fsum((a - mx) * (b - my) for a, b in zip(x, y)) / n
    if ssxm == 0 or ssym == 0:
        return None
    r = ssxym / mp.sqrt(ssxm * ssym)
    slope = ssxym / ssxm
    out = {"slope": slope, "intercept": my - slope * mx, "rvalue": r}
    df = n - 2
    tiny = mp.mpf(1.0e-20)
    t = r * mp.sqrt(df / ((1 - r + tiny) * (1 + r + tiny)))
    out["pvalue"] = pvalue_sym(t, lambda u: t_sf(u, df), c["opts"]["alternative"])
    se = mp.sqrt((1 - r * r) * ssym / ssxm / df)
    out["stderr"] = se
    out["intercept_stderr"] = se * mp.sqrt(ssxm + mx * mx)
    return out


def kendall_counts(n):
    """Number of permutations of n with k inversions, k = 0..n(n-1)/2."""
    c = [1]
    for j in range(2, n + 1):
        nc = [0] * (len(c) + j - 1)
        run = 0
        for k in range(len(nc)):
            run += c[k] if k < len(c) else 0
            if k - j >= 0:
                run -= c[k - j]
            nc[k] = run
        c = nc
    return c


def o_kendalltau(c, data):
    x, y = [num(v) for v in data[0]], [num(v) for v in data[1]]
    if len(x) != len(y) or len(x) < 3 or not finite(data[0], data[1]):
        return None
    n = len(x)
    con = dis = xt = yt = nt = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx, dy = x[i] - x[j], y[i] - y[j]
            if dx == 0 and dy == 0:
                nt += 1
                xt += 1
                yt += 1
            elif dx == 0:
                xt += 1
            elif dy == 0:
                yt += 1
            elif (dx > 0) == (dy > 0):
                con += 1
            else:
                dis += 1
    tot = n * (n - 1) // 2
    if xt == tot or yt == tot:
        return None
    o = c["opts"]
    cmd = con - dis
    if o["variant"] == "b":
        tau = mp.mpf(cmd) / mp.sqrt(tot - xt) / mp.sqrt(tot - yt)
    else:
        m = min(len(set(x)), len(set(y)))
        tau = 2 * mp.mpf(cmd) / (mp.mpf(n) ** 2 * (m - 1) / m)
    out = {"statistic": tau}
    method = o["method"]
    if method == "exact" and (xt or yt):
        return None
    if method == "auto":
        method = "exact" if (xt == 0 and yt == 0 and (n <= 33 or min(dis, tot - dis) <= 1)) else "asymptotic"
    alt = o["alternative"]
    if method == "exact":
        cnt = kendall_counts(n)          # by number of inversions = discordant pairs
        total = math.factorial(n)
        le = Fraction(sum(cnt[:dis + 1]), total)   # P(D <= dis)  (tau large)
        ge = Fraction(sum(cnt[dis:]), total)       # P(D >= dis)  (tau small)
        if alt == "greater":
            p = le
        elif alt == "less":
            p = ge
        else:
            p = min(1, 2 * min(le, ge))
        out["pvalue"] = p
        return out
    if n < 3:
        return out
    xs = {}
    for v in x:
        xs[v] = xs.get(v, 0) + 1
    ys = {}
    for v in y:
        ys[v] = ys.get(v, 0) + 1
    x0 = sum(k * (k - 1) * (k - 2) for k in xs.values())
    x1 = sum(k * (k - 1) * (2 * k + 5) for k in xs.values())
    y0 = sum(k * (k - 1) * (k - 2) for k in ys.values())
    y1 = sum(k * (k - 1) * (2 * k + 5) for k in ys.values())
    m = mp.mpf(n) * (n - 1)
    v = (m * (2 * n + 5) - x1 - y1) / 18 + (2 * mp.mpf(xt) * yt) / m + mp.mpf(x0) * y0 / (9 * m * (n - 2))
    z = cmd / mp.sqrt(v)
    out["pvalue"] = pvalue_sym(z, norm_sf, alt)
    return out


# ---------------------------------------------------------- Kolmogorov

def kstwo_cdf_durbin(n, d):
    d = M(d)
    k = int(mp.ceil(n * d))
    h = k - n * d
    m = 2 * k - 1
    H = mp.matrix(m, m)
    for i in range(m):
        for j in range(m):
            if i - j + 1 >= 0:
                H[i, j] = mp.mpf(1) / mp.factorial(i - j + 1)
    for i in range(m):
        H[i, 0] -= h ** (i + 1) / mp.factorial(i + 1)
        H[m - 1, i] -= h ** (m - i) / mp.factorial(m - i)
    H[m - 1, 0] += (max(2 * h - 1, 0) ** m) / mp.factorial(m)
    P = H ** n
    return P[k - 1, k - 1] * mp.factorial(n) / mp.mpf(n) ** n


def smirnov_exact(n, d):
    d = M(d)
    if d <= 0:
        return mp.mpf(1)
    if d >= 1:
        return mp.mpf(0)
    s = mp.mpf(0)
    j = 0
    while j <= n and 1 - d - mp.mpf(j) / n > 0:
        s += mp.binomial(n, j) * (1 - d - mp.mpf(j) / n) ** (n - j) * (d + mp.mpf(j) / n) ** (j - 1)
        j += 1
    return d * s


def kstwo_sf_exact(n, d):
    """P(D_n > d), or None when too costly."""
    dd = M(d)
    if dd <= mp.mpf(0.5) / n:
        return mp.mpf(1)
    if dd >= 1:
        return mp.mpf(0)
    if dd >= 0.5:
        return 2 * smirnov_exact(n, d)          # exact for d >= 1/2
    if n * dd <= 25:
        return 1 - kstwo_cdf_durbin(n, d)
    return None


def kolmogorov_exact(x):
    x = M(x)
    if x <= 0:
        return mp.mpf(1)
    return 2 * mp.nsum(lambda k: (-1) ** (k - 1) * mp.exp(-2 * k * k * x * x), [1, mp.inf])


def o_kstwo_sf(c, data):
    n, d = int(num(data[0][0])), num(data[0][1])
    sf = kstwo_sf_exact(n, d)
    if sf is None:
        return None
    return {"sf": sf, "cdf": 1 - sf}


def o_smirnov(c, data):
    n, d = int(num(data[0][0])), num(data[0][1])
    if n > 5000:
        return None
    return {"sf": smirnov_exact(n, d)}


def o_kolmogorov(c, data):
    return {"sf": kolmogorov_exact(num(data[0][0]))}


def t4_cdf(x):
    t = (x - mp.mpf(0.1)) / mp.mpf(1.3)
    return 1 - t_sf(t, 4)


def o_ks_1samp(c, data):
    x = sorted(num(v) for v in data[0])
    if not x or not finite(x):
        return None
    o = c["opts"]
    cdf = norm_cdf if o["cdf"] == "norm" else t4_cdf
    n = len(x)
    cv = [cdf(M(v)) for v in x]
    dplus = max(mp.mpf(i + 1) / n - cv[i] for i in range(n))
    dminus = max(cv[i] - mp.mpf(i) / n for i in range(n))
    alt = o["alternative"]
    if alt == "greater":
        return {"statistic": dplus, "pvalue": smirnov_exact(n, dplus)}
    if alt == "less":
        return {"statistic": dminus, "pvalue": smirnov_exact(n, dminus)}
    d = max(dplus, dminus)
    out = {"statistic": d}
    method = o["method"]
    if method == "asymp":
        out["pvalue"] = kolmogorov_exact(d * mp.sqrt(n))
    elif method == "approx":
        out["pvalue"] = min(1, 2 * smirnov_exact(n, d))
    else:
        p = kstwo_sf_exact(n, d)
        if p is not None:
            out["pvalue"] = p
    return out


def ks2_counts(n1, n2, inside):
    """Number of lattice paths (0,0) -> (n1,n2) with every point satisfying inside(i, j)."""
    row = [0] * (n2 + 1)
    for i in range(n1 + 1):
        new = [0] * (n2 + 1)
        for j in range(n2 + 1):
            if not inside(i, j):
                continue
            if i == 0 and j == 0:
                new[j] = 1
                continue
            new[j] = (row[j] if i > 0 else 0) + (new[j - 1] if j > 0 else 0)
        row = new
    return row[n2]


def o_ks_2samp(c, data):
    a, b = [num(v) for v in data[0]], [num(v) for v in data[1]]
    if not a or not b or not finite(a, b):
        return None
    n1, n2 = len(a), len(b)
    o = c["opts"]
    method = o["method"]
    if method == "auto":
        method = "exact" if max(n1, n2) <= 10000 else "asymp"
    if method != "exact" or n1 * n2 > 400000:
        return None
    # D as the exact rational k/n1 - j/n2 at the pooled values
    sa, sb = sorted(a), sorted(b)
    vals = sorted(set(a) | set(b))
    i = j = 0
    dmax = Fraction(-2)
    dmin = Fraction(2)
    for v in vals:
        while i < n1 and sa[i] <= v:
            i += 1
        while j < n2 and sb[j] <= v:
            j += 1
        diff = Fraction(i, n1) - Fraction(j, n2)
        dmax, dmin = max(dmax, diff), min(dmin, diff)
    alt = o["alternative"]
    if alt == "less" or (alt == "two-sided" and -dmin > dmax):
        d = max(-dmin, 0)
    else:
        d = dmax
    if d == 0:
        return {"pvalue": 1.0}
    total = comb(n1 + n2, n1)
    if alt == "two-sided":
        inside = ks2_counts(n1, n2, lambda i, j: abs(Fraction(i, n1) - Fraction(j, n2)) < d)
    elif alt == "greater":
        inside = ks2_counts(n1, n2, lambda i, j: Fraction(i, n1) - Fraction(j, n2) < d)
    else:
        inside = ks2_counts(n1, n2, lambda i, j: Fraction(j, n2) - Fraction(i, n1) < d)
    return {"statistic": d, "pvalue": Fraction(total - inside, total)}


# ------------------------------------------------------------- rank tests

def mwu_counts(n1, n2, maxu):
    c = [0] * (maxu + 1)
    c[0] = 1
    for i in range(1, n1 + 1):
        for u in range(i, maxu + 1):
            c[u] += c[u - i]
        for u in range(maxu, n2 + i - 1, -1):
            c[u] -= c[u - n2 - i]
    return c


def o_mannwhitneyu(c, data):
    a, b = [num(v) for v in data[0]], [num(v) for v in data[1]]
    if not a or not b or not finite(a, b):
        return None
    n1, n2 = len(a), len(b)
    r, ties = ranks(a + b)
    u1 = sum(r[:n1]) - Fraction(n1 * (n1 + 1), 2)
    u2 = n1 * n2 - u1
    o = c["opts"]
    alt = o["alternative"]
    if alt == "greater":
        u, fct = u1, 1
    elif alt == "less":
        u, fct = u2, 1
    else:
        u, fct = max(u1, u2), 2
    method = o["method"]
    if method == "auto":
        method = "asymptotic" if (n1 > 8 and n2 > 8) or any(t > 1 for t in ties) else "exact"
    out = {"statistic": u1}
    if method == "exact":
        k = int(u)
        lo, hi = min(n1, n2), max(n1, n2)
        if k * min(n1, n2) > 30_000_000:
            return out
        total = comb(n1 + n2, n1)
        kc = n1 * n2 - k
        if k < kc:
            cnt = mwu_counts(lo, hi, k)
            sf = Fraction(total - sum(cnt) + cnt[k], total)
        else:
            cnt = mwu_counts(lo, hi, kc)
            sf = Fraction(sum(cnt), total)
        out["pvalue"] = min(1, sf * fct)
        return out
    n = n1 + n2
    tt = sum(t ** 3 - t for t in ties)
    s = mp.sqrt(mp.mpf(n1 * n2) / 12 * ((n + 1) - mp.mpf(tt) / (n * (n - 1))))
    numr = mp.mpf(u1.numerator) / u1.denominator - mp.mpf(n1 * n2) / 2
    if s == 0:
        return out
    if o["use_continuity"]:
        sign = 1 if alt == "greater" else -1 if alt == "less" else mp.sign(numr)
        numr -= mp.mpf(0.5) * sign
    z = numr / s
    out["pvalue"] = min(1, pvalue_sym(z, norm_sf, alt))
    if o["method"] == "asymptotic":
        out["zstatistic"] = z
    return out


def wilcoxon_counts(n):
    c = [1]
    for k in range(1, n + 1):
        nc = [0] * (len(c) + k)
        for i, v in enumerate(c):
            nc[i] += v
            nc[i + k] += v
        c = nc
    return c


def o_wilcoxon(c, data):
    if len(data) > 1 and len(data[1]) > 0:
        if len(data[0]) != len(data[1]):
            return None
        d = [num(a) - num(b) for a, b in zip(data[0], data[1])]
    else:
        d = [num(v) for v in data[0]]
    if not d or not finite(d):
        return None
    o = c.get("opts") or {}
    zm = o.get("zero_method", "wilcox")
    method = o.get("method", "auto")
    alt = o.get("alternative", "two-sided")
    corr = o.get("correction", False)
    n = len(d)
    nz = sum(1 for v in d if v == 0)
    if method == "auto" and n > 50:
        method = "asymptotic"
    keep = [v for v in d if v != 0] if zm == "wilcox" else d
    r, ties = ranks([abs(v) for v in keep])
    rp = sum((ri for ri, v in zip(r, keep) if v > 0), Fraction(0))
    rm = sum((ri for ri, v in zip(r, keep) if v < 0), Fraction(0))
    rz = sum((ri for ri, v in zip(r, keep) if v == 0), Fraction(0))
    if zm == "zsplit":
        rp += rz / 2
        rm += rz / 2
    count = len(keep)
    has_ties = any(t > 1 for t in ties)
    stat = min(rp, rm) if alt == "two-sided" else rp
    out = {"statistic": stat}
    perm = False
    if method == "auto":
        if not (has_ties or nz > 0):
            method = "exact"
        elif n <= 13:
            perm = True
        else:
            method = "asymptotic"
    if perm:
        w = [ri for ri, v in zip(r, keep) if v != 0]
        const = rz / 2 if zm == "zsplit" else 0
        obs = rp
        null = []
        for s in range(1 << len(w)):
            null.append(sum((w[i] for i in range(len(w)) if s >> i & 1), Fraction(0)) + const)
        gamma = abs(Fraction(2.0 ** -52) * 100 * obs)
        le = Fraction(sum(1 for v in null if v <= obs + gamma), len(null))
        ge = Fraction(sum(1 for v in null if v >= obs - gamma), len(null))
        out["pvalue"] = le if alt == "less" else ge if alt == "greater" else min(1, 2 * min(le, ge))
        return out
    if method == "exact":
        if zm == "wilcox":
            cnt_n = count
        else:
            cnt_n = n
        cnt = wilcoxon_counts(cnt_n)
        tot = 2 ** cnt_n
        mean_ = Fraction(cnt_n * (cnt_n + 1), 4)

        def pcdf(k):
            if k < 0:
                return Fraction(0)
            return Fraction(sum(cnt[:k + 1]), tot)

        def psf(k):
            return Fraction(sum(cnt[max(k, 0):]), tot)
        lo, hi = math.ceil(rp), math.floor(rp)
        if alt == "less":
            p = pcdf(lo)
        elif alt == "greater":
            p = psf(hi)
        else:
            p = min(1, 2 * min(psf(hi), pcdf(lo)))
        out["pvalue"] = p
        return out
    # asymptotic
    cn = mp.mpf(count)
    mn = cn * (cn + 1) / 4
    se = cn * (cn + 1) * (2 * cn + 1)
    tt = list(ties)
    if zm == "pratt":
        z0 = mp.mpf(nz)
        mn -= z0 * (z0 + 1) / 4
        se -= z0 * (z0 + 1) * (2 * z0 + 1)
        if nz > 0:
            tt[0] = 0
    se = mp.sqrt((se - mp.mpf(sum(t ** 3 - t for t in tt)) / 2) / 24)
    if se == 0:
        return out
    z = (mp.mpf(rp.numerator) / rp.denominator - mn) / se
    if corr:
        sign = 1 if alt == "greater" else -1 if alt == "less" else mp.sign(z)
        z -= sign * mp.mpf(0.5) / se
    out["pvalue"] = pvalue_sym(z, norm_sf, alt)
    if method == "asymptotic" and o.get("method") == "asymptotic":
        out["zstatistic"] = -abs(z) if alt == "two-sided" else z
    return out


def o_kruskal(c, data):
    if len(data) < 2 or any(len(g) == 0 for g in data) or not finite(*data):
        return None
    allv = [num(v) for g in data for v in g]
    r, ties = ranks(allv)
    n = len(allv)
    tc = 1 - Fraction(sum(t ** 3 - t for t in ties), n ** 3 - n)
    if tc == 0:
        return None
    off = 0
    ssbn = Fraction(0)
    for g in data:
        s = sum(r[off:off + len(g)])
        ssbn += s * s / len(g)
        off += len(g)
    h = (Fraction(12, n * (n + 1)) * ssbn - 3 * (n + 1)) / tc
    hm = mp.mpf(h.numerator) / h.denominator
    return {"statistic": h, "pvalue": chi2_sf(hm, len(data) - 1)}


# ------------------------------------------------------------- normality

def skew_kurt(a):
    m = mean(a)
    m2 = mean([(v - m) ** 2 for v in a])
    m3 = mean([(v - m) ** 3 for v in a])
    m4 = mean([(v - m) ** 4 for v in a])
    return m2, m3, m4


def skew_z(a):
    n = len(a)
    m2, m3, _ = skew_kurt(a)
    if m2 == 0:
        return None
    b2 = m3 / m2 ** mp.mpf(1.5)
    y = b2 * mp.sqrt(mp.mpf((n + 1) * (n + 3)) / (6 * (n - 2)))
    beta2 = mp.mpf(3) * (n * n + 27 * n - 70) * (n + 1) * (n + 3) / ((n - 2) * (n + 5) * (n + 7) * (n + 9))
    w2 = -1 + mp.sqrt(2 * (beta2 - 1))
    delta = 1 / mp.sqrt(mp.log(w2) / 2)
    alpha = mp.sqrt(2 / (w2 - 1))
    if y == 0:
        y = mp.mpf(1)
    return delta * mp.log(y / alpha + mp.sqrt((y / alpha) ** 2 + 1))


def kurt_z(a):
    n = mp.mpf(len(a))
    m2, _, m4 = skew_kurt(a)
    if m2 == 0:
        return None
    b2 = m4 / m2 ** 2
    e = 3 * (n - 1) / (n + 1)
    varb2 = 24 * n * (n - 2) * (n - 3) / ((n + 1) ** 2 * (n + 3) * (n + 5))
    x = (b2 - e) / mp.sqrt(varb2)
    sb1 = 6 * (n * n - 5 * n + 2) / ((n + 7) * (n + 9)) * mp.sqrt(6 * (n + 3) * (n + 5) / (n * (n - 2) * (n - 3)))
    A = 6 + 8 / sb1 * (2 / sb1 + mp.sqrt(1 + 4 / sb1 ** 2))
    term1 = 1 - 2 / (9 * A)
    den = 1 + x * mp.sqrt(2 / (A - 4))
    if den == 0:
        return None
    term2 = mp.sign(den) * mp.cbrt((1 - 2 / A) / abs(den))
    return (term1 - term2) / mp.sqrt(2 / (9 * A))


def o_skewtest(c, data):
    if len(data[0]) < 8 or not finite(data[0]):
        return None
    z = skew_z([M(v) for v in data[0]])
    if z is None:
        return None
    return {"statistic": z, "pvalue": pvalue_sym(z, norm_sf, c["opts"]["alternative"])}


def o_kurtosistest(c, data):
    if len(data[0]) < 5 or not finite(data[0]):
        return None
    z = kurt_z([M(v) for v in data[0]])
    if z is None:
        return None
    return {"statistic": z, "pvalue": pvalue_sym(z, norm_sf, c["opts"]["alternative"])}


def o_normaltest(c, data):
    if len(data[0]) < 8 or not finite(data[0]):
        return None
    a = [M(v) for v in data[0]]
    s, k = skew_z(a), kurt_z(a)
    if s is None or k is None:
        return None
    st = s * s + k * k
    return {"statistic": st, "pvalue": chi2_sf(st, 2)}


def o_jarque_bera(c, data):
    if len(data[0]) < 2 or not finite(data[0]):
        return None
    a = [M(v) for v in data[0]]
    m2, m3, m4 = skew_kurt(a)
    if m2 == 0:
        return None
    s = m3 / m2 ** mp.mpf(1.5)
    k = m4 / m2 ** 2 - 3
    st = mp.mpf(len(a)) / 6 * (s * s + k * k / 4)
    return {"statistic": st, "pvalue": chi2_sf(st, 2)}


def o_shapiro(c, data):
    x = [num(v) for v in data[0]]
    n = len(x)
    if n < 3 or not finite(x):
        return None
    c0 = x[n // 2]
    y = sorted(M(v) - M(c0) for v in x)   # SciPy subtracts in float64 after sorting: exact here
    y = [mp.mpf(float(v)) for v in y]
    my = mean(y)
    den = mp.fsum((v - my) ** 2 for v in y)
    if den == 0:
        return None
    if n == 3:
        h = mp.sqrt(2) / 2
        a = [-h, 0, h]
        w = mp.fsum(ai * yi for ai, yi in zip(a, y)) ** 2 / den
        w = min(max(w, mp.mpf(0.75)), 1)
        return {"statistic": w, "pvalue": max(0, min(1, 1 - 6 / mp.pi * mp.acos(mp.sqrt(w))))}
    m = [mp.sqrt(2) * mp.erfinv(2 * ((i + 1 - mp.mpf(3) / 8) / (n + mp.mpf(1) / 4)) - 1) for i in range(n)]
    u = mp.mpf(n) ** -0.5
    mTm = mp.fsum(v * v for v in m)
    cc = [v / mp.sqrt(mTm) for v in m]
    an = cc[-1] + mp.mpf(0.221157) * u - mp.mpf(0.147981) * u ** 2 - mp.mpf(2.071190) * u ** 3 \
        + mp.mpf(4.434685) * u ** 4 - mp.mpf(2.706056) * u ** 5
    anm1 = cc[-2] + mp.mpf(0.042981) * u - mp.mpf(0.293762) * u ** 2 - mp.mpf(1.752461) * u ** 3 \
        + mp.mpf(5.682633) * u ** 4 - mp.mpf(3.582633) * u ** 5
    if n <= 5:
        phi = (mTm - 2 * m[-1] ** 2) / (1 - 2 * an ** 2)
    else:
        phi = (mTm - 2 * m[-1] ** 2 - 2 * m[-2] ** 2) / (1 - 2 * an ** 2 - 2 * anm1 ** 2)
    a = [v / mp.sqrt(phi) for v in m]
    if n > 5:
        a[-2], a[1] = anm1, -anm1
    a[-1], a[0] = an, -an
    w = mp.fsum(ai * yi for ai, yi in zip(a, y)) ** 2 / den
    if n <= 11:
        nn = mp.mpf(n)
        gamma = mp.mpf(-2.273) + mp.mpf(0.459) * nn
        mu = mp.mpf(0.5440) - mp.mpf(0.39978) * nn + mp.mpf(0.025054) * nn ** 2 - mp.mpf(0.0006714) * nn ** 3
        sigma = mp.exp(mp.mpf(1.3822) - mp.mpf(0.77857) * nn + mp.mpf(0.062767) * nn ** 2 - mp.mpf(0.0020322) * nn ** 3)
        arg = gamma - mp.log(1 - w)
        if arg <= 0:
            return {"statistic": w}
        g = -mp.log(arg)
    else:
        lu = mp.log(n)
        mu = mp.mpf(-1.5861) - mp.mpf(0.31082) * lu - mp.mpf(0.083751) * lu ** 2 + mp.mpf(0.0038915) * lu ** 3
        sigma = mp.exp(mp.mpf(-0.4803) - mp.mpf(0.082676) * lu + mp.mpf(0.0030302) * lu ** 2)
        g = mp.log(1 - w)
    z = (g - mu) / sigma
    return {"statistic": w, "pvalue": norm_sf(z)}


def o_anderson(c, data, scipy_out):
    dist = c["opts"]["dist"]
    if dist == "logistic":
        return None
    x = [M(v) for v in data[0]]
    y = sorted(x)
    n = len(y)
    xbar = mean(x)
    if dist == "norm":
        loc, scale = xbar, mp.sqrt(var(x, 1))
        lc = [mp.log(norm_cdf((v - loc) / scale)) for v in y]
        ls = [mp.log(norm_sf((v - loc) / scale)) for v in y]
    elif dist == "expon":
        loc, scale = mp.mpf(0), xbar
        lc = [mp.log(-mp.expm1(-v / xbar)) for v in y]
        ls = [-v / xbar for v in y]
    else:
        dd = [-v for v in x] if dist == "gumbel_l" else x
        m0 = min(dd)
        dv = [v - m0 for v in dd]

        def func(s):
            w = [mp.exp(-v / s) for v in dv]
            return mean(dv) - mp.fsum(v * wi for v, wi in zip(dv, w)) / mp.fsum(w) - s
        scale = mp.findroot(func, M(scipy_out["scale"]))
        lr = m0 - scale * (mp.log(mp.fsum(mp.exp(-v / scale) for v in dv)) - mp.log(n))
        loc = -lr if dist == "gumbel_l" else lr
        if dist == "gumbel_r":
            lc = [-mp.exp(-(v - loc) / scale) for v in y]
            ls = [mp.log(-mp.expm1(-mp.exp(-(v - loc) / scale))) for v in y]
        else:
            lc = [mp.log(-mp.expm1(-mp.exp((v - loc) / scale))) for v in y]
            ls = [-mp.exp((v - loc) / scale) for v in y]
    a2 = -n - mp.fsum((2 * mp.mpf(i + 1) - 1) / n * (lc[i] + ls[n - 1 - i]) for i in range(n))
    crit = [num(v) for v in scipy_out["critical_values"]]
    sig = [num(v) / 100 for v in scipy_out["significance_level"]]
    a2f = float(a2)
    if a2f <= crit[0]:
        p = sig[0]
    elif a2f >= crit[-1]:
        p = sig[-1]
    else:
        j = 0
        while crit[j + 1] <= a2f:
            j += 1
        p = (mp.mpf(sig[j + 1]) - sig[j]) / (mp.mpf(crit[j + 1]) - crit[j]) * (a2 - crit[j]) + sig[j]
    return {"statistic": a2, "pvalue": p, "loc": loc, "scale": scale}


# ----------------------------------------------------------------- ANOVA

def o_f_oneway(c, data):
    if len(data) < 2 or any(len(g) == 0 for g in data) or not finite(*data):
        return None
    gs = [[M(v) for v in g] for g in data]
    if all(len(g) == 1 for g in gs):
        return None
    if all(len(set(g)) == 1 for g in gs):
        return None
    k = len(gs)
    allv = [v for g in gs for v in g]
    N = len(allv)
    if c["opts"]["equal_var"]:
        gm = mean(allv)
        ssbn = mp.fsum(len(g) * (mean(g) - gm) ** 2 for g in gs)
        sswn = mp.fsum(mp.fsum((v - mean(g)) ** 2 for v in g) for g in gs)
        dfn, dfd = k - 1, N - k
        if sswn == 0:
            return None
        F = (ssbn / dfn) / (sswn / dfd)
    else:
        if any(len(g) < 2 for g in gs):
            return None
        w = [len(g) / var(g, 1) for g in gs]
        sw = mp.fsum(w)
        yt = [mean(g) for g in gs]
        yh = mp.fsum(wi * y for wi, y in zip(w, yt)) / sw
        q = mp.fsum(1 / mp.mpf(len(g) - 1) * (1 - wi / sw) ** 2 for g, wi in zip(gs, w))
        F = mp.fsum(wi * (y - yh) ** 2 for wi, y in zip(w, yt)) / (k - 1) / (1 + 2 * mp.mpf(k - 2) / (k * k - 1) * q)
        dfn, dfd = k - 1, (k * k - 1) / (3 * q)
    return {"statistic": F, "pvalue": f_sf(F, dfn, dfd)}


def o_bartlett(c, data):
    if len(data) < 2 or any(len(g) < 2 for g in data) or not finite(*data):
        return None
    gs = [[M(v) for v in g] for g in data]
    k = len(gs)
    ni = [len(g) for g in gs]
    ss = [var(g, 1) for g in gs]
    if any(s == 0 for s in ss):
        return None
    nt = sum(ni)
    sp = mp.fsum((n - 1) * s for n, s in zip(ni, ss)) / (nt - k)
    numer = (nt - k) * mp.log(sp) - mp.fsum((n - 1) * mp.log(s) for n, s in zip(ni, ss))
    denom = 1 + mp.mpf(1) / (3 * (k - 1)) * (mp.fsum(mp.mpf(1) / (n - 1) for n in ni) - mp.mpf(1) / (nt - k))
    T = numer / denom
    return {"statistic": max(T, 0), "pvalue": chi2_sf(T, k - 1)}


def o_levene(c, data):
    if len(data) < 2 or any(len(g) == 0 for g in data) or not finite(*data):
        return None
    gs = [[M(v) for v in g] for g in data]
    center = c["opts"]["center"]
    cut = c["opts"]["proportiontocut"]
    zs = []
    for g in gs:
        s = sorted(g)
        if center == "median":
            n = len(s)
            cc = s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2
        elif center == "mean":
            cc = mean(s)
        else:
            lc = int(cut * len(s))
            cc = mean(s[lc:len(s) - lc])
        zs.append([abs(v - cc) for v in g])
    k = len(gs)
    nt = sum(len(g) for g in gs)
    zb = [mean(z) for z in zs]
    zbar = mp.fsum(len(z) * m for z, m in zip(zs, zb)) / nt
    numer = (nt - k) * mp.fsum(len(z) * (m - zbar) ** 2 for z, m in zip(zs, zb))
    denom = (k - 1) * mp.fsum(mp.fsum((v - m) ** 2 for v in z) for z, m in zip(zs, zb))
    if denom == 0:
        return None
    W = numer / denom
    return {"statistic": W, "pvalue": f_sf(W, k - 1, nt - k)}


ORACLES = {
    "ttest_1samp": o_ttest_1samp, "ttest_ind": o_ttest_ind, "ttest_rel": o_ttest_rel,
    "chisquare": o_chisquare, "power_divergence": o_power, "chi2_contingency": o_chi2_contingency,
    "fisher_exact": o_fisher_exact, "pearsonr": o_pearsonr, "spearmanr": o_spearmanr,
    "linregress": o_linregress, "kendalltau": o_kendalltau, "kstwo_sf": o_kstwo_sf,
    "smirnov": o_smirnov, "kolmogorov": o_kolmogorov, "ks_1samp": o_ks_1samp, "ks_2samp": o_ks_2samp,
    "mannwhitneyu": o_mannwhitneyu, "wilcoxon": o_wilcoxon, "wilcoxon1": o_wilcoxon,
    "kruskal": o_kruskal, "skewtest": o_skewtest, "kurtosistest": o_kurtosistest,
    "normaltest": o_normaltest, "jarque_bera": o_jarque_bera, "shapiro": o_shapiro,
    "f_oneway": o_f_oneway, "bartlett": o_bartlett, "levene": o_levene,
}


def to_float(v):
    if isinstance(v, Fraction):
        return f(v.numerator / v.denominator if abs(v.numerator) < 2 ** 1000 else mp.mpf(v.numerator) / v.denominator)
    return f(v)


def main():
    groups = sys.argv[1:] or [n[len("tests_"):-len(".json.gz")] for n in sorted(os.listdir(SRC))
                              if n.startswith("tests_")]
    os.makedirs(OUT, exist_ok=True)
    for g in groups:
        path = os.path.join(SRC, f"tests_{g}.json.gz")
        raw = open(path, "rb").read()
        fx = json.loads(gzip.decompress(raw))
        exact = {}
        for i, c in enumerate(fx["cases"]):
            if c.get("error") or c.get("crash"):
                continue
            fn = ORACLES.get(c["fn"])
            if c["fn"] == "anderson":
                fn = lambda c_, d_, out=c["out"]: o_anderson(c_, d_, out)  # noqa: E731
            if fn is None:
                continue
            data = [fx["datasets"][k] for k in c["data"]]
            res = fn(c, data)
            if not res:
                continue
            exact[str(i)] = {k: to_float(v) for k, v in res.items() if k in c["out"]}
        out = os.path.join(OUT, f"tests_{g}.json.gz")
        with gzip.open(out, "wt", compresslevel=9) as fh:
            json.dump({"source_sha256": hashlib.sha256(raw).hexdigest(), "exact": exact}, fh,
                      separators=(",", ":"))
        print(f"{out}: {len(exact)} cases")


if __name__ == "__main__":
    main()
