"""ULP fixtures of the gamma and beta families, t and F (standard form)."""

import mpmath
from mpmath import mpf
import scipy.stats as st

import contfam
from mpincomplete import gam_P_Q, ib_I_Ic

inf = mpf("inf")


def lu(r, lo, hi):
    import math
    return math.exp(r.uniform(math.log(lo), math.log(hi)))


GAMMA_SHAPES = {
    "small": lambda r: (lu(r, 1e-3, 1.0),),
    "mid": lambda r: (lu(r, 1.0, 30.0),),
    "large": lambda r: (lu(r, 30.0, 1e4),),
    "huge": lambda r: (lu(r, 1e4, 1e6),),
}
GAMMA_COUNTS = {"huge": 12}


def g_pdf(a, x):
    a, x = mpf(a), mpf(x)
    return mpmath.exp((a - 1) * mpmath.log(x) - x - mpmath.loggamma(a))


def g_cdf(a, x):
    return gam_P_Q(mpf(a), mpf(x))[0]


def g_sf(a, x):
    return gam_P_Q(mpf(a), mpf(x))[1]


def b_pdf(a, b, x):
    a, b, x = mpf(a), mpf(b), mpf(x)
    return mpmath.exp((a - 1) * mpmath.log(x) + (b - 1) * mpmath.log1p(-x)
                      + mpmath.loggamma(a + b) - mpmath.loggamma(a) - mpmath.loggamma(b))


def b_cdf(a, b, x):
    return ib_I_Ic(mpf(a), mpf(b), mpf(x))[0]


def b_sf(a, b, x):
    return ib_I_Ic(mpf(a), mpf(b), mpf(x))[1]


def t_pdf(nu, x):
    nu, x = mpf(nu), mpf(x)
    return mpmath.exp(mpmath.loggamma((nu + 1) / 2) - mpmath.loggamma(nu / 2)
                      - mpmath.log(nu * mpmath.pi) / 2 - (nu + 1) / 2 * mpmath.log1p(x * x / nu))


def t_cdf(nu, x):
    nu, x = mpf(nu), mpf(x)
    tail = ib_I_Ic(nu / 2, mpf(1) / 2, nu / (nu + x * x))[0] / 2
    return tail if x <= 0 else 1 - tail


def t_sf(nu, x):
    return t_cdf(nu, -mpf(x))


def f_pdf(m, n, x):
    m, n, x = mpf(m), mpf(n), mpf(x)
    z = m * x / (m * x + n)
    return b_pdf(m / 2, n / 2, z) * m * n / (m * x + n) ** 2


def f_cdf(m, n, x):
    m, n, x = mpf(m), mpf(n), mpf(x)
    return ib_I_Ic(m / 2, n / 2, m * x / (m * x + n))[0]


def f_sf(m, n, x):
    m, n, x = mpf(m), mpf(n), mpf(x)
    return ib_I_Ic(n / 2, m / 2, n / (m * x + n))[0]


FAMILIES = [
    dict(name="gamma", dist=st.gamma, shapes=GAMMA_SHAPES, counts=GAMMA_COUNTS, pdf=g_pdf, cdf=g_cdf, sf=g_sf,
         support=lambda a: (mpf(0), inf)),
    dict(name="chi2", dist=st.chi2,
         shapes={k: (lambda f: (lambda r: (2 * f(r)[0],)))(f) for k, f in GAMMA_SHAPES.items()},
         counts=GAMMA_COUNTS,
         pdf=lambda df, x: g_pdf(mpf(df) / 2, mpf(x) / 2) / 2,
         cdf=lambda df, x: g_cdf(mpf(df) / 2, mpf(x) / 2),
         sf=lambda df, x: g_sf(mpf(df) / 2, mpf(x) / 2),
         support=lambda df: (mpf(0), inf)),
    dict(name="chi", dist=st.chi,
         shapes={k: (lambda f: (lambda r: (2 * f(r)[0],)))(f) for k, f in GAMMA_SHAPES.items()},
         counts=GAMMA_COUNTS,
         pdf=lambda df, x: g_pdf(mpf(df) / 2, mpf(x) ** 2 / 2) * mpf(x),
         cdf=lambda df, x: g_cdf(mpf(df) / 2, mpf(x) ** 2 / 2),
         sf=lambda df, x: g_sf(mpf(df) / 2, mpf(x) ** 2 / 2),
         support=lambda df: (mpf(0), inf)),
    dict(name="invgamma", dist=st.invgamma, shapes=GAMMA_SHAPES, counts=GAMMA_COUNTS,
         pdf=lambda a, x: g_pdf(a, 1 / mpf(x)) / mpf(x) ** 2,
         cdf=lambda a, x: g_sf(a, 1 / mpf(x)),
         sf=lambda a, x: g_cdf(a, 1 / mpf(x)),
         support=lambda a: (mpf(0), inf)),
    dict(name="beta", dist=st.beta,
         shapes={
             "small": lambda r: (lu(r, 1e-2, 1.0), lu(r, 1e-2, 1.0)),
             "mid": lambda r: (lu(r, 1.0, 30.0), lu(r, 1.0, 30.0)),
             "large": lambda r: (lu(r, 30.0, 1e4), lu(r, 30.0, 1e4)),
             "mixed": lambda r: (lu(r, 1e-2, 5.0), lu(r, 30.0, 1e4)),
         },
         counts={"large": 15, "mixed": 30},
         pdf=b_pdf, cdf=b_cdf, sf=b_sf, support=lambda a, b: (mpf(0), mpf(1))),
    dict(name="t", dist=st.t,
         shapes={
             "small": lambda r: (lu(r, 0.1, 1.0),),
             "mid": lambda r: (lu(r, 1.0, 30.0),),
             "large": lambda r: (lu(r, 30.0, 1e4),),
             "huge": lambda r: (lu(r, 1e4, 1e8),),
         },
         counts={"large": 30, "huge": 12},
         pdf=t_pdf, cdf=t_cdf, sf=t_sf, support=lambda nu: (-inf, inf)),
    dict(name="f", dist=st.f,
         shapes={
             "small": lambda r: (lu(r, 0.1, 2.0), lu(r, 0.1, 2.0)),
             "mid": lambda r: (lu(r, 1.0, 30.0), lu(r, 1.0, 30.0)),
             "large": lambda r: (lu(r, 30.0, 1e4), lu(r, 30.0, 1e4)),
             "mixed": lambda r: (lu(r, 1.0, 10.0), lu(r, 100.0, 1e4)),
         },
         counts={"large": 15, "mixed": 30},
         pdf=f_pdf, cdf=f_cdf, sf=f_sf, support=lambda m, n: (mpf(0), inf)),
]

FIXTURES = {}
for fam in FAMILIES:
    FIXTURES.update(contfam.build(fam))


def _ncx2_sum(df, nc, x, upper):
    df, nc, x = mpf(df), mpf(nc), mpf(x)
    mu = nc / 2
    y = x / 2
    j0 = int(mu)
    total = mpf(0)
    # from the Poisson mode outwards, until the terms stop mattering
    for direction in (range(j0, -1, -1), range(j0 + 1, 10 ** 7)):
        prev = None
        for j in direction:
            w = mpmath.exp(-mu + j * mpmath.log(mu) - mpmath.loggamma(j + 1))
            pq = gam_P_Q(df / 2 + j, y)
            t = w * (pq[1] if upper else pq[0])
            total += t
            if prev is not None and t < prev and t < total * mpf(2) ** (-mpmath.mp.prec - 10) \
                    and abs(j - mu) > 10 * mpmath.sqrt(mu + 1):
                break
            prev = t
    return total


def ncx2_pdf(df, nc, x):
    df, nc, x = mpf(df), mpf(nc), mpf(x)
    k = df / 2 - 1
    return mpmath.exp(-(x + nc) / 2) * (x / nc) ** (k / 2) * mpmath.besseli(k, mpmath.sqrt(nc * x)) / 2


FAMILIES.append(
    dict(name="ncx2", dist=st.ncx2,
         shapes={
             "small": lambda r: (lu(r, 0.5, 4.0), lu(r, 0.1, 10.0)),
             "mid": lambda r: (lu(r, 4.0, 50.0), lu(r, 1.0, 100.0)),
             "large": lambda r: (lu(r, 1.0, 50.0), lu(r, 100.0, 2000.0)),
         },
         counts={"small": 30, "mid": 30, "large": 10},
         pdf=ncx2_pdf, cdf=lambda df, nc, x: _ncx2_sum(df, nc, x, False),
         sf=lambda df, nc, x: _ncx2_sum(df, nc, x, True),
         support=lambda df, nc: (mpf(0), inf)))
FIXTURES.update(contfam.build(FAMILIES[-1]))
