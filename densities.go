package stats

import (
	"math"

	"github.com/go-numerics/special"
	"github.com/go-statistics/stats/internal/dd"
)

// The densities of the gamma and beta families, shared by Gamma,
// ChiSquared, Chi, InverseGamma, Beta, F and the discrete distributions
// built on them. Each is computed as m·2^k with a double-double mantissa m
// (dd.ExpMant) so that the caller's scale factors are multiplied in before
// the one final rounding.

// stirlingMin is where the Stirling forms take over from the direct ones.
const stirlingMin = 30

// gammaExponent returns, for x > 0 (a double-double, so that callers can
// pass x²/2 or 1/x unrounded), a > 0, the exponent E and the double-double
// factor c with x^(a-1) e^(-x) / Γ(a) = c · exp(E).
//
// For a < 30, E = (a-1) log x - x carried in double-double and
// c = 1/Γ(a). Beyond, the factors of Γ(a) are taken out analytically
// (Stirling, as in Boost's regularised_gamma_prefix):
//
//	x^(a-1) e^(-x)/Γ(a) = sqrt(a/2π)/x · exp(-a·g((x-a)/a) - corr(a)),
//
// g(u) = u - log(1+u) and corr the Stirling series, so that nothing large
// cancels: E is small near the mode however large a is.
func gammaExponent(a float64, x dd.DD) (e, c dd.DD) {
	if a < stirlingMin {
		am1h, am1l := dd.TwoSum(a, -1)
		e = dd.Sub(dd.Mul(dd.DD{Hi: am1h, Lo: am1l}, logDD(x)), x)
		g := special.Gamma(a)
		if math.IsInf(g, 1) { // a below 1/MaxFloat64: 1/Γ(a) = a/Γ(1+a) ≈ a
			return e, dd.F(a)
		}
		return e, dd.Div(dd.F(1), dd.F(g))
	}
	u := dd.DivF(dd.AddF(x, -a), a)
	g := gOf(u, func() dd.DD { return dd.Sub(logDD(x), dd.Log(a)) }) // log(1+u) = log(x/a)
	e = dd.AddF(dd.Neg(dd.MulF(g, a)), -stirlingCorr(a))
	c = dd.Div(dd.Sqrt(dd.MulF(inv2Pi, a)), x)
	return e, c
}

// gammaDensity returns x^(a-1) e^(-x) / Γ(a) times the double-double s,
// rounded once.
func gammaDensity(a float64, x dd.DD, s dd.DD) float64 {
	e, c := gammaExponent(a, x)
	return expScaled(e, dd.Mul(c, s))
}

// gammaLogDensity returns log(x^(a-1) e^(-x) / Γ(a)) as a double-double.
func gammaLogDensity(a float64, x dd.DD) dd.DD {
	if a < stirlingMin {
		am1h, am1l := dd.TwoSum(a, -1)
		e := dd.Sub(dd.Mul(dd.DD{Hi: am1h, Lo: am1l}, logDD(x)), x)
		if g := special.Gamma(a); !math.IsInf(g, 1) {
			// log of Γ(a) (within an ulp): an absolute error of 2^-53,
			// where log Γ(a) rounded carries ulp(log Γ(a))
			return dd.Sub(e, dd.Log(g))
		}
		return dd.AddF(e, -special.LogGamma(a))
	}
	e, c := gammaExponent(a, x)
	return dd.Add(e, logDD(c))
}

// betaExponent returns, for 0 < x < 1 with y = 1-x, both given as
// double-doubles (so that 1-x is not rounded: the factor y^(b-1) magnifies
// a relative error of y b-fold), the exponent E and the factor c with
// x^(a-1) y^(b-1) / B(a,b) = c · exp(E).
//
//   - otherwise, while B(a,b) is a normal float64: E = (a-1) log x +
//     (b-1) log y in double-double, c = 1/B(a,b);
//   - a, b >= 30: with s = a+b and d = x·b - y·a,
//     x^a y^b / B(a,b) = sqrt(ab/(2πs)) · exp(-a·g(d/a) - b·g(-d/b) +
//     corr(s) - corr(a) - corr(b)), g(u) = u - log(1+u) (log(1+u) taken as
//     log(x·s/a) directly when x·s/a is far from 1), the form of Boost's
//     ibeta_power_terms, free of cancellation near the mode;
//   - beyond: E = (a-1) log x + (b-1) log y - log B(a,b), c = 1.
func betaExponent(a, b float64, x, y dd.DD) (e, c dd.DD) {
	lx, ly := logDD(x), logDD(y)
	if a >= stirlingMin && b >= stirlingMin {
		d := dd.Sub(dd.MulF(x, b), dd.MulF(y, a))
		sh, sl := dd.TwoSum(a, b)
		ls := logDD(dd.DD{Hi: sh, Lo: sl})
		// log(1 + d/a) = log(x s / a), log(1 - d/b) = log(y s / b)
		ga := gOf(dd.DivF(d, a), func() dd.DD { return dd.Sub(dd.Add(lx, ls), dd.Log(a)) })
		gb := gOf(dd.Neg(dd.DivF(d, b)), func() dd.DD { return dd.Sub(dd.Add(ly, ls), dd.Log(b)) })
		e = dd.Neg(dd.Add(dd.MulF(ga, a), dd.MulF(gb, b)))
		e = dd.AddF(e, stirlingCorr(sh)-stirlingCorr(a)-stirlingCorr(b))
		// c = sqrt(ab/(2πs)) / (x y)
		c = dd.Sqrt(dd.Div(dd.Mul(dd.MulFF(a, b), inv2Pi), dd.DD{Hi: sh, Lo: sl}))
		c = dd.Div(c, dd.Mul(x, y))
		return e, c
	}
	am1h, am1l := dd.TwoSum(a, -1)
	bm1h, bm1l := dd.TwoSum(b, -1)
	e = dd.Add(dd.Mul(dd.DD{Hi: am1h, Lo: am1l}, lx), dd.Mul(dd.DD{Hi: bm1h, Lo: bm1l}, ly))
	if bf := special.Beta(a, b); bf > 0x1p-1000 && !math.IsInf(bf, 1) {
		return e, dd.Div(dd.F(1), dd.F(bf))
	}
	return dd.AddF(e, -special.LogBeta(a, b)), dd.F(1)
}

// gOf returns g(u) = u - log(1+u) >= 0: the series near 0 (log1pmxNeg),
// u - log1p(u) up to |u| = 1/2, and u - l() beyond, where l returns
// log(1+u) computed by the caller from the quantities u was made of (when
// 1+u is small, u itself has lost the digits of 1+u).
func gOf(u dd.DD, l func() dd.DD) dd.DD {
	if math.Abs(u.Hi) <= 0.5 {
		return log1pmxNeg(u)
	}
	return dd.Sub(u, l())
}

// betaDensity returns x^(a-1) y^(b-1) / B(a,b) times s, rounded once.
func betaDensity(a, b float64, x, y, s dd.DD) float64 {
	e, c := betaExponent(a, b, x, y)
	return expScaled(e, dd.Mul(c, s))
}

// betaLogDensity returns log(x^(a-1) y^(b-1) / B(a,b)) as a double-double.
func betaLogDensity(a, b float64, x, y dd.DD) dd.DD {
	e, c := betaExponent(a, b, x, y)
	return dd.Add(e, logDD(c))
}

// oneMinus returns 1 - x as an exact double-double.
func oneMinus(x float64) dd.DD {
	h, l := dd.TwoSum(1, -x)
	return dd.DD{Hi: h, Lo: l}
}

// logUpperGamma is log Q(a, y) for y > a + 1 (the upper tail, where Q may
// underflow): log of the gamma density at y plus log y plus the log of
// Legendre's continued fraction Γ(a,y) = e^-y y^a · 1/(y+1-a- 1(1-a)/(y+3-a-
// 2(2-a)/(y+5-a- ...))) (modified Lentz). Below y = a + 1 it is log Q.
func logUpperGamma(a, y float64) dd.DD {
	if !(y > a+1) {
		return dd.F(math.Log(special.GammaQ(a, y)))
	}
	const tiny = 1e-300
	b := y + 1 - a
	c := 1 / tiny
	d := 1 / b
	h := d
	for i := 1.0; i < 1000; i++ {
		an := -i * (i - a)
		b += 2
		d = an*d + b
		if math.Abs(d) < tiny {
			d = tiny
		}
		c = b + an/c
		if math.Abs(c) < tiny {
			c = tiny
		}
		d = 1 / d
		del := d * c
		h *= del
		if math.Abs(del-1) < 1e-17 {
			break
		}
	}
	// Γ(a,y)/Γ(a) = [y^(a-1) e^-y / Γ(a)] · y · h
	return dd.Add(gammaLogDensity(a, dd.F(y)), dd.Add(dd.Log(y), dd.Log(h)))
}
