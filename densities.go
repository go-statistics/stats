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
	e = dd.AddF(dd.Neg(dd.MulF(log1pmxNeg(u), a)), -stirlingCorr(a))
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
		return dd.AddF(e, -special.LogGamma(a))
	}
	e, c := gammaExponent(a, x)
	return dd.Add(e, logDD(c))
}

// betaExponent returns, for 0 < x < 1 with y = 1-x given separately (so
// that a caller who knows 1-x better than fl(1-x) can say so), the exponent
// E and the factor c with x^(a-1) y^(b-1) / B(a,b) = c · exp(E).
//
//   - a, b < 30: E = (a-1) log x + (b-1) log y in double-double, c = 1/B(a,b);
//   - a, b >= 30: with s = a+b and d = x·b - y·a (exact through TwoProd),
//     x^a y^b / B(a,b) = sqrt(ab/(2πs)) · exp(-a·g(d/a) - b·g(-d/b) +
//     corr(s) - corr(a) - corr(b)), g(u) = u - log(1+u), the form of
//     Boost's ibeta_power_terms, free of cancellation near the mode;
//   - otherwise: E = (a-1) log x + (b-1) log y - log B(a,b), c = 1.
func betaExponent(a, b, x, y float64) (e, c dd.DD) {
	lx, ly := dd.Log(x), dd.Log(y)
	if a < stirlingMin && b < stirlingMin {
		am1h, am1l := dd.TwoSum(a, -1)
		bm1h, bm1l := dd.TwoSum(b, -1)
		e = dd.Add(dd.Mul(dd.DD{Hi: am1h, Lo: am1l}, lx), dd.Mul(dd.DD{Hi: bm1h, Lo: bm1l}, ly))
		bf := special.Beta(a, b)
		if bf > 0 && !math.IsInf(bf, 1) {
			return e, dd.Div(dd.F(1), dd.F(bf))
		}
		return dd.AddF(e, -special.LogBeta(a, b)), dd.F(1)
	}
	if a >= stirlingMin && b >= stirlingMin {
		p1, e1 := dd.TwoProd(x, b)
		p2, e2 := dd.TwoProd(y, a)
		dh, dl := dd.TwoSum(p1, -p2)
		d := dd.DD{Hi: dh, Lo: dl + (e1 - e2)}
		s := a + b
		ga := log1pmxNeg(dd.DivF(d, a))
		gb := log1pmxNeg(dd.Neg(dd.DivF(d, b)))
		e = dd.Neg(dd.Add(dd.MulF(ga, a), dd.MulF(gb, b)))
		e = dd.AddF(e, stirlingCorr(s)-stirlingCorr(a)-stirlingCorr(b))
		// c = sqrt(ab/(2πs)) / (x y)
		ab := dd.MulFF(a, b)
		c = dd.Sqrt(dd.DivF(dd.Mul(ab, inv2Pi), s))
		c = dd.Div(c, dd.MulFF(x, y))
		return e, c
	}
	am1h, am1l := dd.TwoSum(a, -1)
	bm1h, bm1l := dd.TwoSum(b, -1)
	e = dd.Add(dd.Mul(dd.DD{Hi: am1h, Lo: am1l}, lx), dd.Mul(dd.DD{Hi: bm1h, Lo: bm1l}, ly))
	return dd.AddF(e, -special.LogBeta(a, b)), dd.F(1)
}

// betaDensity returns x^(a-1) y^(b-1) / B(a,b) times s, rounded once.
func betaDensity(a, b, x, y float64, s dd.DD) float64 {
	e, c := betaExponent(a, b, x, y)
	return expScaled(e, dd.Mul(c, s))
}

// betaLogDensity returns log(x^(a-1) y^(b-1) / B(a,b)) as a double-double.
func betaLogDensity(a, b, x, y float64) dd.DD {
	e, c := betaExponent(a, b, x, y)
	return dd.Add(e, logDD(c))
}
