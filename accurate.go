package stats

import (
	"math"

	"github.com/go-statistics/stats/internal/dd"
)

// Double-double constants (hi + lo), from mpmath at 300 bits.
var (
	invSqrt2Pi  = dd.DD{Hi: 0.3989422804014327, Lo: -2.49232720227773e-17}
	logSqrt2Pi  = dd.DD{Hi: 0.9189385332046728, Lo: -3.8782941580672414e-17}
	sqrt2OverPi = dd.DD{Hi: 0.7978845608028654, Lo: -4.98465440455546e-17}
)

// halfSquare returns x²/2 as an exact double-double (barring underflow).
func halfSquare(x float64) dd.DD {
	h, l := dd.TwoProd(x, x)
	return dd.DD{Hi: 0.5 * h, Lo: 0.5 * l}
}

// negHalfSq returns -x²/2 as an exact double-double (barring underflow).
func negHalfSq(x float64) dd.DD {
	h, l := dd.TwoProd(x, x)
	return dd.DD{Hi: -0.5 * h, Lo: -0.5 * l}
}

// expScaled returns c·exp(a) rounded once: the exponential's double-double
// mantissa is multiplied by c before the final rounding, so the result is
// accurate to about 2^-70 relative before that rounding (dd.ExpMant).
func expScaled(a, c dd.DD) float64 {
	m, k := dd.ExpMant(a)
	p := dd.Mul(m, c)
	return math.Ldexp(p.Hi+p.Lo, k)
}

var (
	inv2Pi     = dd.DD{Hi: 0.15915494309189535, Lo: -9.839338337591243e-18}
	halfLog2Pi = dd.DD{Hi: 0.9189385332046728, Lo: -3.8782941580672414e-17}
)

// logDD returns log(a) for a double-double a > 0: log(a.Hi) + a.Lo/a.Hi,
// the second term being the first order of log1p(a.Lo/a.Hi), exact to
// 2^-106 since |a.Lo/a.Hi| <= 2^-53.
func logDD(a dd.DD) dd.DD { return dd.AddF(dd.Log(a.Hi), a.Lo/a.Hi) }

// log1pDD returns log(1 + u) for a double-double u > -1.
func log1pDD(u dd.DD) dd.DD {
	s, e := dd.TwoSum(1, u.Hi)
	e += u.Lo
	return logDD(dd.DD{Hi: s, Lo: e})
}

// expDD returns exp(a) rounded to float64 (dd.Exp).
func expDD(a dd.DD) float64 { return dd.Exp(a) }

// stirlingCorr is log Γ(a) - ((a - 1/2) log a - a + log(2π)/2) for a >= 30,
// from the asymptotic series (error below 2^-60 relative there).
func stirlingCorr(a float64) float64 {
	r := 1 / a
	r2 := r * r
	return r * (1.0/12 - r2*(1.0/360-r2*(1.0/1260-r2*(1.0/1680-r2*(1.0/1188)))))
}

// log1pmxNeg returns u - log(1+u) >= 0 for a double-double u > -1, with
// relative accuracy near 2^-70 also when |u| is small (series below 1/64,
// where the direct difference would cancel).
func log1pmxNeg(u dd.DD) dd.DD {
	if math.Abs(u.Hi) < 1.0/64 {
		// u²/2 - u³/3 + u⁴/4 - ...: the u² term in double-double, the rest
		// (below 2^-6 of it) in float64.
		x := u.Hi
		sq := dd.Mul(u, u)
		tail := x * x * x * (-1.0/3 + x*(1.0/4+x*(-1.0/5+x*(1.0/6+x*(-1.0/7+x*(1.0/8+x*(-1.0/9+x*(1.0/10+x*(-1.0/11+x*(1.0/12))))))))))
		return dd.AddF(dd.MulF(sq, 0.5), tail)
	}
	return dd.Sub(u, log1pDD(u))
}

// gammaRatioHalfSeries are the coefficients of S(y) in
// Γ(y+1/2)/Γ(y) = √y · S(y), S(y) = Σ c_k y^-k (from the Stirling series:
// log of the ratio = Σ (-1)^(k+1) (B_{k+1}(1/2) - B_{k+1}(0)) / (k(k+1) y^k),
// exponentiated; mpmath). Twelve terms give 1e-20 relative from y = 25.
var gammaRatioHalfSeries = [...]float64{
	1, -0.125, 0.0078125, 0.0048828125, -0.000640869140625, -0.001522064208984375,
	0.0002071857452392578, 0.0011719763278961182, -0.0001557529903948307,
	-0.0016715728561393917, 0.00021717708659707569, 0.003820047111275926,
	-0.0004892696652092354,
}

// gammaRatioHalf returns Γ(x+1/2)/Γ(x) for x > 0 as a double-double,
// accurate to about 1e-20 relative: the asymptotic series at y = x + n >= 25
// and, below, the exact recurrence Γ(x+1/2)/Γ(x) = Γ(y+1/2)/Γ(y) ·
// Π_{i<n} (x+i)/(x+i+1/2), all in double-double (y included, so that x+i
// is not rounded).
func gammaRatioHalf(x float64) dd.DD {
	y := dd.F(x)
	prod := dd.F(1)
	for y.Hi < 25 {
		prod = dd.Mul(prod, dd.Div(y, dd.AddF(y, 0.5)))
		y = dd.AddF(y, 1)
	}
	r := dd.Div(dd.F(1), y)
	s := dd.F(gammaRatioHalfSeries[len(gammaRatioHalfSeries)-1])
	for k := len(gammaRatioHalfSeries) - 2; k >= 0; k-- {
		s = dd.AddF(dd.Mul(s, r), gammaRatioHalfSeries[k])
	}
	return dd.Mul(dd.Mul(s, dd.Sqrt(y)), prod)
}
