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
