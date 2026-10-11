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
