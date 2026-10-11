package stats

import (
	"math"

	"github.com/go-statistics/stats/internal/dd"
)

// Correctly rounded elementary functions for the random-variate paths that
// reproduce SciPy's draws: SciPy evaluates its _ppf formulas with the C
// library's log, exp and pow, which on the reference build round
// correctly (but for rare cases); Go's math package may differ from them
// in the last bit. These are rounded from double-double results accurate
// to about 2^-70.

// crLog is log(x) correctly rounded (x > 0).
func crLog(x float64) float64 { return dd.Log(x).Float() }

// crExp is exp(x) correctly rounded.
func crExp(x float64) float64 { return dd.Exp(dd.F(x)) }

// crPow is x^y correctly rounded for x > 0.
func crPow(x, y float64) float64 { return dd.Exp(dd.MulF(dd.Log(x), y)) }

// crTan is tan(t) correctly rounded for |t| <= π/2.
func crTan(t float64) float64 {
	a := math.Abs(t)
	var r float64
	if a <= quarterPiDD.Hi {
		s, c := sincosSmall(dd.F(a))
		r = dd.Div(s, c).Float()
	} else {
		c, s := sincosSmall(dd.Sub(halfPiDD, dd.F(a)))
		r = dd.Div(s, c).Float()
	}
	return math.Copysign(r, t)
}

// polevl evaluates c[0]·x^n + ... + c[n] by Horner's rule, each product
// rounded before the addition as in C without contraction.
func polevl(x float64, c []float64) float64 {
	a := c[0]
	for _, v := range c[1:] {
		a = float64(a*x) + v
	}
	return a
}

// p1evl is polevl with an implicit leading coefficient 1.
func p1evl(x float64, c []float64) float64 {
	a := x + c[0]
	for _, v := range c[1:] {
		a = float64(a*x) + v
	}
	return a
}

var (
	cephesLP = []float64{4.5270000862445199635215e-5, 4.9854102823193375972212e-1, 6.5787325942061044846969e0,
		2.9911919328553073277375e1, 6.0949667980987787057556e1, 5.7112963590585538103336e1, 2.0039553499201281259648e1}
	cephesLQ = []float64{1.5062909083469192043167e1, 8.3047565967967209469434e1, 2.2176239823732856465394e2,
		3.0909872225312059774938e2, 2.1642788614495947685003e2, 6.0118660497603843919306e1}
	cephesEP = []float64{1.2617719307481059087798e-4, 3.0299440770744196129956e-2, 9.9999999999999999991025e-1}
	cephesEQ = []float64{3.0019850513866445504159e-6, 2.5244834034968410419224e-3, 2.2726554820815502876593e-1,
		2.0000000000000000000897e0}
)

// cephesLog1p is the log1p SciPy's special.log1p evaluates (Cephes, as
// in SciPy's xsf): log(1+x) outside [1/√2, √2], a rational approximation
// within. It reproduces SciPy's values (checked on 40000 points), for the
// random-variate paths that apply SciPy's quantile formulas.
func cephesLog1p(x float64) float64 {
	z := 1 + x
	if z < math.Sqrt2/2 || z > math.Sqrt2 {
		return crLog(z)
	}
	z = x * x
	z = float64(-0.5*z) + float64(x*float64(z*polevl(x, cephesLP)/p1evl(x, cephesLQ)))
	return x + z
}

// cephesExpm1 is the expm1 of SciPy's special.expm1 (Cephes): exp(x) - 1
// outside [-1/2, 1/2], a rational approximation within.
func cephesExpm1(x float64) float64 {
	if x < -0.5 || x > 0.5 || x != x {
		return crExp(x) - 1
	}
	xx := x * x
	r := x * polevl(xx, cephesEP)
	r = r / (polevl(xx, cephesEQ) - r)
	return r + r
}
