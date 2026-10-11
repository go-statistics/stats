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
