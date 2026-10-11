package stats

import "github.com/go-statistics/stats/internal/dd"

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
