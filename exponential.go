package stats

import (
	"fmt"
	"math"

	"github.com/go-ndarray/ndarray"
	"github.com/go-ndarray/ndarray/random"
	"github.com/go-statistics/stats/internal/dd"
)

//stats:continuous Exponential expon

// Exponential is the exponential distribution with rate 1/Scale, shifted
// by Loc: density exp(-x) on [0, ∞) in standard form:
// scipy.stats.expon(loc, scale).
//
// Every function is carried in double-double and rounded once: the density
// and SF are exp(-x) (dd.Exp), CDF is -expm1(-x), LogCDF log1p(-exp(-x))
// or log(-expm1(-x)) whichever does not cancel, PPF -log1p(-p) and ISF
// -log(q).
type Exponential struct {
	Loc, Scale float64
}

func (d Exponential) std() exponStd { return exponStd{} }

type exponStd struct{}

func (exponStd) valid() bool             { return true }
func (exponStd) support() (a, b float64) { return 0, math.Inf(1) }

func (exponStd) pdf(x float64) float64    { return dd.Exp(dd.F(-x)) }
func (exponStd) logpdf(x float64) float64 { return -x }
func (exponStd) cdf(x float64) float64    { return dd.Neg(expm1D(dd.F(-x))).Float() }
func (exponStd) sf(x float64) float64     { return dd.Exp(dd.F(-x)) }
func (exponStd) logcdf(x float64) float64 { return log1mexpD(dd.F(x)).Float() }
func (exponStd) logsf(x float64) float64  { return -x }
func (exponStd) ppf(p float64) float64    { return -log1pDD(dd.F(-p)).Float() }
func (exponStd) isf(q float64) float64    { return -dd.Log(q).Float() }
func (exponStd) mean() float64            { return 1 }
func (exponStd) variance() float64        { return 1 }
func (exponStd) skew() float64            { return 2 }
func (exponStd) kurtosis() float64        { return 6 }
func (exponStd) median() float64          { return math.Ln2 }
func (exponStd) entropy() float64         { return 1 }

// munp is n!.
func (exponStd) munp(n int) float64 { return factorial(n) }

func (exponStd) rvs(g *random.Generator, size []int) (*ndarray.Array, error) {
	return g.StandardExponential(size...)
}

// factorial returns n! as a float64 (+Inf beyond 170).
func factorial(n int) float64 {
	r := 1.0
	for k := 2; k <= n; k++ {
		r *= float64(k)
	}
	return r
}

// FitExponential returns the maximum-likelihood exponential distribution
// for x: Loc the minimum and Scale the mean excess over it, or the other
// one when Loc or Scale is fixed: scipy.stats.expon.fit(x, floc=,
// fscale=). The result is SciPy's to the bit.
func FitExponential[D Data](x D, o *FitOptions) (Exponential, error) {
	v := floats(x)
	fx, err := o.fixed("Exponential", "Loc", "Scale")
	if err != nil {
		return Exponential{}, err
	}
	if err := checkFitData(v); err != nil {
		return Exponential{}, err
	}
	lo, _ := minMax(v)
	loc, haveLoc := fx["Loc"]
	if !haveLoc {
		loc = lo
	} else if lo < loc {
		return Exponential{}, fmt.Errorf("%w: exponential fit with Loc=%v needs every value at or above it", ErrData, loc)
	}
	scale, haveScale := fx["Scale"]
	if !haveScale {
		scale = mean(v) - loc
	}
	d := Exponential{Loc: loc, Scale: scale}
	return d, d.Validate()
}
