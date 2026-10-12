package stats

import (
	"fmt"
	"math"

	"github.com/go-ndarray/ndarray"
	"github.com/go-ndarray/ndarray/random"
	"github.com/go-numerics/special"
	"github.com/go-statistics/stats/internal/dd"
)

//stats:continuous HalfNormal halfnorm

// HalfNormal is the half-normal distribution, |Z| for standard normal Z:
// density √(2/π) exp(-x²/2) on [0, ∞) in standard form:
// scipy.stats.halfnorm(loc, scale).
//
// The density carries x²/2 exactly; SF is 2Φ(-x) and CDF erf(x/√2) with
// the rounding of x/√2 corrected to first order, or 1 - SF beyond 1. PPF is
// √2·erfinv(p) below 1/2 (SciPy's Φ⁻¹((1+p)/2) loses every bit of p below
// 1e-16) and -Φ⁻¹((1-p)/2) above, both exact in their argument.
type HalfNormal struct {
	Loc, Scale float64
}

func (d HalfNormal) std() halfnormStd { return halfnormStd{} }

type halfnormStd struct{}

func (halfnormStd) valid() bool             { return true }
func (halfnormStd) support() (a, b float64) { return 0, math.Inf(1) }

func (halfnormStd) pdf(x float64) float64 {
	if x > rayleighBig {
		return 0
	}
	return expScaledR(negHalfSq(x), sqrt2OverPi)
}

func (halfnormStd) logpdf(x float64) float64 {
	if x > rayleighBig {
		return math.Inf(-1)
	}
	return dd.Sub(negHalfSq(x), halfLogHalfPiDD).Float()
}

// halfnormCDF returns erf(x/√2) for 0 < x < 1: erf at the rounded
// quotient plus the first-order term of its rounding error.
func halfnormCDF(x float64) float64 {
	z := dd.MulF(invSqrt2DD, x)
	return special.Erf(z.Hi) + z.Lo*twoOverSqrtPiDD.Hi*math.Exp(-z.Hi*z.Hi)
}

func (halfnormStd) cdf(x float64) float64 {
	if x < halfnormMedian {
		return halfnormCDF(x)
	}
	return dd.AddF(dd.F(1), -2*special.Ndtr(-x)).Float()
}

func (halfnormStd) sf(x float64) float64 { return 2 * special.Ndtr(-x) }

// halfnormMedian is where CDF and SF cross 1/2: each logarithm is taken
// of the smaller of the two, the larger one's through log1p.
const halfnormMedian = 0.6744897501960817

func (halfnormStd) logcdf(x float64) float64 {
	if x < halfnormMedian {
		return dd.Log(halfnormCDF(x)).Float()
	}
	return log1pDD(dd.F(-2 * special.Ndtr(-x))).Float()
}

// logsf is log1p(-CDF) below the median, log(2Φ(-x)) above, and
// log 2 + log Φ(-x) from x = 37, where Φ(-x) nears the underflow.
func (halfnormStd) logsf(x float64) float64 {
	switch {
	case x < halfnormMedian:
		return log1pDD(dd.F(-halfnormCDF(x))).Float()
	case x < 37:
		return dd.Log(2 * special.Ndtr(-x)).Float()
	}
	return dd.AddF(ln2DD, special.LogNdtr(-x)).Float()
}

// halfnormLower is the quantile of p <= 1/2, √2·erfinv(p).
func halfnormLower(p float64) float64 { return dd.MulF(sqrt2DD, special.ErfInv(p)).Float() }

func (halfnormStd) ppf(p float64) float64 {
	if p > 0.5 {
		return -special.Ndtri((1 - p) / 2)
	}
	return halfnormLower(p)
}

func (halfnormStd) isf(q float64) float64 {
	if q < 0.5 {
		return -special.Ndtri(q / 2)
	}
	return halfnormLower(1 - q)
}

// The moments: mean √(2/π), variance 1 - 2/π, skewness
// √2(4-π)/(π-2)^1.5, excess kurtosis 8(π-3)/(π-2)².
func (halfnormStd) mean() float64     { return 0.7978845608028654 }
func (halfnormStd) variance() float64 { return 0.3633802276324187 }
func (halfnormStd) skew() float64     { return 0.995271746431156 }
func (halfnormStd) kurtosis() float64 { return 0.8691773036059741 }

// median is √2 erfinv(1/2).
func (halfnormStd) median() float64 { return halfnormMedian }

// entropy is log(π/2)/2 + 1/2.
func (halfnormStd) entropy() float64 { return 0.7257913526447274 }

// munp is 2^(n/2) Γ((n+1)/2)/√π.
func (halfnormStd) munp(n int) float64 {
	h := float64(n) / 2
	return math.Exp2(h) * special.Gamma(h+0.5) / math.SqrtPi
}

// rvs is SciPy's |Z| for standard normal Z.
func (halfnormStd) rvs(g *random.Generator, size []int) (*ndarray.Array, error) {
	a, err := g.StandardNormal(size...)
	if err != nil {
		return nil, err
	}
	return a.Abs(), nil
}

// FitHalfNormal returns the maximum-likelihood half-normal distribution
// for x: Loc the minimum and Scale the root mean square of x - Loc, or the
// other one when Loc or Scale is fixed: scipy.stats.halfnorm.fit(x, floc=,
// fscale=). The result is SciPy's to the bit.
func FitHalfNormal[D Data](x D, o *FitOptions) (HalfNormal, error) {
	v := floats(x)
	fx, err := o.fixed("HalfNormal", "Loc", "Scale")
	if err != nil {
		return HalfNormal{}, err
	}
	if err := checkFitData(v); err != nil {
		return HalfNormal{}, err
	}
	lo, _ := minMax(v)
	loc, ok := fx["Loc"]
	if !ok {
		loc = lo
	} else if lo < loc {
		return HalfNormal{}, fmt.Errorf("%w: half-normal fit with Loc=%v needs every value at or above it", ErrData, loc)
	}
	scale, ok := fx["Scale"]
	if !ok {
		scale = math.Sqrt(mean(sqDev(v, loc)))
	}
	d := HalfNormal{Loc: loc, Scale: scale}
	return d, d.Validate()
}
