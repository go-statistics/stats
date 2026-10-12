package stats

import (
	"math"

	"github.com/go-ndarray/ndarray"
	"github.com/go-ndarray/ndarray/random"
	"github.com/go-numerics/special"
	"github.com/go-statistics/stats/internal/dd"
)

//stats:continuous Normal norm

// Normal is the normal (Gaussian) distribution with mean Loc and standard
// deviation Scale: scipy.stats.norm(loc, scale).
//
// The density is exp(-z²/2)/√(2π) with z² carried exactly and the
// exponential in double-double, so it is correctly rounded but for rare
// cases, out to the underflow at |z| ≈ 38.6 (the textbook formula, which
// SciPy uses, loses log2(z²) bits there). CDF, SF and their logarithms are
// special.Ndtr and special.LogNdtr at z and -z; PPF and ISF are
// special.Ndtri at p and, negated, at q.
type Normal struct {
	Loc, Scale float64
}

func (d Normal) std() normalStd { return normalStd{} }

type normalStd struct{}

func (normalStd) valid() bool             { return true }
func (normalStd) support() (a, b float64) { return math.Inf(-1), math.Inf(1) }

func (normalStd) pdf(z float64) float64 { return expScaled(negHalfSq(z), invSqrt2Pi) }

func (normalStd) logpdf(z float64) float64 {
	h := negHalfSq(z)
	if math.IsInf(h.Hi, 0) { // |z| > 1.3e154: z² overflows
		return h.Hi
	}
	return dd.Sub(h, logSqrt2Pi).Float()
}

func (normalStd) cdf(z float64) float64    { return special.Ndtr(z) }
func (normalStd) sf(z float64) float64     { return special.Ndtr(-z) }
func (normalStd) logcdf(z float64) float64 { return special.LogNdtr(z) }
func (normalStd) logsf(z float64) float64  { return special.LogNdtr(-z) }
func (normalStd) ppf(p float64) float64    { return special.Ndtri(p) }
func (normalStd) isf(q float64) float64    { return -special.Ndtri(q) }
func (normalStd) mean() float64            { return 0 }
func (normalStd) variance() float64        { return 1 }
func (normalStd) skew() float64            { return 0 }
func (normalStd) kurtosis() float64        { return 0 }
func (normalStd) median() float64          { return 0 }

// entropy is (1 + log 2π)/2.
func (normalStd) entropy() float64 { return 1.4189385332046727 }

// munp: E[Z^n] is 0 for odd n and (n-1)!! for even n.
func (normalStd) munp(n int) float64 {
	if n%2 == 1 {
		return 0
	}
	return special.Factorial2(n - 1)
}

func (normalStd) rvs(g *random.Generator, size []int) (*ndarray.Array, error) {
	return g.StandardNormal(size...)
}

// FitNormal returns the maximum-likelihood normal distribution for x: the
// mean and the root mean squared deviation (ddof 0), or the other one when
// Loc or Scale is fixed in o.Fixed: scipy.stats.norm.fit(x, floc=, fscale=).
// The result is SciPy's to the bit (the same pairwise sums).
func FitNormal[D Data](x D, o *FitOptions) (Normal, error) {
	v := floats(x)
	fx, err := o.fixed("Normal", "Loc", "Scale")
	if err != nil {
		return Normal{}, err
	}
	if err := checkFitData(v); err != nil {
		return Normal{}, err
	}
	loc, haveLoc := fx["Loc"]
	scale, haveScale := fx["Scale"]
	if !haveLoc {
		loc = mean(v)
	}
	if !haveScale {
		scale = math.Sqrt(mean(sqDev(v, loc)))
	}
	d := Normal{Loc: loc, Scale: scale}
	return d, d.Validate()
}
