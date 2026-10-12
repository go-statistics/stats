package stats

import (
	"math"

	"github.com/go-ndarray/ndarray"
	"github.com/go-ndarray/ndarray/random"
	"github.com/go-statistics/stats/internal/dd"
)

//stats:continuous Laplace laplace

// Laplace is the Laplace (double exponential) distribution with median Loc
// and scale Scale: density exp(-|x|)/2 in standard form:
// scipy.stats.laplace(loc, scale).
//
// The halving is folded into the double-double exponent (exp(-|x| - ln 2)),
// so that the density and the tails round once, also where they are
// subnormal; LogCDF and LogSF use log1p(-exp(-|x|)/2) on the far side,
// where SciPy's log(1 - exp(-x)/2) loses the low bits.
type Laplace struct {
	Loc, Scale float64
}

func (d Laplace) std() laplaceStd { return laplaceStd{} }

type laplaceStd struct{}

func (laplaceStd) valid() bool             { return true }
func (laplaceStd) support() (a, b float64) { return math.Inf(-1), math.Inf(1) }

// halfExp returns exp(-t)/2 = exp(-t - ln 2) as a double-double, t >= 0.
func halfExp(t float64) dd.DD { return expD(dd.Sub(dd.F(-t), ln2DD)) }

func (laplaceStd) pdf(x float64) float64 { return expR(dd.Sub(dd.F(-math.Abs(x)), ln2DD)) }
func (laplaceStd) logpdf(x float64) float64 {
	return dd.Sub(dd.F(-math.Abs(x)), ln2DD).Float()
}

// laplaceCDF returns P(X <= x) as a double-double.
func laplaceCDF(x float64) dd.DD {
	if x < 0 {
		return halfExp(-x)
	}
	return dd.Sub(dd.F(1), halfExp(x))
}

// laplaceCDFf is laplaceCDF rounded, the lower tail rounded once from the
// exponential's mantissa (a double-double rounded part by part would round
// twice where it is subnormal).
func laplaceCDFf(x float64) float64 {
	if x < 0 {
		return expR(dd.Sub(dd.F(x), ln2DD))
	}
	return laplaceCDF(x).Float()
}

func (laplaceStd) cdf(x float64) float64 { return laplaceCDFf(x) }
func (laplaceStd) sf(x float64) float64  { return laplaceCDFf(-x) }

// laplaceLogCDF returns log P(X <= x).
func laplaceLogCDF(x float64) float64 {
	if x < 0 {
		return dd.Sub(dd.F(x), ln2DD).Float()
	}
	return log1pDD(dd.Neg(halfExp(x))).Float()
}

func (laplaceStd) logcdf(x float64) float64 { return laplaceLogCDF(x) }
func (laplaceStd) logsf(x float64) float64  { return laplaceLogCDF(-x) }

// laplaceLower is the quantile of p <= 1/2: log(2p), 2p exact.
func laplaceLower(p float64) float64 { return dd.Log(2 * p).Float() }

func (laplaceStd) ppf(p float64) float64 {
	if p > 0.5 {
		return -laplaceLower(1 - p)
	}
	return laplaceLower(p)
}

func (laplaceStd) isf(q float64) float64 {
	if q > 0.5 {
		return laplaceLower(1 - q)
	}
	return -laplaceLower(q)
}

func (laplaceStd) mean() float64     { return 0 }
func (laplaceStd) variance() float64 { return 2 }
func (laplaceStd) skew() float64     { return 0 }
func (laplaceStd) kurtosis() float64 { return 3 }
func (laplaceStd) median() float64   { return 0 }

// entropy is 1 + log 2.
func (laplaceStd) entropy() float64 { return 1.6931471805599454 }

// munp is n! for even n, 0 for odd n.
func (laplaceStd) munp(n int) float64 {
	if n%2 == 1 {
		return 0
	}
	return factorial(n)
}

func (laplaceStd) rvs(g *random.Generator, size []int) (*ndarray.Array, error) {
	return g.Laplace(0, 1, size...)
}

// FitLaplace returns the maximum-likelihood Laplace distribution for x:
// Loc the median and Scale the mean absolute deviation from it, or the
// other one when Loc or Scale is fixed: scipy.stats.laplace.fit(x, floc=,
// fscale=). The result is SciPy's to the bit.
func FitLaplace[D Data](x D, o *FitOptions) (Laplace, error) {
	v := floats(x)
	fx, err := o.fixed("Laplace", "Loc", "Scale")
	if err != nil {
		return Laplace{}, err
	}
	if err := checkFitData(v); err != nil {
		return Laplace{}, err
	}
	loc, ok := fx["Loc"]
	if !ok {
		loc = npMedian(v)
	}
	scale, ok := fx["Scale"]
	if !ok {
		a := make([]float64, len(v))
		for i, t := range v {
			a[i] = math.Abs(t - loc)
		}
		scale = sum(a) / float64(len(v))
	}
	d := Laplace{Loc: loc, Scale: scale}
	return d, d.Validate()
}

// npMedian is np.median: the middle value, or the mean of the two middle
// values.
func npMedian(v []float64) float64 {
	s := sortedCopy(v)
	n := len(s)
	if n%2 == 1 {
		return s[n/2]
	}
	return (s[n/2-1] + s[n/2]) / 2
}
