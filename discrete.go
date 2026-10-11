package stats

import (
	"math"

	"github.com/go-ndarray/ndarray"
	"github.com/go-ndarray/ndarray/random"
)

// Discrete is the method set every discrete distribution of this package
// has: the methods of a frozen scipy.stats.rv_discrete
// (scipy.stats.binom(n, p, loc).pmf(k) is Binomial{N: n, P: p, Loc:
// loc}.PMF(k)).
//
// Arguments and results are float64 so that the methods compose with
// ndarray's Map (k.Map(d.PMF)) as SciPy's do with arrays. PMF of a
// non-integer is 0; CDF and SF floor their argument; PPF and ISF return
// integers. Loc shifts the support. As with Continuous, out-of-domain
// parameters give NaN, and the NewXxx constructors and Validate an error.
type Discrete interface {
	PMF(k float64) float64
	LogPMF(k float64) float64
	CDF(k float64) float64
	LogCDF(k float64) float64
	SF(k float64) float64
	LogSF(k float64) float64
	PPF(p float64) float64
	ISF(q float64) float64
	Mean() float64
	Var() float64
	Std() float64
	Skew() float64
	Kurtosis() float64
	Median() float64
	Entropy() float64
	Moment(n int) float64
	Interval(confidence float64) (lo, hi float64)
	Support() (lo, hi float64)
	Rvs(g *random.Generator, size ...int) (*ndarray.Array, error)
	Validate() error
}

// discStd is a discrete distribution in standard form (loc 0). As for
// contStd, the generated methods apply the boundary rules first:
//
//   - pmf and logpmf: k an integer in the closed support [a, b], finite;
//   - cdf, sf, logcdf, logsf: k an integer in [a, b) (b may be +Inf);
//   - ppf and isf: p in the open interval (0, 1);
//   - every method: only when valid() is true.
type discStd interface {
	valid() bool
	support() (a, b float64)
	pmf(k float64) float64
	logpmf(k float64) float64
	cdf(k float64) float64
	logcdf(k float64) float64
	sf(k float64) float64
	logsf(k float64) float64
	ppf(p float64) float64
	isf(q float64) float64
	mean() float64
	variance() float64
	skew() float64
	kurtosis() float64
	entropy() float64
	munp(n int) float64
	rvs(g *random.Generator, size []int) (*ndarray.Array, error)
}

// shiftDraws adds loc to integer draws as SciPy does (vals * 1 + loc, then
// a cast to int64): the result is an int64 array.
func shiftDraws(a *ndarray.Array, loc float64) *ndarray.Array {
	if a.DType() != ndarray.Int64 {
		a = a.AsType(ndarray.Int64)
	}
	if loc == 0 {
		return a
	}
	v, _ := ndarray.Data[int64](a)
	for i, k := range v {
		v[i] = int64(float64(k) + loc)
	}
	return a
}

// smallestK returns the smallest integer k in [a, b] for which the
// monotone predicate ok (false, then true) holds, searching from the guess
// k0 by doubling steps and then bisection; it returns b when ok holds
// nowhere below b. It is how SciPy's rv_discrete ppf and isf are defined:
// ppf(p) = smallestK(cdf(k) >= p), isf(q) = smallestK(sf(k) <= q).
func smallestK(k0, a, b float64, ok func(float64) bool) float64 {
	k := math.Max(a, math.Min(b, math.Floor(k0)))
	if k != k {
		k = a
	}
	if ok(k) {
		hi := k // ok(hi)
		step := 1.0
		for hi > a {
			lo := math.Max(a, hi-step)
			if !ok(lo) {
				return bisectK(lo, hi, ok)
			}
			hi = lo
			step *= 2
		}
		return a
	}
	lo := k // !ok(lo)
	step := 1.0
	for {
		hi := math.Min(b, lo+step)
		if hi == b || ok(hi) {
			return bisectK(lo, hi, ok)
		}
		lo = hi
		step *= 2
	}
}

// bisectK returns the smallest k in (lo, hi] with ok(k), given !ok(lo);
// hi when there is none below it.
func bisectK(lo, hi float64, ok func(float64) bool) float64 {
	for hi-lo > 1 {
		mid := math.Floor(lo + (hi-lo)/2)
		if mid == lo || mid == hi { // beyond 2^53 the integers are not all representable
			break
		}
		if ok(mid) {
			hi = mid
		} else {
			lo = mid
		}
	}
	return hi
}
