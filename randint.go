package stats

import (
	"math"

	"github.com/go-ndarray/ndarray"
	"github.com/go-ndarray/ndarray/random"
	"github.com/go-statistics/stats/internal/dd"
)

//stats:discrete DiscreteUniform randint low=Low:int high=High:int

// DiscreteUniform is the uniform distribution on the integers Low, Low+1,
// ..., High-1 (High excluded, High > Low): scipy.stats.randint(low, high,
// loc).
//
// CDF and SF are the exact counts over High-Low, rounded once; their
// logarithms are differences of double-double logarithms of the counts.
type DiscreteUniform struct {
	Low, High int
	Loc       float64
}

func (d DiscreteUniform) std() randintStd {
	lo, hi := float64(d.Low), float64(d.High)
	return randintStd{lo: lo, hi: hi, d: hi - lo}
}

type randintStd struct{ lo, hi, d float64 }

func (s randintStd) valid() bool               { return s.hi > s.lo }
func (s randintStd) support() (lo, hi float64) { return s.lo, s.hi - 1 }

func (s randintStd) pmf(float64) float64    { return 1 / s.d }
func (s randintStd) logpmf(float64) float64 { return -dd.Log(s.d).Float() }

func (s randintStd) cdf(k float64) float64 { return (k - s.lo + 1) / s.d }
func (s randintStd) sf(k float64) float64  { return (s.hi - 1 - k) / s.d }

// logcdf and logsf are log(c/d) for the count c, as a difference of
// double-double logarithms, or, where c/d > 1/2, as log1p of the
// complementary count over d (the difference would cancel).
func (s randintStd) logcdf(k float64) float64 { return logRatio(k-s.lo+1, s.hi-1-k, s.d) }
func (s randintStd) logsf(k float64) float64  { return logRatio(s.hi-1-k, k-s.lo+1, s.d) }

// logRatio returns log(c/d) for counts c + c' = d.
func logRatio(c, cc, d float64) float64 {
	if c > cc {
		return log1pDD(dd.Neg(dd.Div(dd.F(cc), dd.F(d)))).Float()
	}
	return dd.Sub(dd.Log(c), dd.Log(d)).Float()
}

func (s randintStd) ppf(p float64) float64 {
	return smallestK(s.lo+math.Ceil(p*s.d)-1, s.lo, s.hi-1, func(k float64) bool {
		return k >= s.hi-1 || s.cdf(k) >= p
	})
}

func (s randintStd) isf(q float64) float64 {
	return smallestK(s.hi-1-math.Floor(q*s.d), s.lo, s.hi-1, func(k float64) bool {
		return k >= s.hi-1 || s.sf(k) <= q
	})
}

// The moments are SciPy's closed forms.
func (s randintStd) mean() float64     { return (s.hi + s.lo - 1) / 2 }
func (s randintStd) variance() float64 { return (s.d*s.d - 1) / 12 }
func (s randintStd) skew() float64     { return 0 }
func (s randintStd) kurtosis() float64 { return -6.0 / 5.0 * (s.d*s.d + 1) / (s.d*s.d - 1) }
func (s randintStd) entropy() float64  { return math.Log(s.d) }

// munp is E[X^n] = Σ_i C(n,i) lo^(n-i) E[Y^i] for X = lo + Y, Y uniform on
// 0..d-1, whose factorial moments are d_(i+1) / ((i+1) d) (sums of falling
// factorials).
func (s randintStd) munp(n int) float64 {
	r := 0.0
	c := 1.0 // C(n, i)
	for i := 0; i <= n; i++ {
		ey := 1.0
		if i > 0 {
			ey = rawFromFactorial(i, func(j int) float64 { return fallingPow(s.d, j+1) / (float64(j+1) * s.d) })
		}
		r += c * math.Pow(s.lo, float64(n-i)) * ey
		c = c * float64(n-i) / float64(i+1)
	}
	return r
}

// rvs is numpy's Generator.integers(low, high, size), which SciPy calls.
func (s randintStd) rvs(g *random.Generator, size []int) (*ndarray.Array, error) {
	return g.Integers(int64(s.lo), int64(s.hi), size...)
}
