package stats

import (
	"math"

	"github.com/go-ndarray/ndarray"
	"github.com/go-ndarray/ndarray/random"
	"github.com/go-statistics/stats/internal/dd"
)

//stats:discrete Geometric geom p=P

// Geometric is the number of trials up to and including the first success,
// each of probability P in (0, 1]: mass (1-P)^(k-1) P on k = 1, 2, ...:
// scipy.stats.geom(p, loc).
//
// Every function is a closed form in k·log(1-P), carried in double-double
// (log1p(-P) to 2^-70 relative, the product exact) before one exponential:
// SF is exp(k log(1-P)), CDF -expm1 of it, LogCDF log(-expm1) or
// log1p(-exp), so that they stay accurate for P near 0 and k near 2^53.
type Geometric struct {
	P, Loc float64
}

func (d Geometric) std() geomStd {
	return geomStd{p: d.P, q: oneMinus(d.P), lq: log1pDD(dd.F(-d.P))}
}

type geomStd struct {
	p     float64
	q, lq dd.DD // 1-p and log(1-p)
}

func (s geomStd) valid() bool               { return s.p > 0 && s.p <= 1 }
func (s geomStd) support() (lo, hi float64) { return 1, math.Inf(1) }

// klq is k·log(1-p): 0 for k = 0 and -Inf for k > 0 when p = 1.
func (s geomStd) klq(k float64) dd.DD {
	switch {
	case k == 0:
		return dd.F(0)
	case s.p == 1:
		return dd.F(math.Inf(-1))
	}
	return dd.MulF(s.lq, k)
}

func (s geomStd) pmf(k float64) float64    { return expScaled(s.klq(k-1), dd.F(s.p)) }
func (s geomStd) logpmf(k float64) float64 { return dd.Add(s.klq(k-1), dd.Log(s.p)).Float() }

func (s geomStd) cdf(k float64) float64 { return dd.Neg(expm1DD(s.klq(k))).Float() }
func (s geomStd) sf(k float64) float64  { return dd.Exp(s.klq(k)) }

func (s geomStd) logcdf(k float64) float64 { return log1mexpDD(s.klq(k)) }
func (s geomStd) logsf(k float64) float64  { return s.klq(k).Float() }

func (s geomStd) ppf(p float64) float64 {
	g := math.Ceil(math.Log1p(-p) / s.lq.Hi)
	return smallestK(g, 1, math.Inf(1), func(k float64) bool { return s.cdf(k) >= p })
}

func (s geomStd) isf(q float64) float64 {
	g := math.Ceil(math.Log(q) / s.lq.Hi)
	return smallestK(g, 1, math.Inf(1), func(k float64) bool { return s.sf(k) <= q })
}

func (s geomStd) mean() float64     { return 1 / s.p }
func (s geomStd) variance() float64 { return s.q.Float() / s.p / s.p }
func (s geomStd) skew() float64     { return (2 - s.p) / math.Sqrt(s.q.Float()) }
func (s geomStd) kurtosis() float64 { return ((s.p-6)*s.p + 6) / s.q.Float() }

// entropy is (-p log p - q log q)/p, SciPy's -log p - log1p(-p)(1-p)/p, in
// double-double; 0 at p = 1 (where SciPy's form gives NaN).
func (s geomStd) entropy() float64 {
	if s.p == 1 {
		return 0
	}
	h := dd.Neg(dd.Add(dd.Log(s.p), dd.Div(dd.Mul(s.lq, s.q), dd.F(s.p))))
	return h.Float()
}

// munp is E[X^n] from the factorial moments j! q^(j-1) / p^j.
func (s geomStd) munp(n int) float64 {
	q := s.q.Float()
	return rawFromFactorial(n, func(j int) float64 {
		return math.Gamma(float64(j+1)) * math.Pow(q, float64(j-1)) / math.Pow(s.p, float64(j))
	})
}

func (s geomStd) rvs(g *random.Generator, size []int) (*ndarray.Array, error) {
	return g.Geometric(s.p, size...)
}
