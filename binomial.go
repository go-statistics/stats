package stats

import (
	"math"

	"github.com/go-ndarray/ndarray"
	"github.com/go-ndarray/ndarray/random"
	"github.com/go-numerics/special"
	"github.com/go-statistics/stats/internal/dd"
)

//stats:discrete Binomial binom n=N:int p=P

// Binomial is the number of successes in N independent trials of
// probability P: mass C(N,k) P^k (1-P)^(N-k) on k = 0..N:
// scipy.stats.binom(n, p, loc).
//
// PMF and LogPMF use Loader's saddle-point form (see binomTerm), accurate to
// about an ulp for every N up to 2^53 and in the far tails, where SciPy's
// logpmf (log-gamma differences) loses digits to cancellation. CDF and SF
// are special.BetaIncC and special.BetaInc; LogCDF and LogSF switch to
// log1p of the complement near 1 and, below the normal range, to the tail
// summed from its first mass. PPF and ISF follow SciPy's definition (the
// smallest k with CDF(k) >= p, resp. SF(k) <= q) from a Cornish-Fisher
// starting point.
type Binomial struct {
	N      int
	P, Loc float64
}

func (d Binomial) std() binomStd {
	return binomStd{n: float64(d.N), p: d.P, q: oneMinus(d.P)}
}

type binomStd struct {
	n, p float64
	q    dd.DD // 1 - p, exact
}

func (s binomStd) valid() bool               { return s.n >= 0 && s.p >= 0 && s.p <= 1 }
func (s binomStd) support() (lo, hi float64) { return 0, s.n }

// term returns the mass at k as c·exp(e); degenerate (p = 0 or 1) masses
// are exp(0) or exp(-Inf).
func (s binomStd) term(k float64) (e, c dd.DD) {
	switch {
	case s.p == 0 && k > 0, s.p == 1 && k < s.n:
		return dd.F(math.Inf(-1)), dd.F(1)
	case s.p == 0 || s.p == 1:
		return dd.F(0), dd.F(1)
	}
	return binomTerm(k, s.n-k, dd.F(s.n), dd.F(s.p), s.q)
}

func (s binomStd) pmf(k float64) float64    { return massOf(s.term(k)) }
func (s binomStd) logpmf(k float64) float64 { return logMassOf(s.term(k)) }

func (s binomStd) cdf(k float64) float64 { return special.BetaIncC(k+1, s.n-k, s.p) }
func (s binomStd) sf(k float64) float64  { return special.BetaInc(k+1, s.n-k, s.p) }

func (s binomStd) logcdf(k float64) float64 {
	return logCDFFrom(s.cdf(k), s.sf(k), func() (float64, dd.DD, bool) {
		// pmf(j-1)/pmf(j) = j q / ((n-j+1) p)
		sum, ok := ratioSum(k, -1, 0, tailBudget, func(j float64) dd.DD {
			return dd.Div(dd.MulF(s.q, j), dd.MulFF(s.n-j+1, s.p))
		})
		return s.logpmf(k), sum, ok
	})
}

func (s binomStd) logsf(k float64) float64 {
	return logCDFFrom(s.sf(k), s.cdf(k), func() (float64, dd.DD, bool) {
		// pmf(j+1)/pmf(j) = (n-j) p / ((j+1) q)
		sum, ok := ratioSum(k+1, 1, s.n, tailBudget, func(j float64) dd.DD {
			return dd.Div(dd.MulFF(s.n-j, s.p), dd.MulF(s.q, j+1))
		})
		return s.logpmf(k + 1), sum, ok
	})
}

func (s binomStd) guess(z float64) float64 {
	return normalGuess(s.mean(), math.Sqrt(s.variance()), s.skew(), z)
}

func (s binomStd) ppf(p float64) float64 {
	return smallestK(s.guess(special.Ndtri(p)), 0, s.n, func(k float64) bool {
		return k >= s.n || s.cdf(k) >= p
	})
}

func (s binomStd) isf(q float64) float64 {
	return smallestK(s.guess(-special.Ndtri(q)), 0, s.n, func(k float64) bool {
		return k >= s.n || s.sf(k) <= q
	})
}

// pq is p(1-p) with 1-p exact.
func (s binomStd) pq() float64 { return dd.MulF(s.q, s.p).Float() }

func (s binomStd) mean() float64     { return s.n * s.p }
func (s binomStd) variance() float64 { return s.n * s.pq() }

// skew is (1-2p)/sqrt(npq); NaN where npq = 0, as SciPy's
// 1/sqrt(npq) - 2p/sqrt(npq) is.
func (s binomStd) skew() float64 {
	v := s.variance()
	if v == 0 {
		return math.NaN()
	}
	return (1 - 2*s.p) / math.Sqrt(v)
}

// kurtosis is (1-6pq)/(npq): +Inf where pq = 0 < n, NaN for n = 0 (SciPy's
// 1/(npq) - 6/n).
func (s binomStd) kurtosis() float64 {
	if s.n == 0 {
		return math.NaN()
	}
	pq := s.pq()
	return (1 - 6*pq) / (s.n * pq)
}

// entropy sums -p log p over the support (SciPy sums it too, truncated at
// 1000 terms), or, for variances above 1e6, the Edgeworth expansion.
func (s binomStd) entropy() float64 {
	if v := s.variance(); v > entropyEdgeworthMin {
		return entropyLargeVar(v, s.skew(), s.kurtosis())
	}
	if s.p == 0 || s.p == 1 {
		return 0
	}
	mode := math.Floor((s.n + 1) * s.p)
	return entropySum(math.Min(mode, s.n), 0, s.n, s.term)
}

// munp is E[X^n] from the factorial moments n(n-1)...(n-j+1) p^j.
func (s binomStd) munp(n int) float64 {
	return rawFromFactorial(n, func(j int) float64 { return fallingPow(s.n, j) * math.Pow(s.p, float64(j)) })
}

func (s binomStd) rvs(g *random.Generator, size []int) (*ndarray.Array, error) {
	return g.Binomial(int64(s.n), s.p, size...)
}

//stats:discrete Bernoulli bernoulli p=P

// Bernoulli is a single trial of success probability P: mass 1-P at 0 and
// P at 1: scipy.stats.bernoulli(p, loc). The zero value (P = 0) is the
// distribution concentrated at 0.
type Bernoulli struct {
	P, Loc float64
}

func (d Bernoulli) std() bernoulliStd { return bernoulliStd{p: d.P, q: oneMinus(d.P)} }

type bernoulliStd struct {
	p float64
	q dd.DD
}

func (s bernoulliStd) valid() bool               { return s.p >= 0 && s.p <= 1 }
func (s bernoulliStd) support() (lo, hi float64) { return 0, 1 }

func (s bernoulliStd) pmf(k float64) float64 {
	if k == 0 {
		return s.q.Float()
	}
	return s.p
}

func (s bernoulliStd) logpmf(k float64) float64 {
	if k == 0 {
		return log1pDD(dd.F(-s.p)).Float()
	}
	return dd.Log(s.p).Float()
}

// The distribution functions are only asked at k = 0.
func (s bernoulliStd) cdf(float64) float64    { return s.q.Float() }
func (s bernoulliStd) sf(float64) float64     { return s.p }
func (s bernoulliStd) logcdf(float64) float64 { return s.logpmf(0) }
func (s bernoulliStd) logsf(float64) float64  { return s.logpmf(1) }

func (s bernoulliStd) ppf(p float64) float64 {
	if s.q.Float() >= p {
		return 0
	}
	return 1
}

func (s bernoulliStd) isf(q float64) float64 {
	if s.p <= q {
		return 0
	}
	return 1
}

func (s bernoulliStd) binom() binomStd   { return binomStd{n: 1, p: s.p, q: s.q} }
func (s bernoulliStd) mean() float64     { return s.p }
func (s bernoulliStd) variance() float64 { return s.binom().variance() }
func (s bernoulliStd) skew() float64     { return s.binom().skew() }
func (s bernoulliStd) kurtosis() float64 { return s.binom().kurtosis() }
func (s bernoulliStd) munp(int) float64  { return s.p }
func (s bernoulliStd) entropy() float64  { return entr(s.p) + entr(s.q.Float()) }
func (s bernoulliStd) rvs(g *random.Generator, size []int) (*ndarray.Array, error) {
	return g.Binomial(1, s.p, size...)
}

// entr is -x log x, 0 at x = 0 (scipy.special.entr).
func entr(x float64) float64 {
	if x == 0 {
		return 0
	}
	return -x * math.Log(x)
}
