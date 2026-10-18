package stats

import (
	"math"

	"github.com/go-ndarray/ndarray"
	"github.com/go-ndarray/ndarray/random"
	"github.com/go-numerics/special"
	"github.com/go-statistics/stats/internal/dd"
)

//stats:discrete NegativeBinomial nbinom n=N p=P

// NegativeBinomial is the number of failures before the N-th success in
// trials of success probability P: mass Γ(N+k)/(Γ(N) k!) P^N (1-P)^k on
// k = 0, 1, ..., for real N > 0 and P in (0, 1]: scipy.stats.nbinom(n, p,
// loc).
//
// PMF is R's dnbinom: N/(N+k) times the binomial mass of N successes in
// N+k trials, in Loader's form (see binomTerm) with N+k carried in
// double-double. CDF and SF are special.BetaInc(N, k+1, P) and its
// complement.
type NegativeBinomial struct {
	N, P, Loc float64
}

func (d NegativeBinomial) std() nbinomStd {
	return nbinomStd{n: d.N, p: d.P, q: oneMinus(d.P)}
}

type nbinomStd struct {
	n, p float64
	q    dd.DD
}

func (s nbinomStd) valid() bool {
	return s.n > 0 && !math.IsInf(s.n, 1) && s.p > 0 && s.p <= 1
}
func (s nbinomStd) support() (lo, hi float64) { return 0, math.Inf(1) }

func (s nbinomStd) term(k float64) (e, c dd.DD) {
	if s.p == 1 {
		if k == 0 {
			return dd.F(0), dd.F(1)
		}
		return dd.F(math.Inf(-1)), dd.F(1)
	}
	th, tl := dd.TwoSum(s.n, k)
	tot := dd.DD{Hi: th, Lo: tl}
	e, c = binomTerm(s.n, k, tot, dd.F(s.p), s.q)
	return e, dd.Mul(c, dd.Div(dd.F(s.n), tot))
}

func (s nbinomStd) pmf(k float64) float64    { return massOf(s.term(k)) }
func (s nbinomStd) logpmf(k float64) float64 { return logMassOf(s.term(k)) }

func (s nbinomStd) cdf(k float64) float64 { return special.BetaInc(s.n, k+1, s.p) }
func (s nbinomStd) sf(k float64) float64  { return special.BetaIncC(s.n, k+1, s.p) }

// nj returns n + j exactly.
func (s nbinomStd) nj(j float64) dd.DD {
	h, l := dd.TwoSum(s.n, j)
	return dd.DD{Hi: h, Lo: l}
}

func (s nbinomStd) logcdf(k float64) float64 {
	return logCDFFrom(s.cdf(k), s.sf(k), func() (float64, dd.DD, bool) {
		// pmf(j-1)/pmf(j) = j / ((n+j-1) q)
		sum, ok := ratioSum(k, -1, 0, tailBudget, func(j float64) dd.DD {
			return dd.Div(dd.F(j), dd.Mul(s.nj(j-1), s.q))
		})
		return s.logpmf(k), sum, ok
	})
}

func (s nbinomStd) logsf(k float64) float64 {
	return logCDFFrom(s.sf(k), s.cdf(k), func() (float64, dd.DD, bool) {
		// pmf(j+1)/pmf(j) = (n+j) q / (j+1)
		sum, ok := ratioSum(k+1, 1, math.Inf(1), tailBudget, func(j float64) dd.DD {
			return dd.DivF(dd.Mul(s.nj(j), s.q), j+1)
		})
		return s.logpmf(k + 1), sum, ok
	})
}

func (s nbinomStd) guess(z float64) float64 {
	return normalGuess(s.mean(), math.Sqrt(s.variance()), s.skew(), z)
}

func (s nbinomStd) ppf(p float64) float64 {
	return smallestK(s.guess(special.Ndtri(p)), 0, math.Inf(1), func(k float64) bool { return s.cdf(k) >= p })
}

func (s nbinomStd) isf(q float64) float64 {
	return smallestK(s.guess(-special.Ndtri(q)), 0, math.Inf(1), func(k float64) bool { return s.sf(k) <= q })
}

// nq is n(1-p) with 1-p exact.
func (s nbinomStd) nq() float64 { return dd.MulF(s.q, s.n).Float() }

// The moments are the closed forms SciPy takes from Boost.
func (s nbinomStd) mean() float64     { return s.nq() / s.p }
func (s nbinomStd) variance() float64 { return s.nq() / (s.p * s.p) }
func (s nbinomStd) skew() float64     { return (2 - s.p) / math.Sqrt(s.nq()) }
func (s nbinomStd) kurtosis() float64 { return (6 - s.p*(6-s.p)) / s.nq() }

// entropy sums -p log p from the mode outwards (SciPy sums it too, at
// most 1000 terms), or uses the Edgeworth expansion where the distribution
// is close to normal (variance above 1e6, skewness below 1e-3), or, where
// it is skewed and too wide to sum (standard deviation above
// entropyIntegralMinSD: small P), entropyIntegral.
func (s nbinomStd) entropy() float64 {
	if s.p == 1 {
		return 0
	}
	if g1 := s.skew(); s.variance() > entropyEdgeworthMin && g1 < 1e-3 {
		return entropyLargeVar(s.variance(), g1, s.kurtosis())
	}
	mode := 0.0
	if s.n > 1 {
		mode = math.Floor((s.n - 1) * s.q.Float() / s.p)
	}
	if s.p < 0x1p-53 {
		// P·X/(1-P) is Gamma(N, 1) to O(P): the entropy is that of the
		// gamma density, N + log Γ(N) + (1-N)ψ(N), plus log((1-P)/P),
		// the lattice's own correction being of order 1/variance
		lg, _ := math.Lgamma(s.n)
		return s.n + lg + (1-s.n)*special.Digamma(s.n) + math.Log(s.q.Float()) - math.Log(s.p)
	}
	if sd := math.Sqrt(s.variance()); sd > entropyIntegralMinSD {
		return entropyIntegral(0, mode, sd, s.term)
	}
	return entropySum(mode, 0, math.Inf(1), s.term)
}

// munp is E[X^n] from the factorial moments n(n+1)...(n+j-1) (q/p)^j.
func (s nbinomStd) munp(n int) float64 {
	r := s.q.Float() / s.p
	return rawFromFactorial(n, func(j int) float64 { return poch(s.n, j) * math.Pow(r, float64(j)) })
}

func (s nbinomStd) rvs(g *random.Generator, size []int) (*ndarray.Array, error) {
	return g.NegativeBinomial(s.n, s.p, size...)
}
