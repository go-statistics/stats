package stats

import (
	"math"

	"github.com/go-ndarray/ndarray"
	"github.com/go-ndarray/ndarray/random"
	"github.com/go-numerics/special"
	"github.com/go-statistics/stats/internal/dd"
)

//stats:discrete Hypergeometric hypergeom M=Total:int64 n=Good:int64 N=Draws:int64

// Hypergeometric is the number of good items among Draws drawn without
// replacement from Total items of which Good are good: mass
// C(Good,k) C(Total-Good, Draws-k) / C(Total, Draws):
// scipy.stats.hypergeom(M, n, N, loc) with M = Total, n = Good, N = Draws.
//
// PMF is R's dhyper: the product of two binomial masses over a third, all
// at p = Draws/Total, in Loader's form (see binomTerm) with the three
// exponents combined in double-double before one exponential. CDF and SF
// sum the masses of a tail from its first one by the ratio of successive
// masses (in double-double), starting with the tail on k's side of the
// mean and switching to the other when that one turns out to be the
// smaller: the cost is proportional to the standard deviation, and the
// logarithms (log of the first mass plus log of the scaled sum) stay
// accurate where the probabilities underflow.
type Hypergeometric struct {
	Total, Good, Draws int64
	Loc                float64
}

func (d Hypergeometric) std() hypergeomStd {
	m, n, nd := float64(d.Total), float64(d.Good), float64(d.Draws)
	return hypergeomStd{
		m: m, n: n, nd: nd, bad: m - n,
		p: dd.Div(dd.F(nd), dd.F(m)),
		q: dd.Div(dd.F(m-nd), dd.F(m)),
	}
}

type hypergeomStd struct {
	m, n, nd, bad float64 // total, good, draws, total - good
	p, q          dd.DD   // draws/total and its complement
}

func (s hypergeomStd) valid() bool {
	return s.m > 0 && s.n >= 0 && s.nd >= 0 && s.n <= s.m && s.nd <= s.m
}

func (s hypergeomStd) support() (lo, hi float64) {
	return math.Max(0, s.nd-s.bad), math.Min(s.n, s.nd)
}

// term returns the mass at k (in the support) as c·exp(e).
func (s hypergeomStd) term(k float64) (e, c dd.DD) {
	lo, hi := s.support()
	if lo == hi {
		return dd.F(0), dd.F(1)
	}
	e1, c1 := binomTerm(k, s.n-k, dd.F(s.n), s.p, s.q)
	e2, c2 := binomTerm(s.nd-k, s.bad-s.nd+k, dd.F(s.bad), s.p, s.q)
	e3, c3 := binomTerm(s.nd, s.m-s.nd, dd.F(s.m), s.p, s.q)
	return dd.Sub(dd.Add(e1, e2), e3), dd.Div(dd.Mul(c1, c2), c3)
}

func (s hypergeomStd) pmf(k float64) float64    { return massOf(s.term(k)) }
func (s hypergeomStd) logpmf(k float64) float64 { return logMassOf(s.term(k)) }

// down returns the lower tail P(X <= k) as the first mass's exponent and
// factor and the scaled sum S: P = c·exp(e)·S.
func (s hypergeomStd) down(k float64) (e, c, sum dd.DD) {
	lo, _ := s.support()
	e, c = s.term(k)
	// pmf(j-1)/pmf(j) = j (bad-N+j) / ((n-j+1)(N-j+1))
	sum, _ = ratioSum(k, -1, lo, 0, func(j float64) dd.DD {
		return dd.Div(dd.MulFF(j, s.bad-s.nd+j), dd.MulFF(s.n-j+1, s.nd-j+1))
	})
	return e, c, sum
}

// up returns the upper tail P(X > k) in the same form.
func (s hypergeomStd) up(k float64) (e, c, sum dd.DD) {
	_, hi := s.support()
	e, c = s.term(k + 1)
	// pmf(j+1)/pmf(j) = (n-j)(N-j) / ((j+1)(bad-N+j+1))
	sum, _ = ratioSum(k+1, 1, hi, 0, func(j float64) dd.DD {
		return dd.Div(dd.MulFF(s.n-j, s.nd-j), dd.MulFF(j+1, s.bad-s.nd+j+1))
	})
	return e, c, sum
}

// tailValue is c·exp(e)·sum as a double-double.
func tailValue(e, c, sum dd.DD) dd.DD {
	m, k := dd.ExpMant(e)
	return dd.Ldexp(dd.Mul(dd.Mul(m, c), sum), k)
}

// tails returns (P(X <= k), P(X > k)), the smaller one summed directly and
// the other one as its complement, and which one was summed (lower true:
// the lower tail, given by e, c, sum).
func (s hypergeomStd) tails(k float64) (cdf, sf dd.DD, lower bool, e, c, sum dd.DD) {
	lower = k < s.nd*s.n/s.m
	for range 2 {
		if lower {
			e, c, sum = s.down(k)
			cdf = tailValue(e, c, sum)
			sf = dd.AddF(dd.Neg(cdf), 1)
			if cdf.Hi <= 0.5 {
				break
			}
		} else {
			e, c, sum = s.up(k)
			sf = tailValue(e, c, sum)
			cdf = dd.AddF(dd.Neg(sf), 1)
			if sf.Hi <= 0.5 {
				break
			}
		}
		lower = !lower
	}
	return cdf, sf, lower, e, c, sum
}

func (s hypergeomStd) cdf(k float64) float64 {
	c, _, _, _, _, _ := s.tails(k)
	return c.Float()
}

func (s hypergeomStd) sf(k float64) float64 {
	_, f, _, _, _, _ := s.tails(k)
	return f.Float()
}

// logOf returns the logarithm of the tail that was summed, c·exp(e)·sum.
func logOf(e, c, sum dd.DD) float64 { return dd.Add(e, logDD(dd.Mul(c, sum))).Float() }

func (s hypergeomStd) logcdf(k float64) float64 {
	_, sf, lower, e, c, sum := s.tails(k)
	if lower {
		return logOf(e, c, sum)
	}
	return log1pDD(dd.Neg(sf)).Float()
}

func (s hypergeomStd) logsf(k float64) float64 {
	cdf, _, lower, e, c, sum := s.tails(k)
	if !lower {
		return logOf(e, c, sum)
	}
	return log1pDD(dd.Neg(cdf)).Float()
}

func (s hypergeomStd) guess(z float64) float64 {
	return normalGuess(s.mean(), math.Sqrt(s.variance()), s.skew(), z)
}

func (s hypergeomStd) ppf(p float64) float64 {
	lo, hi := s.support()
	return smallestK(s.guess(special.Ndtri(p)), lo, hi, func(k float64) bool {
		if k >= hi {
			return true
		}
		return s.cdf(k) >= p
	})
}

func (s hypergeomStd) isf(q float64) float64 {
	lo, hi := s.support()
	return smallestK(s.guess(-special.Ndtri(q)), lo, hi, func(k float64) bool {
		if k >= hi {
			return true
		}
		return s.sf(k) <= q
	})
}

// The moments are the closed forms SciPy uses (Boost's mean, variance and
// skewness, its own excess kurtosis), in M = total, n = good, N = draws.
func (s hypergeomStd) mean() float64 { return s.n * s.nd / s.m }

func (s hypergeomStd) variance() float64 {
	m, n, nd := s.m, s.n, s.nd
	return n * nd * (m - n) * (m - nd) / (m * m * (m - 1))
}

func (s hypergeomStd) skew() float64 {
	m, n, nd := s.m, s.n, s.nd
	return (m - 2*n) * math.Sqrt(m-1) * (m - 2*nd) / (math.Sqrt(n*nd*(m-n)*(m-nd)) * (m - 2))
}

func (s hypergeomStd) kurtosis() float64 {
	m, n, nd := s.m, s.n, s.nd
	b := m - n
	g2 := m*(m+1) - 6*nd*(m-nd) - 6*n*b
	g2 *= (m - 1) * m * m
	g2 += 6 * n * nd * (m - nd) * b * (5*m - 6)
	g2 /= n * nd * (m - nd) * b * (m - 2) * (m - 3)
	return g2
}

// entropy sums -p log p over the support from the mode outwards (SciPy
// sums the whole support), or uses the Edgeworth expansion for variances
// above 1e6.
func (s hypergeomStd) entropy() float64 {
	lo, hi := s.support()
	switch v := s.variance(); {
	case lo == hi:
		return 0
	case v > entropyEdgeworthMin:
		return entropyLargeVar(v, s.skew(), s.kurtosis())
	}
	mode := math.Floor((s.n + 1) * (s.nd + 1) / (s.m + 2))
	return entropySum(math.Max(lo, math.Min(mode, hi)), lo, hi, s.term)
}

// munp is E[X^n] from the factorial moments n_(j) N_(j) / M_(j) (falling
// factorials).
func (s hypergeomStd) munp(n int) float64 {
	return rawFromFactorial(n, func(j int) float64 {
		r := 1.0
		for i := 0; i < j && r != 0; i++ {
			fi := float64(i)
			r *= (s.n - fi) * (s.nd - fi) / (s.m - fi)
		}
		return r
	})
}

// rvs is numpy's Generator.hypergeometric(n, M-n, N), which SciPy calls.
func (s hypergeomStd) rvs(g *random.Generator, size []int) (*ndarray.Array, error) {
	return hypergeometricDraws(g, int64(s.n), int64(s.bad), int64(s.nd), size)
}
