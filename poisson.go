package stats

import (
	"math"

	"github.com/go-ndarray/ndarray"
	"github.com/go-ndarray/ndarray/random"
	"github.com/go-numerics/special"
	"github.com/go-statistics/stats/internal/dd"
)

//stats:discrete Poisson poisson mu=Mu

// Poisson is the Poisson distribution of mean Mu >= 0: mass
// Mu^k e^(-Mu) / k! on k = 0, 1, ...: scipy.stats.poisson(mu, loc). Mu = 0
// is the distribution concentrated at 0, as in SciPy.
//
// PMF and LogPMF use Loader's form exp(-δ(k) - D(k, Mu)) / sqrt(2πk) (see
// poissonTerm), accurate for every Mu up to 1e15 and beyond, where SciPy's
// exp(k log Mu - log Γ(k+1) - Mu) loses about log2(Mu) bits to
// cancellation. CDF and SF are special.GammaQ and special.GammaP.
type Poisson struct {
	Mu, Loc float64
}

func (d Poisson) std() poissonStd { return poissonStd{mu: d.Mu} }

type poissonStd struct{ mu float64 }

func (s poissonStd) valid() bool               { return s.mu >= 0 && !math.IsInf(s.mu, 1) }
func (s poissonStd) support() (lo, hi float64) { return 0, math.Inf(1) }

func (s poissonStd) term(k float64) (e, c dd.DD) {
	if s.mu == 0 {
		if k == 0 {
			return dd.F(0), dd.F(1)
		}
		return dd.F(math.Inf(-1)), dd.F(1)
	}
	return poissonTerm(k, s.mu)
}

func (s poissonStd) pmf(k float64) float64    { return massOf(s.term(k)) }
func (s poissonStd) logpmf(k float64) float64 { return logMassOf(s.term(k)) }

func (s poissonStd) cdf(k float64) float64 { return special.GammaQ(k+1, s.mu) }
func (s poissonStd) sf(k float64) float64  { return special.GammaP(k+1, s.mu) }

func (s poissonStd) logcdf(k float64) float64 {
	return logCDFFrom(s.cdf(k), s.sf(k), func() (float64, dd.DD, bool) {
		// pmf(j-1)/pmf(j) = j / mu
		sum, ok := ratioSum(k, -1, 0, tailBudget, func(j float64) dd.DD { return dd.DivF(dd.F(j), s.mu) })
		return s.logpmf(k), sum, ok
	})
}

func (s poissonStd) logsf(k float64) float64 {
	return logCDFFrom(s.sf(k), s.cdf(k), func() (float64, dd.DD, bool) {
		// pmf(j+1)/pmf(j) = mu / (j+1)
		sum, ok := ratioSum(k+1, 1, math.Inf(1), tailBudget, func(j float64) dd.DD { return dd.DivF(dd.F(s.mu), j+1) })
		return s.logpmf(k + 1), sum, ok
	})
}

func (s poissonStd) guess(z float64) float64 {
	return normalGuess(s.mu, math.Sqrt(s.mu), s.skew(), z)
}

func (s poissonStd) ppf(p float64) float64 {
	return smallestK(s.guess(special.Ndtri(p)), 0, math.Inf(1), func(k float64) bool { return s.cdf(k) >= p })
}

func (s poissonStd) isf(q float64) float64 {
	return smallestK(s.guess(-special.Ndtri(q)), 0, math.Inf(1), func(k float64) bool { return s.sf(k) <= q })
}

func (s poissonStd) mean() float64     { return s.mu }
func (s poissonStd) variance() float64 { return s.mu }

// skew and kurtosis are 1/sqrt(mu) and 1/mu, +Inf at mu = 0 (as SciPy).
func (s poissonStd) skew() float64     { return math.Sqrt(1 / s.mu) }
func (s poissonStd) kurtosis() float64 { return 1 / s.mu }

// entropy sums -p log p (SciPy sums it too, truncated at 1000 terms, which
// is too few beyond mu of about 1e4), or, above mu = 1e6, the expansion
// log(2πe mu)/2 - 1/(12mu) - 1/(24mu²).
func (s poissonStd) entropy() float64 {
	switch {
	case s.mu > entropyEdgeworthMin:
		return entropyLargeVar(s.mu, s.skew(), s.kurtosis())
	case s.mu == 0:
		return 0
	}
	return entropySum(math.Floor(s.mu), 0, math.Inf(1), s.term)
}

// munp is E[X^n] from the factorial moments mu^j.
func (s poissonStd) munp(n int) float64 {
	return rawFromFactorial(n, func(j int) float64 { return math.Pow(s.mu, float64(j)) })
}

func (s poissonStd) rvs(g *random.Generator, size []int) (*ndarray.Array, error) {
	return g.Poisson(s.mu, size...)
}
