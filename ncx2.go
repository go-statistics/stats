package stats

import (
	"math"

	"github.com/go-ndarray/ndarray"
	"github.com/go-ndarray/ndarray/random"
	"github.com/go-numerics/special"
	"github.com/go-statistics/stats/internal/dd"
)

//stats:continuous NoncentralChiSquared ncx2 df=Df nc=Nc

// NoncentralChiSquared is the noncentral chi-squared distribution with Df
// degrees of freedom and noncentrality Nc (the sum of Df squared normal
// variables whose means have squares summing to Nc):
// scipy.stats.ncx2(df, nc, loc, scale). Nc = 0 is ChiSquared.
//
// Every function is the Poisson mixture Σ_j w_j F_{Df+2j}(x),
// w_j = e^(-Nc/2) (Nc/2)^j / j!, summed from its largest term outwards
// until the terms no longer matter: the density with the central chi-squared
// densities, the cdf and the sf with the incomplete gamma functions P and
// Q of each term, so that both tails are sums of positive terms (no
// cancellation). The largest term is evaluated in double-double and the
// others by the exact ratios between neighbours. PPF and ISF invert the cdf
// and sf (invertCDF).
type NoncentralChiSquared struct {
	Df, Nc, Loc, Scale float64
}

func (d NoncentralChiSquared) std() ncx2Std { return ncx2Std{k: d.Df / 2, mu: d.Nc / 2} }

// ncx2Std holds k = df/2 and μ = nc/2 (the Poisson mean).
type ncx2Std struct{ k, mu float64 }

func (s ncx2Std) valid() bool {
	return s.k > 0 && !math.IsInf(s.k, 1) && s.mu >= 0 && !math.IsInf(s.mu, 1)
}
func (s ncx2Std) support() (lo, hi float64) { return 0, math.Inf(1) }

func (s ncx2Std) central() chi2Std { return chi2Std{k: s.k} }

// mixture sums Σ_j w_j·term(j) where term(j) is a central chi-squared
// quantity of shape k+j, by walking both ways from the index where
// w_j·term(j) is largest (found by a short search from start).
// logTerm(j) is log(w_j·term(j)) in double-double; next(j) is the ratio
// t(j+1)/t(j) of consecutive terms (finite, positive).
func (s ncx2Std) mixture(start float64, logTerm func(j float64) dd.DD, ratio func(j float64) float64) float64 {
	// locate the largest term: the ratio crosses 1 there (log-concave
	// terms); start from the caller's estimate and step to it
	// start is the root of ratio(j) = 1, so ratio(j-1) > 1 at its floor
	// (up to rounding, which only costs a step of the walk below)
	j := math.Max(0, math.Floor(start))
	if !(j < 1<<40) {
		// the largest term is beyond any index that matters: the density
		// is far below the smallest subnormal there
		return 0
	}
	for ratio(j) > 1 {
		j++
	}
	t0 := expDD(logTerm(j))
	if !(t0 > 0) {
		return 0 // every term underflows
	}
	// walk away from the largest term by the exact ratios, re-anchoring on
	// a direct evaluation every 8 steps so that rounding does not
	// accumulate along the walk
	sum := t0
	t := t0
	for i := j; i > 0; i-- {
		if math.Mod(j-i+1, 8) == 0 {
			t = expDD(logTerm(i - 1))
		} else {
			t /= ratio(i - 1)
		}
		sum += t
		if t < sum*0x1p-60 {
			break
		}
	}
	t = t0
	for i := j; ; i++ {
		if math.Mod(i-j+1, 8) == 0 {
			t = expDD(logTerm(i + 1))
		} else {
			t *= ratio(i)
		}
		sum += t
		if t < sum*0x1p-60 || t == 0 {
			break
		}
	}
	return sum
}

// poissonLog is log(e^-μ μ^j / j!) in double-double: the gamma(j+1)
// density at μ.
func (s ncx2Std) poissonLog(j float64) dd.DD {
	return gammaLogDensity(j+1, dd.F(s.mu)) // μ > 0 here: μ = 0 is the central case
}

func (s ncx2Std) pdf(x float64) float64 {
	if s.mu == 0 {
		return s.central().pdf(x)
	}
	if x == 0 {
		// only the j = 0 term can be nonzero at 0
		return math.Exp(-s.mu) * s.central().pdf(0)
	}
	y := x / 2
	// t_j = w_j · g_{k+j}(y)/2, t_{j+1}/t_j = μ y / ((j+1)(k+j)): the
	// largest term is near the root of (j+1)(k+j) = μy
	b := s.k + 1
	start := (-b + math.Sqrt(b*b-4*(s.k-s.mu*y))) / 2
	return s.mixture(start, func(j float64) dd.DD {
		// the shape k+j is rounded; d/da log g_a(y) = log y - ψ(a)
		// corrects the rounding to first order
		a, da := dd.TwoSum(s.k, j)
		l := dd.Add(s.poissonLog(j), dd.AddF(gammaLogDensity(a, dd.F(y)), -math.Ln2))
		if da != 0 {
			l = dd.AddF(l, da*(math.Log(y)-special.Digamma(a)))
		}
		return l
	}, func(j float64) float64 {
		return s.mu * y / ((j + 1) * (s.k + j))
	})
}

func (s ncx2Std) logpdf(x float64) float64 { return math.Log(s.pdf(x)) }

// tailSum is Σ w_j P(k+j, y) (upper false) or Σ w_j Q(k+j, y).
func (s ncx2Std) tailSum(y float64, upper bool) float64 {
	f := special.GammaP
	if upper {
		f = special.GammaQ
	}
	term := func(j float64) float64 { return expDD(s.poissonLog(j)) * f(s.k+j, y) }
	// walk from the Poisson mode both ways while the terms matter
	j0 := math.Floor(s.mu)
	sum := 0.0
	prev := math.Inf(1)
	for j := j0; j >= 0; j-- {
		t := term(j)
		sum += t
		if t <= sum*0x1p-60 && t <= prev && j < s.mu-10*math.Sqrt(s.mu+1) {
			break
		}
		prev = t
	}
	prev = math.Inf(1)
	for j := j0 + 1; ; j++ {
		t := term(j)
		sum += t
		if (t <= sum*0x1p-60 && t <= prev && j > s.mu+10*math.Sqrt(s.mu+1)) || (t == 0 && j > s.mu) || t != t || j > s.mu+60*math.Sqrt(s.mu+1)+1e5 {
			break
		}
		prev = t
	}
	return sum
}

func (s ncx2Std) cdf(x float64) float64 {
	if s.mu == 0 {
		return s.central().cdf(x)
	}
	return math.Min(1, s.tailSum(x/2, false))
}

func (s ncx2Std) sf(x float64) float64 {
	if s.mu == 0 {
		return s.central().sf(x)
	}
	v := s.tailSum(x/2, true)
	if !(v >= 1e-280) {
		return expDD(s.logSFDeep(x / 2))
	}
	return math.Min(1, v)
}

func (s ncx2Std) logcdf(x float64) float64 { return logPQ(s.cdf(x), s.sf(x)) }

func (s ncx2Std) logsf(x float64) float64 {
	if s.mu != 0 {
		if v := s.tailSum(x/2, true); !(v >= 1e-280) {
			return s.logSFDeep(x / 2).Float()
		}
	}
	return logPQ(s.sf(x), s.cdf(x))
}

// logSFDeep is log Σ w_j Q(k+j, y) in the far upper tail, where the
// terms underflow: each term in logarithms (logUpperGamma), summed by
// log-sum-exp from j = 0 until the Poisson weights end it.
func (s ncx2Std) logSFDeep(y float64) dd.DD {
	var terms []dd.DD
	best := math.Inf(-1)
	for j := 0.0; ; j++ {
		l := dd.Add(s.poissonLog(j), logUpperGamma(s.k+j, y))
		terms = append(terms, l)
		best = math.Max(best, l.Hi)
		if (l.Hi < best-40 && j > s.mu) || l.Hi != l.Hi || j > s.mu+60*math.Sqrt(s.mu+1)+500 {
			break
		}
	}
	sum := 0.0
	for _, l := range terms {
		sum += expDD(dd.AddF(l, -best))
	}
	return dd.Add(dd.F(best), logDD(dd.F(sum)))
}

// guess is the normal approximation to the quantile with lower tail p and
// upper tail q = 1-p (each given exactly by the caller).
func (s ncx2Std) guess(p, q float64) float64 {
	m, v := s.mean(), s.variance()
	z := special.Ndtri(p)
	if q < p {
		z = -special.Ndtri(q)
	}
	return math.Max(m+math.Sqrt(v)*z, m*1e-3)
}

func (s ncx2Std) ppf(p float64) float64 {
	if s.mu == 0 {
		return s.central().ppf(p)
	}
	if p > 0.5 {
		return invertCDF(1-p, true, s.sf, s.pdf, s.guess(p, 1-p), 0, math.Inf(1))
	}
	return invertCDF(p, false, s.cdf, s.pdf, s.guess(p, 1-p), 0, math.Inf(1))
}

func (s ncx2Std) isf(q float64) float64 {
	if s.mu == 0 {
		return s.central().isf(q)
	}
	if q > 0.5 {
		return invertCDF(1-q, false, s.cdf, s.pdf, s.guess(1-q, q), 0, math.Inf(1))
	}
	return invertCDF(q, true, s.sf, s.pdf, s.guess(1-q, q), 0, math.Inf(1))
}

func (s ncx2Std) mean() float64     { return 2*s.k + 2*s.mu }
func (s ncx2Std) variance() float64 { return 2 * (2*s.k + 2*2*s.mu) }
func (s ncx2Std) skew() float64 {
	df, nc := 2*s.k, 2*s.mu
	return math.Sqrt(8) * (df + 3*nc) / math.Sqrt(math.Pow(df+2*nc, 3))
}
func (s ncx2Std) kurtosis() float64 {
	df, nc := 2*s.k, 2*s.mu
	return 12 * (df + 4*nc) / ((df + 2*nc) * (df + 2*nc))
}
func (s ncx2Std) median() float64 { return s.ppf(0.5) }

// entropy is -∫ f log f by exp-sinh quadrature (SciPy integrates
// numerically too).
func (s ncx2Std) entropy() float64 {
	if s.mu == 0 {
		return s.central().entropy()
	}
	return entropyPositive(s.pdf, s.logpdf, s.mean())
}

// munp is E[X^n] by the recurrence of the noncentral chi-squared raw
// moments: μ'_n = 2^(n-1)(n-1)!(df + n·nc) + Σ_{j=1}^{n-1} 2^(j-1)
// (n-1)!/(n-j)! (df + j·nc) μ'_{n-j}.
func (s ncx2Std) munp(n int) float64 {
	df, nc := 2*s.k, 2*s.mu
	m := make([]float64, n+1)
	m[0] = 1
	for r := 1; r <= n; r++ {
		fact := 1.0 // (r-1)!/(r-j)! accumulated
		v := 0.0
		for j := 1; j <= r; j++ {
			if j > 1 {
				fact *= float64(r - j + 1)
			}
			v += math.Ldexp(fact, j-1) * (df + float64(j)*nc) * m[r-j]
		}
		m[r] = v
	}
	return m[n]
}

func (s ncx2Std) rvs(g *random.Generator, size []int) (*ndarray.Array, error) {
	return g.NoncentralChiSquare(2*s.k, 2*s.mu, size...)
}
