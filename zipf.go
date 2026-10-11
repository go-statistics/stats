package stats

import (
	"math"

	"github.com/go-ndarray/ndarray"
	"github.com/go-ndarray/ndarray/random"
	"github.com/go-numerics/special"
	"github.com/go-statistics/stats/internal/dd"
)

//stats:discrete Zipf zipf a=A

// Zipf is the zeta distribution: mass k^-A / ζ(A) on k = 1, 2, ..., for
// A > 1: scipy.stats.zipf(a, loc).
//
// SF is ζ(A, k+1)/ζ(A) (special.HurwitzZeta). CDF is 1 - SF where SF is
// below 1/2 and otherwise the partial sum Σ_{j<=k} j^-A / ζ(A), by
// Euler-Maclaurin summation beyond j = 20 (the difference ζ(A) - ζ(A, k+1)
// would cancel: SciPy sums the masses one by one). Entropy is
// log ζ(A) - A ζ'(A)/ζ(A), with -ζ'(A) = Σ log j / j^A by the same
// summation (SciPy sums at most 1000 masses, which for small A misses much
// of the tail). The raw moments are SciPy's ζ(A-n)/ζ(A) where A > n+1,
// +Inf otherwise; the variance, skewness and kurtosis are +Inf or NaN
// where SciPy's generic rv_discrete.stats makes them so, and otherwise
// come from the moments of X-1, Σ_(k>=2) (k-1)^r k^-A / ζ(A), formed from
// the differences ζ(A-j, 2) - ζ(A, 2) without the cancellation of
// SciPy's raw-moment differences (which loses six digits at A = 20).
type Zipf struct {
	A, Loc float64
}

func (d Zipf) std() zipfStd {
	eta := special.HurwitzZeta(d.A, 2)
	return zipfStd{a: d.A, z: special.Zeta(d.A), lz: log1pDD(dd.F(eta)), eta: eta}
}

// zipfStd holds a, ζ(a), log ζ(a) and η(a) = ζ(a) - 1 = ζ(a, 2) (the
// last two accurate also where ζ(a) rounds to about 1).
type zipfStd struct {
	a, z float64
	lz   dd.DD
	eta  float64
}

// momentsFromMunp: SciPy's zipf has no _stats, its moment(n) is _munp(n).
func (s zipfStd) momentsFromMunp() {}

func (s zipfStd) valid() bool               { return s.a > 1 && !math.IsInf(s.a, 1) }
func (s zipfStd) support() (lo, hi float64) { return 1, math.Inf(1) }

// mlog is -a log k as a double-double.
func (s zipfStd) mlog(k float64) dd.DD { return dd.MulF(dd.Log(k), -s.a) }

func (s zipfStd) pmf(k float64) float64 {
	return expScaled(s.mlog(k), dd.Div(dd.F(1), dd.F(s.z)))
}

func (s zipfStd) logpmf(k float64) float64 { return dd.Sub(s.mlog(k), s.lz).Float() }

func (s zipfStd) sf(k float64) float64 { return special.HurwitzZeta(s.a, k+1) / s.z }

func (s zipfStd) cdf(k float64) float64 {
	if sf := s.sf(k); sf < 0.5 {
		return 1 - sf
	}
	return zetaPartial(s.a, k).Float() / s.z
}

func (s zipfStd) logcdf(k float64) float64 {
	if sf := s.sf(k); sf < 0.5 {
		return math.Log1p(-sf)
	}
	return dd.Sub(logDD(zetaPartial(s.a, k)), s.lz).Float()
}

// logsf is log SF, and below the normal range -a log(k+1) - log ζ(a) plus
// the log of Σ_j ((k+1)/(k+1+j))^a, summed by ratios.
func (s zipfStd) logsf(k float64) float64 {
	return logCDFFrom(s.sf(k), s.cdf(k), func() (float64, dd.DD, bool) {
		q := k + 1
		sum, ok := ratioSum(q, 1, math.Inf(1), tailBudget, func(j float64) dd.DD {
			return dd.F(math.Pow(j/(j+1), s.a))
		})
		return s.logpmf(q), sum, ok
	})
}

// guess inverts SF(k) ≈ k^(1-a) / ((a-1) ζ(a)) at q, within [1, 2^62].
func (s zipfStd) guess(q float64) float64 {
	g := math.Pow((s.a-1)*s.z*q, -1/(s.a-1))
	return math.Min(g, 0x1p62)
}

func (s zipfStd) ppf(p float64) float64 {
	return smallestK(s.guess(1-p), 1, math.Inf(1), func(k float64) bool { return s.cdf(k) >= p })
}

func (s zipfStd) isf(q float64) float64 {
	return smallestK(s.guess(q), 1, math.Inf(1), func(k float64) bool { return s.sf(k) <= q })
}

// munp is SciPy's: ζ(a-n)/ζ(a) for a > n+1, else +Inf.
func (s zipfStd) munp(n int) float64 {
	if s.a > float64(n+1) {
		return special.Zeta(s.a-float64(n)) / s.z
	}
	return math.Inf(1)
}

// ey returns E[(X-1)^r] = Σ_j C(r,j) (-1)^(r-j) δ_j, δ_j = m_j - 1 =
// (ζ(a-j, 2) - ζ(a, 2))/ζ(a) (the constant terms cancel exactly), for
// a > r+1.
func (s zipfStd) ey(r int) float64 {
	t, c := 0.0, 1.0 // c = C(r, j)
	for j := 0; j <= r; j++ {
		if j > 0 {
			c = c * float64(r-j+1) / float64(j)
			d := (special.HurwitzZeta(s.a-float64(j), 2) - s.eta) / s.z
			if (r-j)%2 == 1 {
				d = -d
			}
			t += c * d
		}
	}
	return t
}

// The moments are +Inf or NaN where SciPy's rv_discrete.stats makes them
// so from infinite raw moments: the mean is +Inf for a <= 2, the variance
// for a <= 3; the skewness is NaN for a <= 3 and +Inf for a <= 4, the
// kurtosis NaN for a <= 4 and +Inf for a <= 5.
func (s zipfStd) mean() float64 {
	if s.a <= 2 {
		return math.Inf(1)
	}
	return 1 + s.ey(1)
}

func (s zipfStd) variance() float64 {
	if s.a <= 3 {
		return math.Inf(1)
	}
	m := s.ey(1)
	return s.ey(2) - m*m
}

func (s zipfStd) skew() float64 {
	switch {
	case s.a <= 3:
		return math.NaN()
	case s.a <= 4:
		return math.Inf(1)
	}
	m, m2 := s.ey(1), s.ey(2)
	v := m2 - m*m
	return (s.ey(3) - 3*m*m2 + 2*m*m*m) / math.Pow(v, 1.5)
}

func (s zipfStd) kurtosis() float64 {
	switch {
	case s.a <= 4:
		return math.NaN()
	case s.a <= 5:
		return math.Inf(1)
	}
	m, m2, m3 := s.ey(1), s.ey(2), s.ey(3)
	v := m2 - m*m
	return (s.ey(4)-4*m*m3+6*m*m*m2-3*m*m*m*m)/(v*v) - 3
}

// entropy is log ζ(a) + a S/ζ(a), S = Σ log j / j^a.
func (s zipfStd) entropy() float64 {
	return dd.AddF(s.lz, s.a*zetaLogSum(s.a)/s.z).Float()
}

func (s zipfStd) rvs(g *random.Generator, size []int) (*ndarray.Array, error) {
	return g.Zipf(s.a, size...)
}

// emStart is where the Euler-Maclaurin summation of the zeta sums starts;
// the terms below are added one by one.
const emStart = 20

// bernoulliEM holds B_2i / (2i)!, i = 1..6, the Euler-Maclaurin weights.
var bernoulliEM = [...]float64{
	1.0 / 12, -1.0 / 720, 1.0 / 30240, -1.0 / 1209600, 1.0 / 47900160, -691.0 / 1307674368000,
}

// zetaPartial returns Σ_{j=1}^{k} j^-a as a double-double: the first terms
// one by one and the rest, from emStart to k, by Euler-Maclaurin summation
// with the integral (m^(1-a) - k^(1-a))/(a-1) taken through expm1 so that it
// stays accurate as a approaches 1.
func zetaPartial(a, k float64) dd.DD {
	s := dd.F(0)
	for j := 1.0; j <= math.Min(k, emStart-1); j++ {
		s = dd.Add(s, expDD2(dd.MulF(dd.Log(j), -a)))
	}
	if k < emStart {
		return s
	}
	m := float64(emStart)
	// ∫_m^k x^-a dx = m^(1-a) (1 - (k/m)^(1-a)) / (a-1)
	l := dd.Sub(dd.Log(k), dd.Log(m))
	integral := dd.Div(dd.Mul(expDD2(dd.MulF(dd.Log(m), 1-a)), dd.Neg(expm1DD(dd.MulF(l, 1-a)))), dd.F(a-1))
	fm, fk := math.Pow(m, -a), math.Pow(k, -a)
	corr := (fm + fk) / 2
	// f^(r)(x) = (-1)^r a(a+1)...(a+r-1) x^(-a-r); the weights multiply the
	// odd derivatives f^(2i-1)(k) - f^(2i-1)(m).
	dm, dk := fm, fk
	for i, w := range bernoulliEM {
		r := float64(2*i + 1) // the derivative order 2i+1 = 1, 3, ...
		// from order r-2 (or 0) to order r: two (one, first time) steps
		if i == 0 {
			dm *= -a / m
			dk *= -a / k
		} else {
			dm *= (a + r - 2) * (a + r - 1) / (m * m)
			dk *= (a + r - 2) * (a + r - 1) / (k * k)
		}
		corr += w * (dk - dm)
	}
	return dd.AddF(dd.Add(s, integral), corr)
}

// zetaLogSum returns Σ_{j>=2} log j / j^a = -ζ'(a): the first terms one by
// one, the rest by Euler-Maclaurin summation, with f(x) = x^-a log x and
// f^(r)(x) = x^(-a-r) (α_r log x + β_r), α_(r+1) = -(a+r) α_r,
// β_(r+1) = -(a+r) β_r + α_r.
func zetaLogSum(a float64) float64 {
	s := dd.F(0)
	for j := 2.0; j < emStart; j++ {
		lj := dd.Log(j)
		s = dd.Add(s, dd.Mul(lj, expDD2(dd.MulF(lj, -a))))
	}
	m := float64(emStart)
	lm := math.Log(m)
	pm := math.Pow(m, 1-a)
	am1 := a - 1
	// ∫_m^∞ x^-a log x dx = m^(1-a) (log m/(a-1) + 1/(a-1)²)
	tail := pm * (lm/am1 + 1/(am1*am1))
	tail += pm / m * lm / 2 // f(m)/2
	al, be := 1.0, 0.0
	for r := 0; r < 2*len(bernoulliEM); r++ {
		al, be = -(a+float64(r))*al, -(a+float64(r))*be+al
		if r%2 == 0 { // now of order r+1, odd
			tail -= bernoulliEM[r/2] * math.Pow(m, -a-float64(r+1)) * (al*lm + be)
		}
	}
	return dd.AddF(s, tail).Float()
}
