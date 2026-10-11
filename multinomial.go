package stats

import (
	"fmt"
	"math"

	"github.com/go-ndarray/ndarray"
	"github.com/go-ndarray/ndarray/random"
	"github.com/go-numerics/special"
	"github.com/go-statistics/stats/internal/dd"
)

// Multinomial is the multinomial distribution of the counts in K
// categories of N independent trials with category probabilities P:
// scipy.stats.multinomial(n, p).
type Multinomial struct {
	n int
	p []float64
}

// NewMultinomial returns the multinomial distribution. It returns an error
// wrapping ErrParam when n < 0, a probability lies outside [0, 1], or the
// probabilities do not sum to 1 within 10ε (SciPy returns NaN from every
// method for such parameters).
func NewMultinomial(n int, p []float64) (*Multinomial, error) {
	if n < 0 {
		return nil, fmt.Errorf("%w: n must be at least 0", ErrParam)
	}
	if len(p) == 0 {
		return nil, fmt.Errorf("%w: p is empty", ErrParam)
	}
	for _, v := range p {
		if !(v >= 0 && v <= 1) {
			return nil, fmt.Errorf("%w: every p must lie in [0, 1]", ErrParam)
		}
	}
	if !(math.Abs(1-sum(p)) <= 10*0x1p-52) {
		return nil, fmt.Errorf("%w: p must sum to 1 (it sums to %v)", ErrParam, sum(p))
	}
	return &Multinomial{n: n, p: append([]float64(nil), p...)}, nil
}

// N returns the number of trials.
func (m *Multinomial) N() int { return m.n }

// P returns the category probabilities (a copy).
func (m *Multinomial) P() []float64 { return append([]float64(nil), m.p...) }

// logPMF is log n! + Σ (xᵢ log pᵢ - log xᵢ!) summed in double-double; -∞
// off the support (a negative or fractional count, or counts not summing
// to n).
func (m *Multinomial) logPMF(x []float64) dd.DD {
	t := 0.0
	for _, v := range x {
		if v < 0 || v != math.Trunc(v) {
			return dd.F(math.Inf(-1))
		}
		t += v
	}
	if t != float64(m.n) {
		return dd.F(math.Inf(-1))
	}
	s := dd.F(special.LogGamma(float64(m.n) + 1))
	for i, v := range x {
		if v == 0 {
			continue
		}
		if m.p[i] == 0 {
			return dd.F(math.Inf(-1))
		}
		s = dd.Add(s, dd.MulF(dd.LogAccurate(m.p[i]), v))
		s = dd.AddF(s, -special.LogGamma(v+1))
	}
	return s
}

func (m *Multinomial) check(x []float64) error {
	if len(x) != len(m.p) {
		return fmt.Errorf("%w: a point needs %d counts, it has %d", ErrData, len(m.p), len(x))
	}
	return nil
}

// LogPMF returns the log-probability of the counts x (K of them):
// scipy.stats.multinomial.logpmf. Counts that are negative, fractional or
// do not sum to N have probability 0 (-Inf); a NaN count gives NaN.
func (m *Multinomial) LogPMF(x []float64) (float64, error) {
	if err := m.check(x); err != nil {
		return math.NaN(), err
	}
	for _, v := range x {
		if math.IsNaN(v) {
			return math.NaN(), nil
		}
	}
	h := m.logPMF(x)
	return h.Hi + h.Lo, nil
}

// PMF returns the probability of the counts x: scipy.stats.multinomial.pmf.
func (m *Multinomial) PMF(x []float64) (float64, error) {
	l, err := m.LogPMF(x)
	if err != nil || math.IsNaN(l) || math.IsInf(l, -1) {
		return math.Exp(l), err
	}
	return expDD(m.logPMF(x)), nil
}

// LogPMFPoints returns LogPMF at each row of x, an (n, K) array of counts
// (any dtype; a 1-D x is one point).
func (m *Multinomial) LogPMFPoints(x *ndarray.Array) ([]float64, error) {
	return m.rows(x, m.LogPMF)
}

// PMFPoints is PMF at each row of x (see LogPMFPoints).
func (m *Multinomial) PMFPoints(x *ndarray.Array) ([]float64, error) {
	return m.rows(x, m.PMF)
}

func (m *Multinomial) rows(x *ndarray.Array, f func([]float64) (float64, error)) ([]float64, error) {
	v, n, err := mvPoints(x, len(m.p))
	if err != nil {
		return nil, err
	}
	k := len(m.p)
	out := make([]float64, n)
	for i := range n {
		out[i], _ = f(v[i*k : (i+1)*k])
	}
	return out, nil
}

// Mean returns n·p: scipy.stats.multinomial.mean.
func (m *Multinomial) Mean() []float64 {
	out := make([]float64, len(m.p))
	for i, p := range m.p {
		out[i] = float64(m.n) * p
	}
	return out
}

// Cov returns the K×K covariance matrix n(diag(p) - p·pᵀ):
// scipy.stats.multinomial.cov.
func (m *Multinomial) Cov() *ndarray.Array {
	k := len(m.p)
	n := float64(m.n)
	c := make([]float64, k*k)
	for i := range k {
		for j := range k {
			c[i*k+j] = n * float64(-m.p[i]*m.p[j])
		}
		c[i*k+i] += float64(n * m.p[i])
	}
	return mustArray(c, k, k)
}

// Entropy returns the entropy -log n! - nΣpᵢlog pᵢ + Σᵢ Σₓ₌₁ⁿ
// binom.pmf(x; n, pᵢ)·log x!, SciPy's formula
// (scipy.stats.multinomial.entropy), with the binomial probabilities by
// Loader's saddle-point algorithm.
func (m *Multinomial) Entropy() float64 {
	k := len(m.p)
	e := make([]float64, k)
	for i, p := range m.p {
		if p > 0 {
			e[i] = -float64(p * math.Log(p))
		}
	}
	term1 := float64(m.n)*sum(e) - special.LogGamma(float64(m.n)+1)
	t := make([]float64, 0, m.n*k)
	for x := 1; x <= m.n; x++ {
		lg := special.LogGamma(float64(x) + 1)
		for _, p := range m.p {
			t = append(t, float64(binomPMF(x, m.n, p)*lg))
		}
	}
	return term1 + sum(t)
}

// Rvs draws size points, an int64 array of shape (size, K), as
// scipy.stats.multinomial.rvs(n, p, size, random_state=g) does: NumPy's
// Generator.multinomial, reproduced bit for bit by go-ndarray.
func (m *Multinomial) Rvs(g *random.Generator, size int) (*ndarray.Array, error) {
	r, err := g.Multinomial(int64(m.n), m.p, size)
	if err != nil {
		return nil, fmt.Errorf("%w: %v", ErrParam, err)
	}
	return r, nil
}

// binomPMF is the binomial probability C(n,x)pˣ(1-p)ⁿ⁻ˣ by Loader's
// algorithm ("Fast and accurate computation of binomial probabilities",
// 2000): exp of a sum of Stirling-series errors and deviance terms, accurate
// to a few ULP where the probability does not underflow.
func binomPMF(x, n int, p float64) float64 {
	q := 1 - p
	switch {
	case p == 0:
		if x == 0 {
			return 1
		}
		return 0
	case q == 0:
		if x == n {
			return 1
		}
		return 0
	case x == 0:
		return math.Exp(float64(n) * math.Log1p(-p))
	case x == n:
		return math.Exp(float64(n) * math.Log(p))
	}
	fx, fn := float64(x), float64(n)
	lc := stirlerr(fn) - stirlerr(fx) - stirlerr(fn-fx) - bd0(fx, fn*p) - bd0(fn-fx, fn*q)
	lf := math.Log(2*math.Pi) + math.Log(fx) + math.Log1p(-fx/fn)
	return math.Exp(lc - 0.5*lf)
}

// stirlerr is log(n!) - log(√(2πn)(n/e)ⁿ), the error of Stirling's
// formula, for integer n ≥ 1.
func stirlerr(n float64) float64 {
	if n <= 15 {
		return stirlerrTable[int(n)-1]
	}
	const (
		s0 = 1.0 / 12
		s1 = 1.0 / 360
		s2 = 1.0 / 1260
		s3 = 1.0 / 1680
		s4 = 1.0 / 1188
	)
	nn := n * n
	switch {
	case n > 500:
		return (s0 - s1/nn) / n
	case n > 80:
		return (s0 - (s1-s2/nn)/nn) / n
	case n > 35:
		return (s0 - (s1-(s2-s3/nn)/nn)/nn) / n
	}
	return (s0 - (s1-(s2-(s3-s4/nn)/nn)/nn)/nn) / n
}

// stirlerrTable is stirlerr(1..15), correctly rounded (mpmath).
var stirlerrTable = [15]float64{0.08106146679532726, 0.0413406959554093, 0.02767792568499834, 0.020790672103765093, 0.016644691189821193, 0.013876128823070748, 0.01189670994589177, 0.010411265261972096, 0.009255462182712733, 0.00833056343336287, 0.007573675487951841, 0.00694284010720953, 0.006408994188004207, 0.0059513701127588475, 0.005554733551962801}

// bd0 is the deviance term x log(x/np) + np - x, computed without
// cancellation when x is close to np.
func bd0(x, np float64) float64 {
	if math.Abs(x-np) < 0.1*(x+np) {
		v := (x - np) / (x + np)
		s := (x - np) * v
		ej := 2 * x * v
		v *= v
		for j := 1; ; j++ {
			ej *= v
			s1 := s + ej/float64(2*j+1)
			if s1 == s {
				return s1
			}
			s = s1
		}
	}
	return x*math.Log(x/np) + np - x
}
