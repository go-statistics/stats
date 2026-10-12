package stats

import (
	"errors"
	"math"
	"testing"

	"github.com/go-ndarray/ndarray"
	"github.com/go-ndarray/ndarray/random"
	"github.com/go-statistics/stats/internal/dd"
)

// The masses and distribution functions allocate nothing.
func TestDiscreteAllocs(t *testing.T) {
	ds := []struct {
		d Discrete
		k float64
	}{
		{Bernoulli{P: 0.3}, 1},
		{Binomial{N: 1000, P: 0.3}, 280},
		{Poisson{Mu: 40}, 35},
		{Geometric{P: 0.01}, 50},
		{NegativeBinomial{N: 3.5, P: 0.2}, 9},
		{Hypergeometric{Total: 500, Good: 120, Draws: 80}, 15},
		{DiscreteUniform{Low: -3, High: 9}, 2},
		{Zipf{A: 2.5}, 40},
	}
	for _, c := range ds {
		n := testing.AllocsPerRun(20, func() {
			c.d.PMF(c.k)
			c.d.LogPMF(c.k)
			c.d.CDF(c.k)
			c.d.SF(c.k)
		})
		if n != 0 {
			t.Errorf("%T: %v allocations per PMF/LogPMF/CDF/SF", c.d, n)
		}
	}
}

// Beyond 2^53 neighbouring integers are 2 or more apart; the search must
// still end (it once looped when the midpoint rounded up to hi).
func TestSmallestKBeyond2p53(t *testing.T) {
	const want = 9.3e16
	got := smallestK(1, 0, math.Inf(1), func(k float64) bool { return k >= want })
	if !(got >= want && got-want <= 16) {
		t.Errorf("smallestK = %v, want about %v", got, want)
	}
	g := Geometric{P: 5.666820945262342e-18}
	if k := g.ISF(0.6078924681088546); !(k > 8e16 && k < 1e17) {
		t.Errorf("Geometric ISF = %v", k)
	}
}

// The Edgeworth expansion of the entropy agrees with the exact sum just
// above the variance where it takes over.
func TestEntropyEdgeworth(t *testing.T) {
	cases := []struct {
		name string
		s    interface {
			variance() float64
			entropy() float64
			skew() float64
			kurtosis() float64
		}
		sum func() float64
	}{
		{"binom", Binomial{N: 4_400_000, P: 0.5}.std(), func() float64 {
			s := Binomial{N: 4_400_000, P: 0.5}.std()
			return entropySum(2_200_000, 0, s.n, s.term)
		}},
		{"binom skewed", Binomial{N: 120_000_000, P: 0.01}.std(), func() float64 {
			s := Binomial{N: 120_000_000, P: 0.01}.std()
			return entropySum(1_200_000, 0, s.n, s.term)
		}},
		{"poisson", Poisson{Mu: 1.3e6}.std(), func() float64 {
			s := Poisson{Mu: 1.3e6}.std()
			return entropySum(1.3e6, 0, math.Inf(1), s.term)
		}},
		{"nbinom", NegativeBinomial{N: 3e7, P: 0.6}.std(), func() float64 {
			s := NegativeBinomial{N: 3e7, P: 0.6}.std()
			return entropySum(math.Floor((s.n-1)*0.4/0.6), 0, math.Inf(1), s.term)
		}},
		{"hypergeom", Hypergeometric{Total: 40_000_000, Good: 16_000_000, Draws: 10_000_000}.std(), func() float64 {
			s := Hypergeometric{Total: 40_000_000, Good: 16_000_000, Draws: 10_000_000}.std()
			return entropySum(4_000_000, 0, 1e7, s.term)
		}},
	}
	for _, c := range cases {
		if c.s.variance() <= entropyEdgeworthMin {
			t.Fatalf("%s: variance %v too small for the test", c.name, c.s.variance())
		}
		got, want := c.s.entropy(), c.sum()
		if math.Abs(got-want) > 1.5e-15*want { // a few ulps: both sides round
			t.Errorf("%s: Edgeworth entropy %v, sum %v (rel %.2g)", c.name, got, want, (got-want)/want)
		}
	}
	// A skewed negative binomial of large variance is summed, not expanded.
	nb := NegativeBinomial{N: 0.5, P: 0.0005}.std()
	if v := nb.variance(); v < entropyEdgeworthMin {
		t.Fatalf("variance %v", v)
	}
	if h := nb.entropy(); !(h > 0 && h < 20) {
		t.Errorf("nbinom(0.5, 0.0005) entropy %v", h)
	}
}

// Degenerate distributions: one point carries all the mass.
func TestDiscreteDegenerate(t *testing.T) {
	inf := math.Inf(1)
	for _, d := range []Discrete{Binomial{N: 6, P: 0}, Poisson{Mu: 0}, NegativeBinomial{N: 2, P: 1}, Geometric{P: 1}, Binomial{N: 6, P: 1}} {
		lo, _ := d.Support()
		at := lo
		if b, ok := d.(Binomial); ok && b.P == 1 {
			at = 6
		}
		if d.PMF(at) != 1 || d.LogPMF(at) != 0 || d.Entropy() != 0 {
			t.Errorf("%+v: pmf(%v) = %v, logpmf %v, entropy %v", d, at, d.PMF(at), d.LogPMF(at), d.Entropy())
		}
		other := at + 1
		if at > lo {
			other = at - 1
		}
		if d.PMF(other) != 0 || d.LogPMF(other) != -inf {
			t.Errorf("%+v: pmf(%v) = %v, logpmf %v", d, other, d.PMF(other), d.LogPMF(other))
		}
		if at == lo && (d.LogSF(at) != -inf || d.LogCDF(at) != 0) {
			t.Errorf("%+v: logsf(%v) = %v, logcdf %v", d, at, d.LogSF(at), d.LogCDF(at))
		}
	}
	if h := (Bernoulli{P: 0}).Entropy(); h != 0 {
		t.Errorf("Bernoulli(0) entropy %v", h)
	}
	// SciPy's logpmf is -8.9e-16 there, its third and fourth moments NaN
	hg := Hypergeometric{Total: 10, Good: 10, Draws: 4}
	if hg.Entropy() != 0 || hg.PMF(4) != 1 || hg.LogPMF(4) != 0 || hg.Moment(3) != 64 {
		t.Errorf("degenerate hypergeometric: entropy %v pmf %v logpmf %v moment(3) %v", hg.Entropy(), hg.PMF(4), hg.LogPMF(4), hg.Moment(3))
	}
	// shiftDraws converts non-integer draws (none of the generators used
	// here returns them, but the helper accepts any array)
	fl, _ := ndarray.FromData([]float64{1, 2.5}, 2)
	if v, _ := ndarray.Data[int64](shiftDraws(fl, 1)); v[0] != 2 || v[1] != 3 {
		t.Errorf("shiftDraws = %v", v)
	}
	g := Geometric{P: 1}
	if g.CDF(3) != 1 || g.SF(3) != 0 || g.PPF(0.5) != 1 || g.ISF(0.5) != 1 {
		t.Errorf("Geometric(1): cdf %v sf %v ppf %v isf %v", g.CDF(3), g.SF(3), g.PPF(0.5), g.ISF(0.5))
	}
}

// The logarithms below the normal range come from the tail sums; when a
// sum does not converge within its budget the logarithm of the
// (underflowed) probability is returned.
func TestTailSums(t *testing.T) {
	s, ok := ratioSum(0, 1, math.Inf(1), 3, func(float64) dd.DD { return dd.F(1) })
	if ok || s.Hi != 4 {
		t.Errorf("ratioSum over budget = %v, %v", s, ok)
	}
	lp := logCDFFrom(1e-310, 1, func() (float64, dd.DD, bool) { return 0, dd.F(1), false })
	if lp != math.Log(1e-310) {
		t.Errorf("logCDFFrom fallback = %v", lp)
	}
	// Deep tails, against values from mpmath.
	for _, c := range []struct {
		name      string
		got, want float64
	}{
		{"binom logcdf", Binomial{N: 4000, P: 0.6}.LogCDF(1000), -1014.3351956733784},
		{"binom logsf", Binomial{N: 4000, P: 0.4}.LogSF(3000), -1015.8399866070057},
		{"poisson logcdf", Poisson{Mu: 3000}.LogCDF(1000), -905.3553953246842},
		{"poisson logsf", Poisson{Mu: 30}.LogSF(500), -943.485563241573},
		{"nbinom logcdf", NegativeBinomial{N: 1000, P: 0.1}.LogCDF(1000), -1025.5599880535476},
		{"nbinom logsf", NegativeBinomial{N: 10, P: 0.9}.LogSF(1000), -2256.413145304911},
		{"zipf logsf", Zipf{A: 400}.LogSF(5), -716.703787691222},
	} {
		if math.Abs(c.got-c.want) > 4e-15*math.Abs(c.want) {
			t.Errorf("%s = %v, want %v", c.name, c.got, c.want)
		}
	}
}

// The moments of Zipf are +Inf or NaN where SciPy's are.
func TestZipfMoments(t *testing.T) {
	nan, inf := math.NaN(), math.Inf(1)
	same := func(a, b float64) bool { return a == b || math.IsNaN(a) && math.IsNaN(b) }
	for _, c := range []struct {
		a                        float64
		mean, vr, skew, kurt, m3 float64
	}{
		{1.5, inf, inf, nan, nan, inf},
		{2.5, 0, inf, nan, nan, inf},
		{3.5, 0, 0, inf, nan, inf},
		{4.5, 0, 0, 0, inf, 0},
	} {
		d := Zipf{A: c.a}
		got := []float64{d.Mean(), d.Var(), d.Skew(), d.Kurtosis(), d.Moment(3)}
		want := []float64{c.mean, c.vr, c.skew, c.kurt, c.m3}
		for i := range got {
			if want[i] == 0 { // finite
				if !finite(got[i]) {
					t.Errorf("Zipf(%v) #%d = %v, want finite", c.a, i, got[i])
				}
				continue
			}
			if !same(got[i], want[i]) {
				t.Errorf("Zipf(%v) #%d = %v, want %v", c.a, i, got[i], want[i])
			}
		}
	}
}

// numpy refuses populations of 10^9 or more good or bad items.
func TestHypergeometricDrawLimit(t *testing.T) {
	big := int64(2e9)
	d := Hypergeometric{Total: big + 5, Good: big, Draws: 3}
	if _, err := d.Rvs(random.DefaultRNG(1), 2); !errors.Is(err, ErrParam) {
		t.Errorf("Rvs with good = 2e9: %v", err)
	}
}
