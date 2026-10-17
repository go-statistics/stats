package stats

import (
	"fmt"
	"math"

	"github.com/go-ndarray/ndarray"
	"github.com/go-ndarray/ndarray/random"
	"github.com/go-numerics/special"
	"github.com/go-statistics/stats/internal/dd"
)

//stats:continuous Gumbel gumbel_r

// Gumbel is the (right-skewed) Gumbel distribution, the type I extreme
// value distribution of maxima: density exp(-(x + e^(-x))) in standard
// form: scipy.stats.gumbel_r(loc, scale).
//
// e = exp(-x) is carried in double-double: the density is
// exp(-x - e), CDF exp(-e), SF -expm1(-e), LogSF log(1 - exp(-e)) in the
// form that does not cancel (SciPy's log(sf) loses everything where SF is
// near 1), each rounded once; PPF -log(-log p) and ISF
// -log(-log1p(-q)) take the inner logarithm in double-double.
type Gumbel struct {
	Loc, Scale float64
}

func (d Gumbel) std() gumbelStd { return gumbelStd{} }

type gumbelStd struct{}

func (gumbelStd) valid() bool             { return true }
func (gumbelStd) support() (a, b float64) { return math.Inf(-1), math.Inf(1) }

// e returns exp(-x) as a double-double (+Inf below -709.78).
func (gumbelStd) e(x float64) dd.DD { return expD(dd.F(-x)) }

func (s gumbelStd) pdf(x float64) float64 {
	e := s.e(x)
	if math.IsInf(e.Hi, 1) {
		return 0
	}
	return expR(dd.Sub(dd.F(-x), e))
}

func (s gumbelStd) logpdf(x float64) float64 {
	e := s.e(x)
	if math.IsInf(e.Hi, 1) {
		return math.Inf(-1)
	}
	return dd.Sub(dd.F(-x), e).Float()
}

func (s gumbelStd) cdf(x float64) float64    { return expR(dd.Neg(s.e(x))) }
func (s gumbelStd) sf(x float64) float64     { return dd.Neg(expm1D(dd.Neg(s.e(x)))).Float() }
func (s gumbelStd) logcdf(x float64) float64 { return -s.e(x).Float() }

// logsf is log(1 - exp(-e)); for e below 2^-26 it is
// log e + log1p(-e/2 + e²/6 - ...) = -x - e/2 to well below the
// rounding, which keeps full accuracy also where e is subnormal.
func (s gumbelStd) logsf(x float64) float64 {
	e := s.e(x)
	if e.Hi < 0x1p-26 {
		return dd.AddF(dd.F(-x), -0.5*e.Hi).Float()
	}
	return log1mexpD(e).Float()
}

func (gumbelStd) ppf(p float64) float64 { return -logDD(dd.Neg(dd.Log(p))).Float() }
func (gumbelStd) isf(q float64) float64 { return -logDD(dd.Neg(log1pDD(dd.F(-q)))).Float() }

// mean is Euler's γ.
func (gumbelStd) mean() float64 { return 0.5772156649015329 }

// variance is π²/6.
func (gumbelStd) variance() float64 { return 1.6449340668482264 }

// skew is 12√6 ζ(3)/π³.
func (gumbelStd) skew() float64     { return 1.1395470994046486 }
func (gumbelStd) kurtosis() float64 { return 2.4 }

// median is -log(log 2).
func (gumbelStd) median() float64 { return 0.36651292058166435 }

// entropy is γ + 1.
func (gumbelStd) entropy() float64 { return 1.5772156649015329 }

// munp is the raw moment from the cumulants κ1 = γ, κk = (k-1)! ζ(k):
// m_n = Σ_{k=1}^{n} C(n-1, k-1) κk m_{n-k}.
func (gumbelStd) munp(n int) float64 {
	m := make([]float64, n+1)
	m[0] = 1
	for j := 1; j <= n; j++ {
		c := 1.0 // C(j-1, k-1)
		for k := 1; k <= j; k++ {
			kappa := 0.5772156649015329
			if k > 1 {
				kappa = factorial(k-1) * special.Zeta(float64(k))
			}
			m[j] += c * kappa * m[j-k]
			c = c * float64(j-k) / float64(k)
		}
	}
	return m[n]
}

// rvs is SciPy's -log(-log U) for uniform draws U, with correctly rounded
// logarithms.
func (gumbelStd) rvs(g *random.Generator, size []int) (*ndarray.Array, error) {
	u, err := g.Uniform(0, 1, size...)
	if err != nil {
		return nil, err
	}
	v, _ := ndarray.Data[float64](u)
	for i, p := range v {
		v[i] = -crLog(-crLog(p))
	}
	return u, nil
}

// FitGumbel returns the maximum-likelihood Gumbel distribution for x:
// scipy.stats.gumbel_r.fit(x, floc=, fscale=). It solves SciPy's
// likelihood equations: with Scale fixed, Loc = -Scale·log(mean
// exp(-x/Scale)) in closed form; otherwise Scale is the root of SciPy's
// equation (bracketed as SciPy brackets it, from 1) found to full
// precision, and Loc follows.
func FitGumbel[D Data](x D, o *FitOptions) (Gumbel, error) {
	v := floats(x)
	fx, err := o.fixed("Gumbel", "Loc", "Scale")
	if err != nil {
		return Gumbel{}, err
	}
	if err := checkFitData(v); err != nil {
		return Gumbel{}, err
	}
	n := float64(len(v))
	t := make([]float64, len(v))
	locFromScale := func(scale float64) float64 {
		for i, y := range v {
			t[i] = -y / scale
		}
		return -scale * (logSumExp(t) - math.Log(n))
	}
	if scale, ok := fx["Scale"]; ok {
		d := Gumbel{Loc: locFromScale(scale), Scale: scale}
		return d, d.Validate()
	}
	lo, hi := minMax(v)
	if lo == hi {
		return Gumbel{}, fmt.Errorf("%w: Gumbel fit of constant data", ErrData)
	}
	var f func(scale float64) float64
	floc, haveLoc := fx["Loc"]
	if haveLoc {
		f = func(scale float64) float64 {
			for i, y := range v {
				t[i] = (floc-y)*math.Exp((floc-y)/scale) + y
			}
			return sum(t) - n*(floc+scale)
		}
	} else {
		m := mean(v)
		w := make([]float64, len(v))
		f = func(scale float64) float64 {
			// the mean of x weighted by exp(-x/scale), shifted for range
			mx := math.Inf(-1)
			for i, y := range v {
				t[i] = -y / scale
				mx = max(mx, t[i])
			}
			for i, y := range v {
				w[i] = math.Exp(t[i] - mx)
				t[i] = y * w[i]
			}
			return m - sum(t)/sum(w) - scale
		}
	}
	a, b := 0.5, 2.0
	for math.Signbit(f(a)) == math.Signbit(f(b)) && (a > 0 || b < math.Inf(1)) {
		a /= 2
		b *= 2
	}
	scale, _ := brentRoot(f, a, b) // NaN (rejected by Validate) without a root
	loc := floc
	if !haveLoc {
		loc = locFromScale(scale)
	}
	d := Gumbel{Loc: loc, Scale: scale}
	return d, d.Validate()
}

// logSumExp returns log Σ exp(a_i), shifted by the maximum.
func logSumExp(a []float64) float64 {
	mx := math.Inf(-1)
	for _, x := range a {
		mx = max(mx, x)
	}
	s := make([]float64, len(a))
	for i, x := range a {
		s[i] = math.Exp(x - mx)
	}
	return mx + math.Log(sum(s))
}
