package stats

import (
	"math"

	"github.com/go-ndarray/ndarray"
	"github.com/go-ndarray/ndarray/random"
	"github.com/go-statistics/stats/internal/dd"
)

//stats:continuous Cauchy cauchy

// Cauchy is the Cauchy (Lorentz) distribution with median Loc and
// half-width Scale: density 1/(π(1+x²)) in standard form:
// scipy.stats.cauchy(loc, scale).
//
// The tail probabilities are atan(1/|x|)/π beyond 1 and 1/2 - atan(|x|)/π
// within, with a double-double arctangent (atanD), so that CDF and SF keep
// full relative accuracy in both tails; PPF and ISF are -cot(πp) or
// tan(π(p - 1/2)) from a double-double sine and cosine of an argument
// never above π/4, accurate where SciPy's tan(π(p - 1/2)) loses the low
// bits of p.
type Cauchy struct {
	Loc, Scale float64
}

func (d Cauchy) std() cauchyStd { return cauchyStd{} }

type cauchyStd struct{}

func (cauchyStd) valid() bool             { return true }
func (cauchyStd) support() (a, b float64) { return math.Inf(-1), math.Inf(1) }

func (cauchyStd) pdf(x float64) float64 {
	if math.Abs(x) > 1e150 {
		return dd.DivF(dd.DivF(invPiDD, x), x).Float()
	}
	return dd.Div(invPiDD, dd.AddF(dd.MulFF(x, x), 1)).Float()
}

func (cauchyStd) logpdf(x float64) float64 {
	ax := math.Abs(x)
	if ax <= 1 {
		return dd.Neg(dd.Add(logPiDD, log1pDD(dd.MulFF(x, x)))).Float()
	}
	u := dd.Div(dd.F(1), dd.F(ax))
	l := dd.Add(dd.MulF(dd.Log(ax), 2), log1pDD(dd.Mul(u, u)))
	return dd.Neg(dd.Add(logPiDD, l)).Float()
}

// cauchyTail returns P(X > t) for t >= 0 as a double-double.
func cauchyTail(t float64) dd.DD {
	if t >= 1 {
		return dd.Mul(atanD(dd.Div(dd.F(1), dd.F(t))), invPiDD)
	}
	return dd.Sub(dd.F(0.5), dd.Mul(atanD(dd.F(t)), invPiDD))
}

// cauchyCDF returns P(X <= x) as a double-double.
func cauchyCDF(x float64) dd.DD {
	if x < 0 {
		return cauchyTail(-x)
	}
	return dd.Sub(dd.F(1), cauchyTail(x))
}

func (cauchyStd) cdf(x float64) float64    { return cauchyCDF(x).Float() }
func (cauchyStd) sf(x float64) float64     { return cauchyCDF(-x).Float() }
func (cauchyStd) logcdf(x float64) float64 { return logPair(cauchyCDF(x), cauchyCDF(-x)) }
func (cauchyStd) logsf(x float64) float64  { return logPair(cauchyCDF(-x), cauchyCDF(x)) }

// cauchyLower returns the quantile of 0 < p <= 1/2: -cot(πp) up to 1/4,
// tan(π(p - 1/2)) beyond (p - 1/2 exact), and -1/(πp) below 2^-500 where
// the next term of the cotangent, πp/3, is far below the rounding.
func cauchyLower(p float64) float64 {
	switch {
	case p < 0x1p-500:
		return -dd.DivF(invPiDD, p).Float()
	case p <= 0.25:
		s, c := sincosSmall(dd.MulF(piDD, p))
		return -dd.Div(c, s).Float()
	}
	s, c := sincosSmall(dd.MulF(piDD, 0.5-p))
	return -dd.Div(s, c).Float()
}

func (cauchyStd) ppf(p float64) float64 {
	if p > 0.5 {
		return -cauchyLower(1 - p)
	}
	return cauchyLower(p)
}

func (cauchyStd) isf(q float64) float64 {
	if q > 0.5 {
		return cauchyLower(1 - q)
	}
	return -cauchyLower(q)
}

func (cauchyStd) mean() float64     { return math.NaN() }
func (cauchyStd) variance() float64 { return math.NaN() }
func (cauchyStd) skew() float64     { return math.NaN() }
func (cauchyStd) kurtosis() float64 { return math.NaN() }
func (cauchyStd) median() float64   { return 0 }

// entropy is log(4π).
func (cauchyStd) entropy() float64 { return 2.5310242469692907 }

// munp: no moment exists.
func (cauchyStd) munp(int) float64 { return math.NaN() }

// rvs is SciPy's: the quantile of uniform draws, computed as SciPy's
// quantile (Boost's) computes it, -1/tan(π·P) with P = U, or U - 1 above
// 1/2, the product rounded and the tangent correctly rounded.
func (cauchyStd) rvs(g *random.Generator, size []int) (*ndarray.Array, error) {
	u, err := g.Uniform(0, 1, size...)
	if err != nil {
		return nil, err
	}
	v, _ := ndarray.Data[float64](u)
	for i, p := range v {
		v[i] = cauchyDraw(p)
	}
	return u, nil
}

// cauchyDraw is Boost's Cauchy quantile of p as SciPy's draws evaluate it.
func cauchyDraw(p float64) float64 {
	if p == 0.5 {
		return 0
	}
	if p > 0.5 {
		p--
	}
	return -1 / crTan(math.Pi*p)
}

// FitCauchy returns the maximum-likelihood Cauchy distribution for x:
// scipy.stats.cauchy.fit(x, floc=, fscale=), maximised numerically from
// SciPy's starting point (the median and half the interquartile range), to
// a tighter optimum than SciPy's.
func FitCauchy[D Data](x D, o *FitOptions) (Cauchy, error) {
	v := floats(x)
	fx, err := o.fixed("Cauchy", "Loc", "Scale")
	if err != nil {
		return Cauchy{}, err
	}
	if err := checkFitData(v); err != nil {
		return Cauchy{}, err
	}
	s := sortedCopy(v)
	p25, p50, p75 := percentile(s, 25), percentile(s, 50), percentile(s, 75)
	scale0 := (p75 - p25) / 2
	if !(scale0 > 0) {
		scale0 = 1
	}
	p := mleProblem{
		names:    []string{"Loc", "Scale"},
		positive: []bool{false, true},
		start:    []float64{p50, scale0},
		loglik: func(th []float64) float64 {
			return contLogLik(v, cauchyStd{}, th[0], th[1])
		},
	}
	th, _ := fitMLE(p, fx) // the density is positive everywhere: the start has a likelihood
	d := Cauchy{Loc: th[0], Scale: th[1]}
	return d, d.Validate()
}

// percentile is np.percentile(s, q) for sorted s (linear interpolation,
// NumPy's lerp).
func percentile(s []float64, q float64) float64 {
	h := q / 100 * float64(len(s)-1)
	i := int(math.Floor(h))
	t := h - float64(i)
	if i+1 >= len(s) {
		return s[len(s)-1]
	}
	a, b := s[i], s[i+1]
	d := b - a
	if t >= 0.5 {
		return b - float64(d*(1-t))
	}
	return a + float64(d*t)
}
