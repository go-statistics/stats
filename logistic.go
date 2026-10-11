package stats

import (
	"fmt"
	"math"

	"github.com/go-ndarray/ndarray"
	"github.com/go-ndarray/ndarray/random"
	"github.com/go-numerics/special"
	"github.com/go-statistics/stats/internal/dd"
)

//stats:continuous Logistic logistic

// Logistic is the logistic distribution with median Loc and scale Scale:
// density e^(-x)/(1+e^(-x))² in standard form: scipy.stats.logistic(loc,
// scale).
//
// With e = exp(-|x|) in double-double, the density is e/(1+e)², CDF and
// SF e/(1+e) and 1/(1+e), their logarithms -log1p(e) and -|x| - log1p(e),
// each rounded once (also where subnormal). PPF is the logit, through
// log1p((2p-1)/(1-p)) near p = 1/2 where log p - log(1-p) would cancel.
type Logistic struct {
	Loc, Scale float64
}

func (d Logistic) std() logisticStd { return logisticStd{} }

type logisticStd struct{}

func (logisticStd) valid() bool             { return true }
func (logisticStd) support() (a, b float64) { return math.Inf(-1), math.Inf(1) }

// onePlusExpNeg returns 1 + exp(-t) for t >= 0 as a double-double.
func onePlusExpNeg(t float64) dd.DD { return dd.AddF(expD(dd.F(-t)), 1) }

func (logisticStd) pdf(x float64) float64 {
	t := math.Abs(x)
	d := onePlusExpNeg(t)
	return expScaledR(dd.F(-t), dd.Div(dd.F(1), dd.Mul(d, d)))
}

func (logisticStd) logpdf(x float64) float64 {
	t := math.Abs(x)
	return dd.Sub(dd.F(-t), dd.MulF(log1pDD(expD(dd.F(-t))), 2)).Float()
}

// logisticCDF returns P(X <= x) = 1/(1 + exp(-x)), rounded once.
func logisticCDF(x float64) float64 {
	if x < 0 {
		return expScaledR(dd.F(x), dd.Div(dd.F(1), onePlusExpNeg(-x)))
	}
	return dd.Div(dd.F(1), onePlusExpNeg(x)).Float()
}

// logisticLogCDF returns log P(X <= x) = -log1p(exp(-x)).
func logisticLogCDF(x float64) float64 {
	if x < 0 {
		return dd.Sub(dd.F(x), log1pDD(expD(dd.F(x)))).Float()
	}
	return -log1pDD(expD(dd.F(-x))).Float()
}

func (logisticStd) cdf(x float64) float64    { return logisticCDF(x) }
func (logisticStd) sf(x float64) float64     { return logisticCDF(-x) }
func (logisticStd) logcdf(x float64) float64 { return logisticLogCDF(x) }
func (logisticStd) logsf(x float64) float64  { return logisticLogCDF(-x) }

// logit returns log(p/(1-p)) for 0 < p < 1.
func logit(p float64) float64 {
	if p >= 0.25 && p <= 0.75 {
		// 2p - 1 is exact here, 1 - p carried in double-double.
		h, l := dd.TwoSum(1, -p)
		return log1pDD(dd.Div(dd.F(2*p-1), dd.DD{Hi: h, Lo: l})).Float()
	}
	return dd.Sub(dd.Log(p), log1pDD(dd.F(-p))).Float()
}

func (logisticStd) ppf(p float64) float64 { return logit(p) }
func (logisticStd) isf(q float64) float64 { return -logit(q) }

func (logisticStd) mean() float64 { return 0 }

// variance is π²/3.
func (logisticStd) variance() float64 { return 3.289868133696453 }
func (logisticStd) skew() float64     { return 0 }
func (logisticStd) kurtosis() float64 { return 1.2 }
func (logisticStd) median() float64   { return 0 }
func (logisticStd) entropy() float64  { return 2 }

// munp is 0 for odd n and 2·n!·(1 - 2^(1-n))·ζ(n) for even n.
func (logisticStd) munp(n int) float64 {
	if n%2 == 1 {
		return 0
	}
	return 2 * factorial(n) * -math.Expm1(float64(1-n)*math.Ln2) * special.Zeta(float64(n))
}

func (logisticStd) rvs(g *random.Generator, size []int) (*ndarray.Array, error) {
	return g.Logistic(0, 1, size...)
}

// FitLogistic returns the maximum-likelihood logistic distribution for x:
// scipy.stats.logistic.fit(x, floc=, fscale=). Loc and Scale solve the
// likelihood equations Σ expit((x-Loc)/Scale) = n/2 and
// Σ c·tanh(c/2) = n, c = (x-Loc)/Scale, as in SciPy, here by nested
// bracketed searches to full precision (the first equation is monotone in
// Loc for each Scale; the second changes sign once in Scale along it).
func FitLogistic[D Data](x D, o *FitOptions) (Logistic, error) {
	v := floats(x)
	fx, err := o.fixed("Logistic", "Loc", "Scale")
	if err != nil {
		return Logistic{}, err
	}
	if err := checkFitData(v); err != nil {
		return Logistic{}, err
	}
	n := float64(len(v))
	c := make([]float64, len(v))
	dlLoc := func(loc, scale float64) float64 {
		for i, t := range v {
			c[i] = special.Expit((t - loc) / scale)
		}
		return sum(c) - n/2
	}
	dlScale := func(loc, scale float64) float64 {
		for i, t := range v {
			z := (t - loc) / scale
			c[i] = z * math.Tanh(z/2)
		}
		return sum(c) - n
	}
	lo, hi := minMax(v)
	locFor := func(scale float64) float64 {
		if f, ok := fx["Loc"]; ok {
			return f
		}
		r, _ := brentRoot(func(l float64) float64 { return dlLoc(l, scale) }, lo, hi)
		return r
	}
	scale, ok := fx["Scale"]
	if !ok {
		if lo == hi {
			return Logistic{}, fmt.Errorf("%w: logistic fit of constant data", ErrData)
		}
		s0 := math.Sqrt(variance(v, 0) / (math.Pi * math.Pi / 3))
		f := func(s float64) float64 { return dlScale(locFor(s), s) }
		a, b := s0/2, s0*2
		for f(a) < 0 {
			a /= 2
		}
		for f(b) > 0 {
			b *= 2
		}
		scale, _ = brentRoot(f, a, b)
	}
	d := Logistic{Loc: locFor(scale), Scale: scale}
	return d, d.Validate()
}
