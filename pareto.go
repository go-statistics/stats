package stats

import (
	"fmt"
	"math"

	"github.com/go-ndarray/ndarray"
	"github.com/go-ndarray/ndarray/random"
	"github.com/go-statistics/stats/internal/dd"
)

//stats:continuous Pareto pareto b=B

// Pareto is the Pareto distribution with shape B: density B/x^(B+1) on
// [1, ∞) in standard form: scipy.stats.pareto(b, loc, scale).
//
// With t = B log x in double-double, SF is exp(-t) and CDF -expm1(-t),
// rounded once (SciPy's 1 - x^-b loses everything near x = 1), the
// quantiles exp(-log1p(-p)/B) and exp(-log(q)/B). Moments that do not
// exist are +Inf or NaN as in SciPy.
type Pareto struct {
	B, Loc, Scale float64
}

func (d Pareto) std() paretoStd { return paretoStd{b: d.B} }

type paretoStd struct{ b float64 }

func (s paretoStd) valid() bool               { return s.b > 0 && !math.IsInf(s.b, 1) }
func (s paretoStd) support() (lo, hi float64) { return 1, math.Inf(1) }

// bp1 is b + 1 exactly.
func (s paretoStd) bp1() dd.DD {
	h, l := dd.TwoSum(s.b, 1)
	return dd.DD{Hi: h, Lo: l}
}

func (s paretoStd) t(x float64) dd.DD { return dd.MulF(dd.Log(x), s.b) }

func (s paretoStd) pdf(x float64) float64 {
	return expScaledR(dd.Neg(dd.Mul(s.bp1(), dd.Log(x))), dd.F(s.b))
}

func (s paretoStd) logpdf(x float64) float64 {
	return dd.Sub(dd.Log(s.b), dd.Mul(s.bp1(), dd.Log(x))).Float()
}

func (s paretoStd) cdf(x float64) float64    { return dd.Neg(expm1D(dd.Neg(s.t(x)))).Float() }
func (s paretoStd) sf(x float64) float64     { return expR(dd.Neg(s.t(x))) }
func (s paretoStd) logcdf(x float64) float64 { return log1mexpSmall(s.t(x)) }
func (s paretoStd) logsf(x float64) float64  { return -s.t(x).Float() }

func (s paretoStd) ppf(p float64) float64 { return expR(dd.DivF(log1pDD(dd.F(-p)), -s.b)) }
func (s paretoStd) isf(q float64) float64 { return expR(dd.DivF(dd.Log(q), -s.b)) }

func (s paretoStd) mean() float64 {
	if s.b > 1 {
		return s.b / (s.b - 1)
	}
	return math.Inf(1)
}

func (s paretoStd) variance() float64 {
	b := s.b
	if b > 2 {
		return b / (b - 2) / ((b - 1) * (b - 1))
	}
	return math.Inf(1)
}

func (s paretoStd) skew() float64 {
	b := s.b
	if b > 3 {
		return 2 * (b + 1) * math.Sqrt(b-2) / ((b - 3) * math.Sqrt(b))
	}
	return math.NaN()
}

func (s paretoStd) kurtosis() float64 {
	b := s.b
	if b > 4 {
		return 6 * (((b+1)*b-6)*b - 2) / (((b - 7) * b + 12) * b)
	}
	return math.NaN()
}

// median is 2^(1/b).
func (s paretoStd) median() float64 { return expR(dd.DivF(ln2DD, s.b)) }

// entropy is 1 + 1/b - log b.
func (s paretoStd) entropy() float64 {
	return dd.Sub(dd.AddF(dd.Div(dd.F(1), dd.F(s.b)), 1), dd.Log(s.b)).Float()
}

// munp is b/(b-n), +Inf where the moment does not exist.
func (s paretoStd) munp(n int) float64 {
	fn := float64(n)
	if s.b > fn {
		return s.b / (s.b - fn)
	}
	return math.Inf(1)
}

// rvs is SciPy's pow(1 - U, -1/b) for uniform draws U, the power
// correctly rounded.
func (s paretoStd) rvs(g *random.Generator, size []int) (*ndarray.Array, error) {
	u, err := g.Uniform(0, 1, size...)
	if err != nil {
		return nil, err
	}
	e := -1 / s.b
	v, _ := ndarray.Data[float64](u)
	for i, p := range v {
		v[i] = crPow(1-p, e)
	}
	return u, nil
}

// FitPareto returns the maximum-likelihood Pareto distribution for x:
// scipy.stats.pareto.fit(x, fb=, floc=, fscale=). With Loc or Scale fixed
// the estimates have closed forms (Loc + Scale = min(x) and B = n/Σ
// log((x-Loc)/Scale)), reproduced to the bit; with both free, Scale
// solves SciPy's Lagrange condition, bracketed as SciPy brackets it but
// found to full precision.
func FitPareto[D Data](x D, o *FitOptions) (Pareto, error) {
	v := floats(x)
	fx, err := o.fixed("Pareto", "B", "Loc", "Scale")
	if err != nil {
		return Pareto{}, err
	}
	if err := checkFitData(v); err != nil {
		return Pareto{}, err
	}
	lo, _ := minMax(v)
	fb, haveB := fx["B"]
	floc, haveLoc := fx["Loc"]
	fscale, haveScale := fx["Scale"]
	if haveLoc && lo-floc < fscale { // fscale is 0 when not fixed
		return Pareto{}, fmt.Errorf("%w: Pareto fit with Loc=%v, Scale=%v needs every value at or above Loc+Scale", ErrData, floc, fscale)
	}
	n := float64(len(v))
	t := make([]float64, len(v))
	shapeOf := func(scale, loc float64) float64 {
		if haveB {
			return fb
		}
		for i, y := range v {
			t[i] = crLog((y - loc) / scale)
		}
		return n / sum(t)
	}
	var loc, scale float64
	switch {
	case !haveLoc && !haveScale:
		f := func(scale float64) float64 {
			l := lo - scale
			b := shapeOf(scale, l)
			for i, y := range v {
				t[i] = 1 / (y - l)
			}
			return (b+1)*sum(t) - n*b/scale
		}
		a, c := 0.5, 2.0
		for math.Signbit(f(a)) == math.Signbit(f(c)) && (a > 0 || c < math.Inf(1)) {
			a /= 2
			c *= 2
		}
		scale, _ = brentRoot(f, a, c) // NaN (rejected by Validate) without a root
		loc = lo - scale
		b := shapeOf(scale, loc)
		if !(scale+loc < lo) {
			scale = math.Nextafter(lo-loc, 0)
		}
		d := Pareto{B: b, Loc: loc, Scale: scale}
		return d, d.Validate()
	case !haveLoc:
		loc = lo - fscale
	default:
		loc = floc
	}
	scale = fscale
	if !haveScale {
		scale = lo - loc
	}
	d := Pareto{B: shapeOf(scale, loc), Loc: loc, Scale: scale}
	return d, d.Validate()
}
