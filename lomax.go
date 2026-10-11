package stats

import (
	"math"

	"github.com/go-ndarray/ndarray"
	"github.com/go-ndarray/ndarray/random"
	"github.com/go-statistics/stats/internal/dd"
)

//stats:continuous Lomax lomax c=C

// Lomax is the Lomax (Pareto of the second kind) distribution with shape
// C: density C/(1+x)^(C+1) on [0, ∞) in standard form:
// scipy.stats.lomax(c, loc, scale), Pareto(C) shifted by -1.
//
// With t = C log1p(x) in double-double, SF is exp(-t), CDF -expm1(-t),
// LogCDF log(1 - exp(-t)) without cancellation, PPF expm1(-log1p(-p)/C)
// and ISF expm1(-log(q)/C), each rounded once. Moments that do not exist
// are +Inf or NaN as in SciPy.
type Lomax struct {
	C, Loc, Scale float64
}

func (d Lomax) std() lomaxStd { return lomaxStd{c: d.C} }

type lomaxStd struct{ c float64 }

func (s lomaxStd) valid() bool               { return s.c > 0 && !math.IsInf(s.c, 1) }
func (s lomaxStd) support() (lo, hi float64) { return 0, math.Inf(1) }

// cp1 is c + 1 exactly.
func (s lomaxStd) cp1() dd.DD {
	h, l := dd.TwoSum(s.c, 1)
	return dd.DD{Hi: h, Lo: l}
}

func (s lomaxStd) t(x float64) dd.DD { return dd.MulF(log1pDD(dd.F(x)), s.c) }

func (s lomaxStd) pdf(x float64) float64 {
	return expScaledR(dd.Neg(dd.Mul(s.cp1(), log1pDD(dd.F(x)))), dd.F(s.c))
}

func (s lomaxStd) logpdf(x float64) float64 {
	return dd.Sub(dd.Log(s.c), dd.Mul(s.cp1(), log1pDD(dd.F(x)))).Float()
}

func (s lomaxStd) cdf(x float64) float64    { return dd.Neg(expm1D(dd.Neg(s.t(x)))).Float() }
func (s lomaxStd) sf(x float64) float64     { return expR(dd.Neg(s.t(x))) }
func (s lomaxStd) logcdf(x float64) float64 { return log1mexpSmall(s.t(x)) }
func (s lomaxStd) logsf(x float64) float64  { return -s.t(x).Float() }

func (s lomaxStd) ppf(p float64) float64 {
	return expm1D(dd.DivF(log1pDD(dd.F(-p)), -s.c)).Float()
}
func (s lomaxStd) isf(q float64) float64 { return expm1D(dd.DivF(dd.Log(q), -s.c)).Float() }

func (s lomaxStd) mean() float64 {
	if s.c > 1 {
		return 1 / (s.c - 1)
	}
	return math.Inf(1)
}

func (s lomaxStd) variance() float64 { return paretoStd{b: s.c}.variance() }
func (s lomaxStd) skew() float64     { return paretoStd{b: s.c}.skew() }
func (s lomaxStd) kurtosis() float64 { return paretoStd{b: s.c}.kurtosis() }

// median is 2^(1/c) - 1.
func (s lomaxStd) median() float64 { return expm1D(dd.DivF(ln2DD, s.c)).Float() }

// entropy is 1 + 1/c - log c.
func (s lomaxStd) entropy() float64 { return paretoStd{b: s.c}.entropy() }

// munp is n!/((c-1)(c-2)...(c-n)) for c > n, +Inf otherwise.
func (s lomaxStd) munp(n int) float64 {
	if !(s.c > float64(n)) {
		return math.Inf(1)
	}
	r := 1.0
	for k := 1; k <= n; k++ {
		r *= float64(k) / (s.c - float64(k))
	}
	return r
}

// rvs is SciPy's expm1(-log1p(-U)/c) for uniform draws U, with SciPy's
// own log1p and expm1.
func (s lomaxStd) rvs(g *random.Generator, size []int) (*ndarray.Array, error) {
	u, err := g.Uniform(0, 1, size...)
	if err != nil {
		return nil, err
	}
	v, _ := ndarray.Data[float64](u)
	for i, p := range v {
		v[i] = cephesExpm1(-cephesLog1p(-p) / s.c)
	}
	return u, nil
}

// FitLomax returns the maximum-likelihood Lomax distribution for x:
// scipy.stats.lomax.fit(x, fc=, floc=, fscale=), maximised numerically
// from SciPy's starting point (C = 1, Loc and Scale by the method of
// moments moved to cover the data), to a tighter optimum than SciPy's.
func FitLomax[D Data](x D, o *FitOptions) (Lomax, error) {
	v := floats(x)
	fx, err := o.fixed("Lomax", "C", "Loc", "Scale")
	if err != nil {
		return Lomax{}, err
	}
	if err := checkFitData(v); err != nil {
		return Lomax{}, err
	}
	// SciPy's start covers the data: it has a likelihood
	sh, loc, scale, _ := fitContinuous(v, fx, []string{"C"}, []bool{true}, []float64{1},
		func(s []float64) contStd { return lomaxStd{c: s[0]} })
	d := Lomax{C: sh[0], Loc: loc, Scale: scale}
	return d, d.Validate()
}
