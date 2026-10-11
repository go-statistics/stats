package stats

import (
	"fmt"
	"math"

	"github.com/go-ndarray/ndarray"
	"github.com/go-ndarray/ndarray/random"
	"github.com/go-numerics/special"
	"github.com/go-statistics/stats/internal/dd"
)

//stats:continuous Gamma gamma a=A

// Gamma is the gamma distribution with shape A: density
// x^(A-1) e^(-x) / Γ(A) in standard form: scipy.stats.gamma(a, loc,
// scale).
//
// CDF and SF are special.GammaP and special.GammaQ, PPF and ISF their
// inverses. The density carries its exponent in double-double and, for
// A >= 30, takes Stirling's factors out analytically (see gammaExponent),
// where SciPy's exp(xlogy(a-1, x) - x - gammaln(a)) loses about log2(A)
// bits to cancellation.
type Gamma struct {
	A, Loc, Scale float64
}

func (d Gamma) std() gammaStd { return gammaStd{a: d.A} }

type gammaStd struct{ a float64 }

func (s gammaStd) valid() bool               { return s.a > 0 && !math.IsInf(s.a, 1) }
func (s gammaStd) support() (lo, hi float64) { return 0, math.Inf(1) }

func (s gammaStd) pdf(x float64) float64 {
	if x == 0 {
		return gammaAtZero(s.a, 1)
	}
	return gammaDensity(s.a, dd.F(x), dd.F(1))
}

// gammaAtZero is the density x^(a-1)/Γ(a) at x = 0: +Inf below a = 1, 0
// above, and c at a = 1 (c the density's factor).
func gammaAtZero(a, c float64) float64 {
	switch {
	case a < 1:
		return math.Inf(1)
	case a > 1:
		return 0
	}
	return c
}

func (s gammaStd) logpdf(x float64) float64 {
	if x == 0 {
		return math.Log(gammaAtZero(s.a, 1))
	}
	return gammaLogDensity(s.a, dd.F(x)).Float()
}

func (s gammaStd) cdf(x float64) float64 { return special.GammaP(s.a, x) }
func (s gammaStd) sf(x float64) float64  { return special.GammaQ(s.a, x) }

func (s gammaStd) logcdf(x float64) float64 {
	return logPQ(special.GammaP(s.a, x), special.GammaQ(s.a, x))
}
func (s gammaStd) logsf(x float64) float64 {
	return logPQ(special.GammaQ(s.a, x), special.GammaP(s.a, x))
}

func (s gammaStd) ppf(p float64) float64 { return special.GammaPInv(s.a, p) }
func (s gammaStd) isf(q float64) float64 { return special.GammaQInv(s.a, q) }

func (s gammaStd) mean() float64     { return s.a }
func (s gammaStd) variance() float64 { return s.a }
func (s gammaStd) skew() float64     { return 2 / math.Sqrt(s.a) }
func (s gammaStd) kurtosis() float64 { return 6 / s.a }
func (s gammaStd) median() float64   { return special.GammaPInv(s.a, 0.5) }

// entropy is log Γ(a) - (a-1)ψ(a) + a (gammaEntropyPart).
func (s gammaStd) entropy() float64 { return gammaEntropyPart(s.a) }

// munp is the rising factorial a(a+1)...(a+n-1) (SciPy's poch(a, n)).
func (s gammaStd) munp(n int) float64 { return poch(s.a, n) }

func (s gammaStd) rvs(g *random.Generator, size []int) (*ndarray.Array, error) {
	return g.StandardGamma(s.a, size...)
}

// poch is the rising factorial a(a+1)...(a+n-1), n >= 0.
func poch(a float64, n int) float64 {
	r := 1.0
	for k := 0; k < n; k++ {
		r *= a + float64(k)
	}
	return r
}

// logPQ returns log p for a probability p whose complement is q, through
// log1p(-q) when p is close to 1 (where log p would round q away).
func logPQ(p, q float64) float64 {
	if p > 0.5 {
		return math.Log1p(-q)
	}
	return math.Log(p)
}

// FitGamma returns the maximum-likelihood gamma distribution for x:
// scipy.stats.gamma.fit(x, fa=, floc=, fscale=). With Loc fixed it solves
// the likelihood equations as SciPy does (log a - ψ(a) = log x̄ - mean log x,
// then Scale = x̄/A); otherwise it maximises the likelihood numerically from
// SciPy's starting point, to a tighter optimum than SciPy's fmin reaches.
func FitGamma[D Data](x D, o *FitOptions) (Gamma, error) {
	v := floats(x)
	fx, err := o.fixed("Gamma", "A", "Loc", "Scale")
	if err != nil {
		return Gamma{}, err
	}
	if err := checkFitData(v); err != nil {
		return Gamma{}, err
	}
	if loc, ok := fx["Loc"]; ok {
		return fitGammaFixedLoc(v, loc, fx)
	}
	sk := skewOf(v)
	a0 := 4 / (1e-8 + sk*sk)
	sh, loc, scale, err := fitContinuous(v, fx, []string{"A"}, []bool{true}, []float64{a0},
		func(s []float64) contStd { return gammaStd{a: s[0]} })
	if err != nil {
		return Gamma{}, err
	}
	d := Gamma{A: sh[0], Loc: loc, Scale: scale}
	return d, d.Validate()
}

func fitGammaFixedLoc(v []float64, loc float64, fx map[string]float64) (Gamma, error) {
	y := make([]float64, len(v))
	for i, t := range v {
		if !(t > loc) {
			return Gamma{}, fmt.Errorf("%w: gamma fit with Loc=%v needs every value above it", ErrData, loc)
		}
		y[i] = t - loc
	}
	xbar := mean(y)
	logs := make([]float64, len(y))
	for i, t := range y {
		logs[i] = math.Log(t)
	}
	mlog := mean(logs)
	a, haveA := fx["A"]
	scale, haveScale := fx["Scale"]
	switch {
	case haveScale:
		a = digammaInv(mlog - math.Log(scale))
	case !haveA:
		s := math.Log(xbar) - mlog
		a = solveLogMinusDigamma(s)
		scale = xbar / a
	default:
		scale = xbar / a
	}
	d := Gamma{A: a, Loc: loc, Scale: scale}
	return d, d.Validate()
}

// solveLogMinusDigamma solves log a - ψ(a) = s > 0 for a (the gamma shape
// MLE), by Newton's method on log a from the Minka/Thom approximation,
// polished to full precision.
func solveLogMinusDigamma(s float64) float64 {
	a := (3 - s + math.Sqrt((s-3)*(s-3)+24*s)) / (12 * s)
	for i, done := 0, false; i < 100 && !done; i++ {
		f := logMinusDigamma(a) - s
		fp := 1/a - special.Polygamma(1, a)
		na := math.Max(a-f/fp, a/8)
		done = math.Abs(na-a) <= 4e-16*a
		a = na
	}
	return a
}

// digammaInv solves ψ(a) = y for a > 0 (Newton from Minka's starting
// point, as SciPy's _digammainv).
func digammaInv(y float64) float64 {
	a := math.Exp(y) + 0.5
	if y < -2.22 {
		a = -1 / (y - special.Digamma(1))
	}
	for i, done := 0, false; i < 100 && !done; i++ {
		na := math.Max(a-(special.Digamma(a)-y)/special.Polygamma(1, a), a/8)
		done = math.Abs(na-a) <= 4e-16*a
		a = na
	}
	return a
}

// skewOf is the biased sample skewness (scipy.stats.skew(x), bias=True).
func skewOf(v []float64) float64 {
	m := mean(v)
	d2 := make([]float64, len(v))
	d3 := make([]float64, len(v))
	for i, t := range v {
		e := t - m
		d2[i] = float64(e * e)
		d3[i] = float64(d2[i] * e)
	}
	m2, m3 := mean(d2), mean(d3)
	return m3 / math.Pow(m2, 1.5)
}

// igamma returns P(a, x) (or Q(a, x) when upper) at the double-double x:
// the incomplete function at x.Hi plus the first-order correction
// density·x.Lo (the gamma density being dP/dx).
func igamma(a float64, x dd.DD, upper bool) float64 {
	if upper {
		q := special.GammaQ(a, x.Hi)
		if x.Lo != 0 {
			q -= gammaDensity(a, dd.F(x.Hi), dd.F(x.Lo))
		}
		return q
	}
	p := special.GammaP(a, x.Hi)
	if x.Lo != 0 {
		p += gammaDensity(a, dd.F(x.Hi), dd.F(x.Lo))
	}
	return p
}

// logMinusDigamma is log a - ψ(a), computed as 1/(2a) + R(a) from a = 30
// (the difference cancels: it is about 1/(2a)).
func logMinusDigamma(a float64) float64 {
	if a < stirlingMin {
		return math.Log(a) - special.Digamma(a)
	}
	return 0.5/a + digammaTail(a)
}
