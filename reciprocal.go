package stats

import (
	"fmt"
	"math"

	"github.com/go-ndarray/ndarray"
	"github.com/go-ndarray/ndarray/random"
	"github.com/go-statistics/stats/internal/dd"
)

//stats:continuous Reciprocal reciprocal a=A b=B

// Reciprocal is the reciprocal (log-uniform) distribution on [A, B],
// 0 < A < B: density 1/(x log(B/A)), so that log X is uniform on
// [log A, log B]: scipy.stats.reciprocal(a, b, loc, scale), also known as
// scipy.stats.loguniform (see LogUniform).
//
// log(B/A) and log(x/A) are carried in double-double, through log1p of
// the exact difference when the ratio is near 1, so that CDF and SF keep
// full relative accuracy at both ends of the support (SciPy computes SF
// as 1 - CDF and loses everything near B). PPF is A·exp(p·log(B/A)) and
// ISF B·exp(-q·log(B/A)), each rounded once. The moments are formed in
// double-double from the raw moments (B^n - A^n)/(n log(B/A)).
type Reciprocal struct {
	A, B, Loc, Scale float64
}

// LogUniform is the log-uniform distribution, scipy.stats.loguniform: the
// same distribution as Reciprocal, under the name SciPy also gives it.
type LogUniform = Reciprocal

func (d Reciprocal) std() reciprocalStd {
	s := reciprocalStd{a: d.A, b: d.B}
	if s.valid() {
		s.l = logRatio(s.b, s.a)
	}
	return s
}

// reciprocalStd carries l = log(b/a) in double-double.
type reciprocalStd struct {
	a, b float64
	l    dd.DD
}

func (s reciprocalStd) valid() bool {
	return s.a > 0 && s.b > s.a && !math.IsInf(s.b, 1)
}
func (s reciprocalStd) support() (lo, hi float64) { return s.a, s.b }

func (s reciprocalStd) pdf(x float64) float64 {
	return dd.Div(dd.F(1), dd.MulF(s.l, x)).Float()
}

func (s reciprocalStd) logpdf(x float64) float64 {
	return dd.Neg(dd.Add(dd.Log(x), logDD(s.l))).Float()
}

func (s reciprocalStd) cdfD(x float64) dd.DD { return dd.Div(logRatio(x, s.a), s.l) }
func (s reciprocalStd) sfD(x float64) dd.DD  { return dd.Div(logRatio(s.b, x), s.l) }

func (s reciprocalStd) cdf(x float64) float64    { return s.cdfD(x).Float() }
func (s reciprocalStd) sf(x float64) float64     { return s.sfD(x).Float() }
func (s reciprocalStd) logcdf(x float64) float64 { return logPair(s.cdfD(x), s.sfD(x)) }
func (s reciprocalStd) logsf(x float64) float64  { return logPair(s.sfD(x), s.cdfD(x)) }

func (s reciprocalStd) ppf(p float64) float64 { return expScaledR(dd.MulF(s.l, p), dd.F(s.a)) }
func (s reciprocalStd) isf(q float64) float64 { return expScaledR(dd.MulF(s.l, -q), dd.F(s.b)) }

// ratioMoment is E[(Y/a)^n] = expm1(n l)/(n l), n >= 1, in double-double.
func (s reciprocalStd) ratioMoment(n int) dd.DD {
	t := dd.MulF(s.l, float64(n))
	return dd.Div(expm1D(t), t)
}

// central returns the variance (of Y/a), skewness and excess kurtosis,
// from the raw moments in double-double.
func (s reciprocalStd) central() (v, g1, g2 float64) {
	m1, m2, m3, m4 := s.ratioMoment(1), s.ratioMoment(2), s.ratioMoment(3), s.ratioMoment(4)
	m1s := dd.Mul(m1, m1)
	mu2 := dd.Sub(m2, m1s)
	// mu3 = m3 - 3 m1 m2 + 2 m1³
	mu3 := dd.Add(dd.Sub(m3, dd.MulF(dd.Mul(m1, m2), 3)), dd.MulF(dd.Mul(m1s, m1), 2))
	// mu4 = m4 - 4 m1 m3 + 6 m1² m2 - 3 m1⁴
	mu4 := dd.Sub(dd.Add(dd.Sub(m4, dd.MulF(dd.Mul(m1, m3), 4)), dd.MulF(dd.Mul(m1s, m2), 6)), dd.MulF(dd.Mul(m1s, m1s), 3))
	v = mu2.Float()
	g1 = dd.Div(mu3, dd.Mul(mu2, sqrtD(mu2))).Float()
	g2 = dd.AddF(dd.Div(mu4, dd.Mul(mu2, mu2)), -3).Float()
	return v, g1, g2
}

func (s reciprocalStd) mean() float64 { return dd.MulF(s.ratioMoment(1), s.a).Float() }
func (s reciprocalStd) variance() float64 {
	v, _, _ := s.central()
	return v * s.a * s.a
}
func (s reciprocalStd) skew() float64 {
	_, g1, _ := s.central()
	return g1
}
func (s reciprocalStd) kurtosis() float64 {
	_, _, g2 := s.central()
	return g2
}
func (s reciprocalStd) median() float64 { return s.ppf(0.5) }

// entropy is log a + l/2 + log l.
func (s reciprocalStd) entropy() float64 {
	return dd.Add(dd.Add(dd.Log(s.a), dd.MulF(s.l, 0.5)), logDD(s.l)).Float()
}

// munp is (b^n - a^n)/(n l) = a^n expm1(n l)/(n l).
func (s reciprocalStd) munp(n int) float64 {
	an := expD(dd.MulF(dd.Log(s.a), float64(n)))
	return dd.Mul(an, s.ratioMoment(n)).Float()
}

// rvs is SciPy's exp(log a + U·(log b - log a)) for uniform draws U, with
// the logarithms and the exponential correctly rounded.
func (s reciprocalStd) rvs(g *random.Generator, size []int) (*ndarray.Array, error) {
	u, err := g.Uniform(0, 1, size...)
	if err != nil {
		return nil, err
	}
	la, lb := crLog(s.a), crLog(s.b)
	w := lb - la
	v, _ := ndarray.Data[float64](u)
	for i, q := range v {
		v[i] = crExp(la + float64(q*w))
	}
	return u, nil
}

// FitReciprocal returns the maximum-likelihood reciprocal distribution for
// x: scipy.stats.reciprocal.fit(x, fa=, fb=, floc=, fscale=). As in SciPy
// the family is over-parameterised and Scale is held at 1 unless fixed
// otherwise in o.Fixed. With Loc fixed the likelihood is maximal for the
// narrowest support holding the data, A and B the extremes of
// (x - Loc)/Scale, which SciPy's numerical search only approaches; with
// Loc free (the likelihood is then unbounded as Loc nears the minimum) it
// is maximised numerically from SciPy's starting point.
func FitReciprocal[D Data](x D, o *FitOptions) (Reciprocal, error) {
	v := floats(x)
	fx, err := o.fixed("Reciprocal", "A", "B", "Loc", "Scale")
	if err != nil {
		return Reciprocal{}, err
	}
	if err := checkFitData(v); err != nil {
		return Reciprocal{}, err
	}
	if _, ok := fx["Scale"]; !ok {
		fx["Scale"] = 1
		if len(fx) == 4 {
			return Reciprocal{}, errAllFixed
		}
	}
	scale := fx["Scale"]
	if loc, ok := fx["Loc"]; ok {
		return fitReciprocalFixedLoc(v, fx, loc, scale)
	}
	// With Loc free, A and B are profiled out (the extremes of
	// (x - Loc)/Scale, or their fixed values) and the likelihood is
	// maximised over w = min(x) - Loc > 0, from SciPy's starting Loc (the
	// method of moments moved to cover the data). The likelihood is
	// unbounded only for w below the float64 resolution of min(x), so the
	// interior optimum is what a numerical search finds.
	lo, hi := minMax(v)
	loc0, _ := fitLocScaleSupport(v, Reciprocal{A: lo, B: hi}.std())
	w0 := lo - loc0
	if !(w0 > 0) {
		w0 = hi - lo
	}
	prof := func(loc float64) (Reciprocal, float64) {
		d, err := fitReciprocalFixedLoc(v, fx, loc, scale)
		if err != nil {
			return d, math.Inf(-1)
		}
		return d, contLogLik(v, d.std(), loc, scale)
	}
	p := mleProblem{
		names:    []string{"W"},
		positive: []bool{true},
		start:    []float64{w0},
		loglik: func(th []float64) float64 {
			_, ll := prof(lo - th[0])
			return ll
		},
	}
	th, err := fitMLE(p, nil)
	if err != nil {
		return Reciprocal{}, err
	}
	d, _ := prof(lo - th[0])
	return d, d.Validate()
}

func fitReciprocalFixedLoc(v []float64, fx map[string]float64, loc, scale float64) (Reciprocal, error) {
	z := make([]float64, len(v))
	for i, t := range v {
		z[i] = (t - loc) / scale
	}
	lo, hi := minMax(z)
	a, haveA := fx["A"]
	b, haveB := fx["B"]
	if !haveA {
		a = lo
	}
	if !haveB {
		b = hi
	}
	if a > lo || b < hi {
		return Reciprocal{}, fmt.Errorf("%w: reciprocal fit: the data fall outside [Loc + A·Scale, Loc + B·Scale]", ErrData)
	}
	d := Reciprocal{A: a, B: b, Loc: loc, Scale: scale}
	return d, d.Validate()
}
