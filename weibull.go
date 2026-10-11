package stats

import (
	"math"

	"github.com/go-ndarray/ndarray"
	"github.com/go-ndarray/ndarray/random"
	"github.com/go-numerics/optimize"
	"github.com/go-numerics/special"
	"github.com/go-statistics/stats/internal/dd"
)

//stats:continuous Weibull weibull_min c=C

// Weibull is the Weibull (minimum) distribution with shape C: density
// C x^(C-1) exp(-x^C) on [0, ∞) in standard form:
// scipy.stats.weibull_min(c, loc, scale).
//
// x^C is carried in double-double as exp(C log x): SF is exp(-x^C) and CDF
// -expm1(-x^C) rounded once (SciPy's exp(-pow(x, c)) has the relative
// error of pow times x^c, up to hundreds of ulps in the tail), LogCDF
// log(1 - exp(-x^C)) without cancellation, and the quantiles
// exp(log(-log1p(-p))/C) and exp(log(-log q)/C). The variance, skewness
// and kurtosis come from the differences log Γ(1+k/C) - k log Γ(1+1/C),
// summed as a series for large C where SciPy's differences of gamma
// functions cancel.
type Weibull struct {
	C, Loc, Scale float64
}

func (d Weibull) std() weibullStd { return weibullStd{c: d.C} }

type weibullStd struct{ c float64 }

func (s weibullStd) valid() bool               { return s.c > 0 && !math.IsInf(s.c, 1) }
func (s weibullStd) support() (lo, hi float64) { return 0, math.Inf(1) }

// xc returns x^c as a double-double and log x.
func (s weibullStd) xc(x float64) (xc, l dd.DD) {
	l = dd.Log(x)
	return expD(dd.MulF(l, s.c)), l
}

// atZero is the density at 0: +Inf below c = 1, 1 at c = 1, 0 above.
func (s weibullStd) atZero() float64 {
	switch {
	case s.c < 1:
		return math.Inf(1)
	case s.c > 1:
		return 0
	}
	return 1
}

// cm1 is c - 1 exactly.
func (s weibullStd) cm1() dd.DD {
	h, l := dd.TwoSum(s.c, -1)
	return dd.DD{Hi: h, Lo: l}
}

func (s weibullStd) pdf(x float64) float64 {
	if x == 0 {
		return s.atZero()
	}
	xc, l := s.xc(x)
	if math.IsInf(xc.Hi, 1) {
		return 0
	}
	return expScaledR(dd.Sub(dd.Mul(s.cm1(), l), xc), dd.F(s.c))
}

func (s weibullStd) logpdf(x float64) float64 {
	if x == 0 {
		return math.Log(s.atZero())
	}
	xc, l := s.xc(x)
	if math.IsInf(xc.Hi, 1) {
		return math.Inf(-1)
	}
	return dd.Add(dd.Log(s.c), dd.Sub(dd.Mul(s.cm1(), l), xc)).Float()
}

func (s weibullStd) cdf(x float64) float64 {
	xc, _ := s.xc(x)
	return dd.Neg(expm1D(dd.Neg(xc))).Float()
}

func (s weibullStd) sf(x float64) float64 {
	xc, _ := s.xc(x)
	return expR(dd.Neg(xc))
}

// logcdf is log(1 - exp(-x^c)); below x^c = 2^-26 it is
// c log x - x^c/2 to well below the rounding (exact also where x^c is
// subnormal).
func (s weibullStd) logcdf(x float64) float64 {
	xc, l := s.xc(x)
	if xc.Hi < 0x1p-26 {
		return dd.AddF(dd.MulF(l, s.c), -0.5*xc.Hi).Float()
	}
	return log1mexpD(xc).Float()
}

func (s weibullStd) logsf(x float64) float64 {
	xc, _ := s.xc(x)
	return -xc.Float()
}

// root returns t^(1/c) for a double-double t > 0.
func (s weibullStd) root(t dd.DD) float64 { return expR(dd.DivF(logDD(t), s.c)) }

func (s weibullStd) ppf(p float64) float64 { return s.root(dd.Neg(log1pDD(dd.F(-p)))) }
func (s weibullStd) isf(q float64) float64 { return s.root(dd.Neg(dd.Log(q))) }

// dk returns D_k = log Γ(1+kh) - k log Γ(1+h), h = 1/c: by the series
// log Γ(1+z) = -γz + Σ_{j>=2} (-1)^j ζ(j) z^j / j when kh < 1/2 (the γ
// terms cancel exactly), else from LogGamma.
func (s weibullStd) dk(k int) float64 {
	h := 1 / s.c
	fk := float64(k)
	if fk*h >= 0.5 {
		return special.LogGamma(1+fk*h) - fk*special.LogGamma(1+h)
	}
	r := 0.0
	hj := h // h^j
	kj := fk
	for j := 2; j < 80; j++ {
		hj *= h
		kj *= fk
		t := special.Zeta(float64(j)) * (kj - fk) * hj / float64(j)
		if j%2 == 1 {
			t = -t
		}
		r += t
		if math.Abs(t) < 1e-19*math.Abs(r) {
			break
		}
	}
	return r
}

// central returns μ2/m1², μ3/m1³ and μ4/m1⁴ (m1 = Γ(1+1/c)) from
// r_k = Γ(1+k/c)/m1^k = exp(D_k).
func (s weibullStd) central() (v2, v3, v4 float64) {
	e2, e3, e4 := math.Expm1(s.dk(2)), math.Expm1(s.dk(3)), math.Expm1(s.dk(4))
	v2 = e2
	v3 = e3 - 3*e2
	v4 = e4 - 4*e3 + 6*e2
	return
}

func (s weibullStd) mean() float64 { return special.Gamma(1 + 1/s.c) }
func (s weibullStd) variance() float64 {
	m := s.mean()
	v2, _, _ := s.central()
	return m * m * v2
}
func (s weibullStd) skew() float64 {
	v2, v3, _ := s.central()
	return v3 / math.Pow(v2, 1.5)
}
func (s weibullStd) kurtosis() float64 {
	v2, _, v4 := s.central()
	return v4/(v2*v2) - 3
}

// median is (log 2)^(1/c).
func (s weibullStd) median() float64 { return s.root(ln2DD) }

// entropy is -γ/c - log c + γ + 1.
func (s weibullStd) entropy() float64 {
	return dd.Sub(dd.AddF(dd.Sub(eulerDD, dd.DivF(eulerDD, s.c)), 1), dd.Log(s.c)).Float()
}

// munp is Γ(1 + n/c).
func (s weibullStd) munp(n int) float64 { return special.Gamma(1 + float64(n)/s.c) }

// rvs is SciPy's: the quantile formula pow(-log1p(-U), 1/c) of its _ppf
// applied to uniform draws, with SciPy's log1p and a correctly rounded
// power.
func (s weibullStd) rvs(g *random.Generator, size []int) (*ndarray.Array, error) {
	u, err := g.Uniform(0, 1, size...)
	if err != nil {
		return nil, err
	}
	e := 1 / s.c
	v, _ := ndarray.Data[float64](u)
	for i, p := range v {
		t := -cephesLog1p(-p)
		if t == 0 {
			v[i] = 0
			continue
		}
		v[i] = crPow(t, e)
	}
	return u, nil
}

// weibullSkew is the skewness of the shape c as SciPy's fit computes it
// (from gamma functions, for the starting point only).
func weibullSkew(c float64) float64 {
	g1 := special.Gamma(1 + 1/c)
	g2 := special.Gamma(1 + 2/c)
	g3 := special.Gamma(1 + 3/c)
	num := 2*g1*g1*g1 - 3*g1*g2 + g3
	den := math.Pow(g2-g1*g1, 1.5)
	return num / den
}

// FitWeibull returns the maximum-likelihood Weibull distribution for x:
// scipy.stats.weibull_min.fit(x, fc=, floc=, fscale=). It starts where
// SciPy starts (C matching the sample skewness, Scale and Loc the
// variance and mean, or SciPy's generic start when the skewness is below
// what any C reaches) and maximises the likelihood numerically to a
// tighter optimum than SciPy's.
func FitWeibull[D Data](x D, o *FitOptions) (Weibull, error) {
	v := floats(x)
	fx, err := o.fixed("Weibull", "C", "Loc", "Scale")
	if err != nil {
		return Weibull{}, err
	}
	if err := checkFitData(v); err != nil {
		return Weibull{}, err
	}
	mk := func(sh []float64) contStd { return weibullStd{c: sh[0]} }
	sk := skewOf(v)
	fc, haveC := fx["C"]
	if !haveC && sk < weibullSkew(1e4) {
		sh, loc, scale, err := fitContinuous(v, fx, []string{"C"}, []bool{true}, []float64{1}, mk)
		if err != nil {
			return Weibull{}, err
		}
		d := Weibull{C: sh[0], Loc: loc, Scale: scale}
		return d, d.Validate()
	}
	c := fc
	if !haveC {
		r, err := optimize.Bisect(func(c float64) float64 { return weibullSkew(c) - sk }, 0.02, 1e4, nil)
		if err != nil {
			return Weibull{}, err
		}
		c = r.Root
	}
	scale, ok := fx["Scale"]
	if !ok {
		g1 := special.Gamma(1 + 1/c)
		scale = math.Sqrt(variance(v, 0) / (special.Gamma(1+2/c) - g1*g1))
	}
	loc, ok := fx["Loc"]
	if !ok {
		loc = mean(v) - scale*special.Gamma(1+1/c)
		// SciPy's optimiser penalises points outside the support; this
		// one needs a start of positive likelihood.
		if lo, hi := minMax(v); loc >= lo {
			loc = lo - 0.1*(hi-lo) - 1e-3*math.Abs(lo)
		}
	}
	p := mleProblem{
		names:    []string{"C", "Loc", "Scale"},
		positive: []bool{true, false, true},
		start:    []float64{c, loc, scale},
		loglik: func(th []float64) float64 {
			return contLogLik(v, weibullStd{c: th[0]}, th[1], th[2])
		},
	}
	th, err := fitMLE(p, fx)
	if err != nil {
		return Weibull{}, err
	}
	d := Weibull{C: th[0], Loc: th[1], Scale: th[2]}
	return d, d.Validate()
}
