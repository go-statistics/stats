package stats

import (
	"math"

	"github.com/go-ndarray/ndarray"
	"github.com/go-ndarray/ndarray/random"
	"github.com/go-numerics/special"
	"github.com/go-statistics/stats/internal/dd"
)

//stats:continuous InverseGamma invgamma a=A

// InverseGamma is the distribution of 1/Y for Y gamma with shape A:
// scipy.stats.invgamma(a, loc, scale). CDF(x) = Q(A, 1/x), SF = P(A, 1/x),
// PPF(p) = 1/Q⁻¹(A, p); the density is the gamma density at 1/x over x².
type InverseGamma struct {
	A, Loc, Scale float64
}

func (d InverseGamma) std() invGammaStd { return invGammaStd{a: d.A} }

type invGammaStd struct{ a float64 }

func (s invGammaStd) valid() bool               { return s.a > 0 && !math.IsInf(s.a, 1) }
func (s invGammaStd) support() (lo, hi float64) { return 0, math.Inf(1) }

func (s invGammaStd) pdf(x float64) float64 {
	if x == 0 {
		return 0
	}
	// g(1/x)/x², 1/x and 1/x² in double-double
	return gammaDensity(s.a, dd.Div(dd.F(1), dd.F(x)), dd.Div(dd.F(1), dd.MulFF(x, x)))
}

func (s invGammaStd) logpdf(x float64) float64 {
	if x == 0 {
		return math.Inf(-1)
	}
	return dd.Sub(gammaLogDensity(s.a, dd.Div(dd.F(1), dd.F(x))), dd.MulF(dd.Log(x), 2)).Float()
}

func (s invGammaStd) cdf(x float64) float64 { return special.GammaQ(s.a, 1/x) }
func (s invGammaStd) sf(x float64) float64  { return special.GammaP(s.a, 1/x) }
func (s invGammaStd) logcdf(x float64) float64 {
	return logPQ(special.GammaQ(s.a, 1/x), special.GammaP(s.a, 1/x))
}
func (s invGammaStd) logsf(x float64) float64 {
	return logPQ(special.GammaP(s.a, 1/x), special.GammaQ(s.a, 1/x))
}
func (s invGammaStd) ppf(p float64) float64 { return 1 / special.GammaQInv(s.a, p) }
func (s invGammaStd) isf(q float64) float64 { return 1 / special.GammaPInv(s.a, q) }

func (s invGammaStd) mean() float64 {
	if s.a > 1 {
		return 1 / (s.a - 1)
	}
	return math.Inf(1)
}
func (s invGammaStd) variance() float64 {
	if s.a > 2 {
		return 1 / (s.a - 1) / (s.a - 1) / (s.a - 2)
	}
	return math.Inf(1)
}
func (s invGammaStd) skew() float64 {
	if s.a > 3 {
		return 4 * math.Sqrt(s.a-2) / (s.a - 3)
	}
	return math.NaN()
}
func (s invGammaStd) kurtosis() float64 {
	if s.a > 4 {
		return 6 * (5*s.a - 11) / (s.a - 3) / (s.a - 4)
	}
	return math.NaN()
}
func (s invGammaStd) median() float64 { return s.ppf(0.5) }

func (s invGammaStd) entropy() float64 {
	a := s.a
	if a >= 200 {
		return (1-3*math.Log(a)+math.Ln2+math.Log(math.Pi))/2 + 2.0/3/a + math.Pow(a, -2)/12 - math.Pow(a, -3)/90 - math.Pow(a, -4)/120
	}
	return a - (a+1)*special.Digamma(a) + special.LogGamma(a)
}

// munp is Γ(a-n)/Γ(a) = 1/((a-1)...(a-n)) for n < a, +Inf beyond.
func (s invGammaStd) munp(n int) float64 {
	if float64(n) >= s.a {
		return math.Inf(1)
	}
	r := 1.0
	for k := 1; k <= n; k++ {
		r /= s.a - float64(k)
	}
	return r
}

// rvs is SciPy's generic inverse-cdf sampling: ppf of uniform draws.
func (s invGammaStd) rvs(g *random.Generator, size []int) (*ndarray.Array, error) {
	u, err := g.Uniform(0, 1, size...)
	if err != nil {
		return nil, err
	}
	return u.Map(func(p float64) float64 { return 1 / special.GammaQInv(s.a, p) }), nil
}

// FitInverseGamma returns the maximum-likelihood inverse gamma
// distribution for x: scipy.stats.invgamma.fit(x, fa=, floc=, fscale=),
// numerically from SciPy's starting point.
func FitInverseGamma[D Data](x D, o *FitOptions) (InverseGamma, error) {
	v := floats(x)
	fx, err := o.fixed("InverseGamma", "A", "Loc", "Scale")
	if err != nil {
		return InverseGamma{}, err
	}
	if err := checkFitData(v); err != nil {
		return InverseGamma{}, err
	}
	sh, loc, scale, err := fitContinuous(v, fx, []string{"A"}, []bool{true}, []float64{1},
		func(s []float64) contStd { return invGammaStd{a: s[0]} })
	if err != nil {
		return InverseGamma{}, err
	}
	d := InverseGamma{A: sh[0], Loc: loc, Scale: scale}
	return d, d.Validate()
}
