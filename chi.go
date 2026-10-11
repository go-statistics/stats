package stats

import (
	"math"

	"github.com/go-ndarray/ndarray"
	"github.com/go-ndarray/ndarray/random"
	"github.com/go-numerics/special"
	"github.com/go-statistics/stats/internal/dd"
)

//stats:continuous Chi chi df=Df

// Chi is the chi distribution (the length of a standard normal vector in
// Df dimensions): scipy.stats.chi(df, loc, scale). Its functions are the
// gamma distribution's with shape Df/2 at x²/2, the square carried in
// double-double where the density needs it.
type Chi struct {
	Df, Loc, Scale float64
}

func (d Chi) std() chiStd { return chiStd{k: d.Df / 2} }

type chiStd struct{ k float64 }

func (s chiStd) valid() bool               { return s.k > 0 && !math.IsInf(s.k, 1) }
func (s chiStd) support() (lo, hi float64) { return 0, math.Inf(1) }

// pdf is x·g(x²/2) with g the gamma(k) density: x^(2k-1) e^(-x²/2) /
// (2^(k-1) Γ(k)).
func (s chiStd) pdf(x float64) float64 {
	if x == 0 {
		// x^(2k-1): +Inf below k = 1/2, 0 above, √(2/π) at k = 1/2.
		return gammaAtZero(2*s.k, 0.7978845608028654)
	}
	return gammaDensity(s.k, halfSquare(x), dd.F(x))
}

func (s chiStd) logpdf(x float64) float64 {
	if x == 0 {
		return math.Log(s.pdf(0))
	}
	// log x + (k-1) log(x²/2) - x²/2 - log Γ(k), via the gamma form
	return dd.Add(gammaLogDensity(s.k, halfSquare(x)), dd.Log(x)).Float()
}

func (s chiStd) cdf(x float64) float64 { return special.GammaP(s.k, x*x/2) }
func (s chiStd) sf(x float64) float64  { return special.GammaQ(s.k, x*x/2) }
func (s chiStd) logcdf(x float64) float64 {
	return logPQ(special.GammaP(s.k, x*x/2), special.GammaQ(s.k, x*x/2))
}
func (s chiStd) logsf(x float64) float64 {
	return logPQ(special.GammaQ(s.k, x*x/2), special.GammaP(s.k, x*x/2))
}
func (s chiStd) ppf(p float64) float64 { return math.Sqrt(2 * special.GammaPInv(s.k, p)) }
func (s chiStd) isf(q float64) float64 { return math.Sqrt(2 * special.GammaQInv(s.k, q)) }

// moments returns the mean, variance, skewness and excess kurtosis from
// SciPy's formulas, μ = √2·Γ(k+1/2)/Γ(k), μ2 = df - μ², ..., evaluated in
// double-double: for large df they cancel catastrophically in float64
// (SciPy's kurtosis at df = 400 is wrong in the first digit; here every
// moment is within a few ulp of mpmath's).
func (s chiStd) moments() (mean, mu2, g1, g2 float64) {
	df := dd.F(2 * s.k)
	mu := dd.Mul(dd.DD{Hi: 1.4142135623730951, Lo: -9.667293313452913e-17}, gammaRatioHalf(s.k))
	mumu := dd.Mul(mu, mu)
	v := dd.Sub(df, mumu)
	// g1 = (2μ³ + μ(1 - 2df)) / μ2^1.5
	num1 := dd.Add(dd.MulF(dd.Mul(mumu, mu), 2), dd.Mul(mu, dd.AddF(dd.MulF(df, -2), 1)))
	sd := dd.Sqrt(v)
	g1d := dd.Div(num1, dd.Mul(v, sd))
	// g2 = (2df(1-df) - 6μ⁴ + 4μ²(2df-1)) / μ2²
	num2 := dd.Mul(dd.MulF(df, 2), dd.AddF(dd.Neg(df), 1))
	num2 = dd.Sub(num2, dd.MulF(dd.Mul(mumu, mumu), 6))
	num2 = dd.Add(num2, dd.Mul(dd.MulF(mumu, 4), dd.AddF(dd.MulF(df, 2), -1)))
	g2d := dd.Div(num2, dd.Mul(v, v))
	return mu.Float(), v.Float(), g1d.Float(), g2d.Float()
}

func (s chiStd) mean() float64 {
	m, _, _, _ := s.moments()
	return m
}

func (s chiStd) variance() float64 {
	_, v, _, _ := s.moments()
	return v
}

func (s chiStd) skew() float64 {
	_, _, g1, _ := s.moments()
	return g1
}

func (s chiStd) kurtosis() float64 {
	_, _, _, g2 := s.moments()
	return g2
}

func (s chiStd) median() float64 { return s.ppf(0.5) }

func (s chiStd) entropy() float64 {
	df := 2 * s.k
	if df < 300 {
		return special.LogGamma(0.5*df) + 0.5*(df-math.Ln2-(df-1)*special.Digamma(0.5*df))
	}
	return 0.5 + math.Log(math.Pi)/2 - 1/df/6 - math.Pow(df, -2)/6 - 4.0/45*math.Pow(df, -3) + math.Pow(df, -4)/15
}

// munp is 2^(n/2) Γ(k+n/2)/Γ(k).
func (s chiStd) munp(n int) float64 {
	fn := float64(n)
	return math.Exp(fn/2*math.Ln2 + special.LogGamma(s.k+fn/2) - special.LogGamma(s.k))
}

// rvs is SciPy's sqrt(chi2.rvs(df)).
func (s chiStd) rvs(g *random.Generator, size []int) (*ndarray.Array, error) {
	a, err := g.ChiSquare(2*s.k, size...)
	if err != nil {
		return nil, err
	}
	return a.Sqrt(), nil
}

// FitChi returns the maximum-likelihood chi distribution for x:
// scipy.stats.chi.fit(x, fdf=, floc=, fscale=), numerically from SciPy's
// starting point.
func FitChi[D Data](x D, o *FitOptions) (Chi, error) {
	v := floats(x)
	fx, err := o.fixed("Chi", "Df", "Loc", "Scale")
	if err != nil {
		return Chi{}, err
	}
	if err := checkFitData(v); err != nil {
		return Chi{}, err
	}
	sh, loc, scale, err := fitContinuous(v, fx, []string{"Df"}, []bool{true}, []float64{1},
		func(s []float64) contStd { return chiStd{k: s[0] / 2} })
	if err != nil {
		return Chi{}, err
	}
	d := Chi{Df: sh[0], Loc: loc, Scale: scale}
	return d, d.Validate()
}
