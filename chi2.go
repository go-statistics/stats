package stats

import (
	"math"

	"github.com/go-ndarray/ndarray"
	"github.com/go-ndarray/ndarray/random"
	"github.com/go-numerics/special"
	"github.com/go-statistics/stats/internal/dd"
)

//stats:continuous ChiSquared chi2 df=Df

// ChiSquared is the chi-squared distribution with Df degrees of freedom:
// scipy.stats.chi2(df, loc, scale). It is the gamma distribution with
// shape Df/2 and scale 2, computed through it (x/2 and Df/2 are exact).
type ChiSquared struct {
	Df, Loc, Scale float64
}

func (d ChiSquared) std() chi2Std { return chi2Std{k: d.Df / 2} }

// chi2Std holds k = df/2.
type chi2Std struct{ k float64 }

func (s chi2Std) valid() bool               { return s.k > 0 && !math.IsInf(s.k, 1) }
func (s chi2Std) support() (lo, hi float64) { return 0, math.Inf(1) }

func (s chi2Std) pdf(x float64) float64 {
	if x == 0 {
		return gammaAtZero(s.k, 0.5)
	}
	return gammaDensity(s.k, dd.F(x/2), dd.F(0.5))
}

func (s chi2Std) logpdf(x float64) float64 {
	if x == 0 {
		return math.Log(gammaAtZero(s.k, 0.5))
	}
	return dd.Sub(gammaLogDensity(s.k, dd.F(x/2)), dd.DD{Hi: 0.6931471805599453, Lo: 2.3190468138462996e-17}).Float()
}

func (s chi2Std) cdf(x float64) float64 { return special.GammaP(s.k, x/2) }
func (s chi2Std) sf(x float64) float64  { return special.GammaQ(s.k, x/2) }
func (s chi2Std) logcdf(x float64) float64 {
	return logPQ(special.GammaP(s.k, x/2), special.GammaQ(s.k, x/2))
}
func (s chi2Std) logsf(x float64) float64 {
	return logPQ(special.GammaQ(s.k, x/2), special.GammaP(s.k, x/2))
}
func (s chi2Std) ppf(p float64) float64 { return 2 * special.GammaPInv(s.k, p) }
func (s chi2Std) isf(q float64) float64 { return 2 * special.GammaQInv(s.k, q) }

func (s chi2Std) mean() float64     { return 2 * s.k }
func (s chi2Std) variance() float64 { return 4 * s.k }
func (s chi2Std) skew() float64     { return 2 * math.Sqrt(2.0/(2*s.k)) }
func (s chi2Std) kurtosis() float64 { return 12.0 / (2 * s.k) }
func (s chi2Std) median() float64   { return 2 * special.GammaPInv(s.k, 0.5) }

// entropy is the gamma entropy at k = df/2 plus log 2 (the scale).
func (s chi2Std) entropy() float64 { return gammaEntropyPart(s.k) + math.Ln2 }

// munp is 2^n Γ(k+n)/Γ(k).
func (s chi2Std) munp(n int) float64 { return math.Ldexp(poch(s.k, n), n) }

func (s chi2Std) rvs(g *random.Generator, size []int) (*ndarray.Array, error) {
	return g.ChiSquare(2*s.k, size...)
}

// FitChiSquared returns the maximum-likelihood chi-squared distribution
// for x: scipy.stats.chi2.fit(x, fdf=, floc=, fscale=), numerically from
// SciPy's starting point.
func FitChiSquared[D Data](x D, o *FitOptions) (ChiSquared, error) {
	v := floats(x)
	fx, err := o.fixed("ChiSquared", "Df", "Loc", "Scale")
	if err != nil {
		return ChiSquared{}, err
	}
	if err := checkFitData(v); err != nil {
		return ChiSquared{}, err
	}
	sh, loc, scale, err := fitContinuous(v, fx, []string{"Df"}, []bool{true}, []float64{1},
		func(s []float64) contStd { return chi2Std{k: s[0] / 2} })
	if err != nil {
		return ChiSquared{}, err
	}
	d := ChiSquared{Df: sh[0], Loc: loc, Scale: scale}
	return d, d.Validate()
}
