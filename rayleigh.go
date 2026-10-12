package stats

import (
	"fmt"
	"math"

	"github.com/go-ndarray/ndarray"
	"github.com/go-ndarray/ndarray/random"
	"github.com/go-numerics/special"
	"github.com/go-statistics/stats/internal/dd"
)

//stats:continuous Rayleigh rayleigh

// Rayleigh is the Rayleigh distribution: density x exp(-x²/2) on [0, ∞) in
// standard form: scipy.stats.rayleigh(loc, scale), the chi distribution
// with 2 degrees of freedom.
//
// x²/2 is exact in double-double: the density, SF exp(-x²/2) and CDF
// -expm1(-x²/2) round once, LogCDF does not cancel, and the quantiles are
// the square roots of -2 log1p(-p) and -2 log q in double-double.
type Rayleigh struct {
	Loc, Scale float64
}

func (d Rayleigh) std() rayleighStd { return rayleighStd{} }

type rayleighStd struct{}

func (rayleighStd) valid() bool             { return true }
func (rayleighStd) support() (a, b float64) { return 0, math.Inf(1) }

// big is where x² overflows.
const rayleighBig = 1e154

func (rayleighStd) pdf(x float64) float64 {
	if x > rayleighBig {
		return 0
	}
	return expScaledR(negHalfSq(x), dd.F(x))
}

func (rayleighStd) logpdf(x float64) float64 {
	if x > rayleighBig {
		return math.Inf(-1)
	}
	return dd.Add(dd.Log(x), negHalfSq(x)).Float()
}

// rayleighTiny is where x²/2 is below the rounding of exp(-x²/2) - 1 and
// may be subnormal: x·(x/2), x/2 exact, rounds once there.
const rayleighTiny = 0x1p-500

func (rayleighStd) cdf(x float64) float64 {
	switch {
	case x > rayleighBig:
		return 1
	case x < rayleighTiny:
		return float64(0.5*x) * x
	}
	return dd.Neg(expm1D(negHalfSq(x))).Float()
}

func (rayleighStd) sf(x float64) float64 {
	if x > rayleighBig {
		return 0
	}
	return expR(negHalfSq(x))
}

// logcdf is log(1 - exp(-x²/2)): 2 log x - log 2 - x²/4 below x = 2^-13
// (computed without forming x², which may underflow).
func (rayleighStd) logcdf(x float64) float64 {
	switch {
	case x > rayleighBig:
		return 0
	case x < 0x1p-13:
		return dd.AddF(dd.Sub(dd.MulF(dd.Log(x), 2), ln2DD), -0.25*x*x).Float()
	}
	return log1mexpD(halfSquare(x)).Float()
}

func (rayleighStd) logsf(x float64) float64 {
	switch {
	case x > rayleighBig:
		return math.Inf(-1)
	case x < rayleighTiny:
		return -float64(0.5*x) * x
	}
	return negHalfSq(x).Float()
}

func (rayleighStd) ppf(p float64) float64 { return sqrtD(dd.MulF(log1pDD(dd.F(-p)), -2)).Float() }
func (rayleighStd) isf(q float64) float64 { return sqrtD(dd.MulF(dd.Log(q), -2)).Float() }

// The moments: mean √(π/2), variance (4-π)/2, skewness
// 2(π-3)√π/(4-π)^1.5, excess kurtosis 6π/(4-π) - 16/(4-π)².
func (rayleighStd) mean() float64     { return 1.2533141373155003 }
func (rayleighStd) variance() float64 { return 0.4292036732051034 }
func (rayleighStd) skew() float64     { return 0.6311106578189372 }
func (rayleighStd) kurtosis() float64 { return 0.24508930068763807 }

// median is √(2 log 2).
func (rayleighStd) median() float64 { return 1.1774100225154747 }

// entropy is γ/2 + 1 - log(2)/2.
func (rayleighStd) entropy() float64 { return 0.9420342421707938 }

// munp is 2^(n/2) Γ(1 + n/2).
func (rayleighStd) munp(n int) float64 {
	h := float64(n) / 2
	return math.Exp2(h) * special.Gamma(1+h)
}

// rvs is SciPy's sqrt(chi2.rvs(2)).
func (rayleighStd) rvs(g *random.Generator, size []int) (*ndarray.Array, error) {
	a, err := g.ChiSquare(2, size...)
	if err != nil {
		return nil, err
	}
	return a.Sqrt(), nil
}

// FitRayleigh returns the maximum-likelihood Rayleigh distribution for x:
// scipy.stats.rayleigh.fit(x, floc=, fscale=). With Loc fixed, Scale is
// the root mean square of (x - Loc) over √2 (SciPy's value to the bit);
// otherwise Loc solves the likelihood equation, bracketed as SciPy
// brackets it below min(x) and found to full precision.
func FitRayleigh[D Data](x D, o *FitOptions) (Rayleigh, error) {
	v := floats(x)
	fx, err := o.fixed("Rayleigh", "Loc", "Scale")
	if err != nil {
		return Rayleigh{}, err
	}
	if err := checkFitData(v); err != nil {
		return Rayleigh{}, err
	}
	n := float64(len(v))
	t := make([]float64, len(v))
	scaleMLE := func(loc float64) float64 {
		for i, y := range v {
			d := y - loc
			t[i] = float64(d * d)
		}
		return math.Sqrt(sum(t) / (2 * n))
	}
	lo, _ := minMax(v)
	if loc, ok := fx["Loc"]; ok {
		if !(lo > loc) {
			return Rayleigh{}, fmt.Errorf("%w: Rayleigh fit with Loc=%v needs every value above it", ErrData, loc)
		}
		d := Rayleigh{Loc: loc, Scale: scaleMLE(loc)}
		return d, d.Validate()
	}
	s1 := make([]float64, len(v))
	s3 := make([]float64, len(v))
	fscale, haveScale := fx["Scale"]
	fun := func(loc float64) float64 {
		for i, y := range v {
			d := y - loc
			s1[i] = d
			t[i] = float64(d * d)
			s3[i] = 1 / d
		}
		if haveScale {
			return sum(s1) - fscale*fscale*sum(s3)
		}
		return sum(s1) - sum(t)/(2*n)*sum(s3)
	}
	rb := math.Nextafter(lo, math.Inf(-1))
	lb := rb - 1
	diff := 1.0
	for math.Signbit(fun(lb)) == math.Signbit(fun(rb)) {
		diff *= 2
		lb = rb - diff
		if math.IsInf(lb, -1) {
			return Rayleigh{}, fmt.Errorf("%w: no bracket for the Rayleigh likelihood equation", ErrFit)
		}
	}
	loc, _ := brentRoot(fun, lb, rb)
	scale := fscale
	if !haveScale {
		scale = scaleMLE(loc)
	}
	d := Rayleigh{Loc: loc, Scale: scale}
	return d, d.Validate()
}
