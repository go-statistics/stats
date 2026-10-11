package stats

import (
	"fmt"

	"github.com/go-ndarray/ndarray"
	"github.com/go-ndarray/ndarray/random"
	"github.com/go-statistics/stats/internal/dd"
)

//stats:continuous Uniform uniform

// Uniform is the continuous uniform distribution on [Loc, Loc+Scale]:
// scipy.stats.uniform(loc, scale).
//
// LogCDF is the correctly rounded log x and LogSF log1p(-x) in
// double-double (SciPy's log(1-x) loses the low bits of x below 1/2).
type Uniform struct {
	Loc, Scale float64
}

func (d Uniform) std() uniformStd { return uniformStd{} }

type uniformStd struct{}

func (uniformStd) valid() bool              { return true }
func (uniformStd) support() (a, b float64)  { return 0, 1 }
func (uniformStd) pdf(float64) float64      { return 1 }
func (uniformStd) logpdf(float64) float64   { return 0 }
func (uniformStd) cdf(x float64) float64    { return x }
func (uniformStd) sf(x float64) float64     { return 1 - x }
func (uniformStd) logcdf(x float64) float64 { return dd.Log(x).Float() }
func (uniformStd) logsf(x float64) float64  { return log1pDD(dd.F(-x)).Float() }
func (uniformStd) ppf(p float64) float64    { return p }
func (uniformStd) isf(q float64) float64    { return 1 - q }
func (uniformStd) mean() float64            { return 0.5 }
func (uniformStd) variance() float64        { return 1.0 / 12 }
func (uniformStd) skew() float64            { return 0 }
func (uniformStd) kurtosis() float64        { return -1.2 }
func (uniformStd) median() float64          { return 0.5 }
func (uniformStd) entropy() float64         { return 0 }

// munp is 1/(n+1).
func (uniformStd) munp(n int) float64 { return 1 / float64(n+1) }

func (uniformStd) rvs(g *random.Generator, size []int) (*ndarray.Array, error) {
	return g.Uniform(0, 1, size...)
}

// FitUniform returns the maximum-likelihood uniform distribution for x:
// scipy.stats.uniform.fit(x, floc=, fscale=), to the bit. Loc is the
// minimum and Scale the range; with Loc fixed, Scale reaches the maximum;
// with Scale fixed (at least the range), the support is centred over the
// data, SciPy's choice among the equally likely ones.
func FitUniform[D Data](x D, o *FitOptions) (Uniform, error) {
	v := floats(x)
	fx, err := o.fixed("Uniform", "Loc", "Scale")
	if err != nil {
		return Uniform{}, err
	}
	if err := checkFitData(v); err != nil {
		return Uniform{}, err
	}
	lo, hi := minMax(v)
	var loc, scale float64
	if fscale, ok := fx["Scale"]; ok {
		ptp := hi - lo
		if ptp > fscale {
			return Uniform{}, fmt.Errorf("%w: uniform fit with Scale=%v below the range of the data, %v", ErrData, fscale, ptp)
		}
		loc, scale = lo-float64(0.5*(fscale-ptp)), fscale
	} else if floc, ok := fx["Loc"]; ok {
		if lo < floc {
			return Uniform{}, fmt.Errorf("%w: uniform fit with Loc=%v needs every value at or above it", ErrData, floc)
		}
		loc, scale = floc, hi-floc
	} else {
		loc, scale = lo, hi-lo
	}
	d := Uniform{Loc: loc, Scale: scale}
	return d, d.Validate()
}
