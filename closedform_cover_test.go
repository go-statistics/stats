package stats

import (
	"errors"
	"math"
	"testing"

	"github.com/go-numerics/special"
	"github.com/go-statistics/stats/internal/dd"
)

// The edges of the closed-form families that the fixtures do not reach:
// arguments beyond the float64 range of an intermediate, degenerate
// shapes, and the error paths of the fits.

func TestClosedFormExtremes(t *testing.T) {
	inf := math.Inf(1)
	cases := []struct {
		name      string
		got, want float64
	}{
		{"gumbel pdf far left", Gumbel{Scale: 1}.PDF(-800), 0},
		{"gumbel logpdf far left", Gumbel{Scale: 1}.LogPDF(-800), -inf},
		{"halfnorm pdf huge", HalfNormal{Scale: 1}.PDF(1e200), 0},
		{"halfnorm logpdf huge", HalfNormal{Scale: 1}.LogPDF(1e200), -inf},
		{"rayleigh pdf huge", Rayleigh{Scale: 1}.PDF(1e200), 0},
		{"rayleigh logpdf huge", Rayleigh{Scale: 1}.LogPDF(1e200), -inf},
		{"rayleigh cdf huge", Rayleigh{Scale: 1}.CDF(1e200), 1},
		{"rayleigh sf huge", Rayleigh{Scale: 1}.SF(1e200), 0},
		{"rayleigh logcdf huge", Rayleigh{Scale: 1}.LogCDF(1e200), 0},
		{"rayleigh logsf huge", Rayleigh{Scale: 1}.LogSF(1e200), -inf},
		{"lognorm pdf 0", LogNormal{S: 1, Scale: 1}.PDF(0), 0},
		{"lognorm logpdf 0", LogNormal{S: 1, Scale: 1}.LogPDF(0), -inf},
		{"lognorm pdf huge z", LogNormal{S: 1e-160, Scale: 1}.PDF(2), 0},
		{"lognorm logpdf huge z", LogNormal{S: 1e-160, Scale: 1}.LogPDF(2), -inf},
		{"laplace cdf left", laplaceCDF(-1).Float(), 0.5 * math.Exp(-1)},
		{"triang pdf c=1 at 1", Triangular{C: 1, Scale: 1}.PDF(1), 2},
		{"weibull pdf 0, c<1", Weibull{C: 0.5, Scale: 1}.PDF(0), inf},
		{"weibull pdf 0, c=1", Weibull{C: 1, Scale: 1}.PDF(0), 1},
		{"weibull pdf 0, c>1", Weibull{C: 2, Scale: 1}.PDF(0), 0},
		{"weibull pdf huge", Weibull{C: 3, Scale: 1}.PDF(1e200), 0},
		{"weibull logpdf huge", Weibull{C: 3, Scale: 1}.LogPDF(1e200), -inf},
		{"weibull draw 0", weibullDraw(0, 2), 0},
		{"cauchy draw 1/2", cauchyDraw(0.5), 0},
		{"logD 0", logD(dd.DD{}).Hi, -inf},
	}
	for _, c := range cases {
		if !(c.got == c.want || math.Abs(c.got-c.want) <= 1e-15*math.Abs(c.want)) {
			t.Errorf("%s = %v, want %v", c.name, c.got, c.want)
		}
	}
	if e := expD(dd.F(800)); !math.IsInf(e.Hi, 1) || e.Lo != 0 {
		t.Errorf("expD(800) = %v", e)
	}
	if e := expD(dd.F(709.9)); !math.IsInf(e.Hi, 1) || e.Lo != 0 {
		t.Errorf("expD(709.9) = %v", e)
	}
	if r, ok := brentRoot(func(x float64) float64 { return 1 }, 0, 1); ok || !math.IsNaN(r) {
		t.Errorf("brentRoot without a root = %v, %v", r, ok)
	}
	// ldexpRound decides the subnormal rounding on the low part both ways
	for _, c := range []struct {
		m    dd.DD
		want float64
	}{
		{dd.DD{Hi: 1.25, Lo: 0x1p-60}, 1.25 * 0x1p-1072},
		{dd.DD{Hi: 1.375, Lo: 0x1p-60}, 1.5 * 0x1p-1072},
		{dd.DD{Hi: 1.375, Lo: -0x1p-60}, 1.25 * 0x1p-1072},
		{dd.DD{Hi: 1.125, Lo: 0x1p-60}, 1.25 * 0x1p-1072},
		{dd.DD{Hi: 1.125, Lo: -0x1p-60}, 1.0 * 0x1p-1072},
	} {
		if got := ldexpRound(c.m, -1072); got != c.want {
			t.Errorf("ldexpRound(%v) = %g, want %g", c.m, got, c.want)
		}
	}
}

func TestClosedFormFitErrors(t *testing.T) {
	type fitter func(x []float64, o *FitOptions) (any, error)
	fits := map[string]fitter{
		"Exponential": func(x []float64, o *FitOptions) (any, error) { return FitExponential(x, o) },
		"Uniform":     func(x []float64, o *FitOptions) (any, error) { return FitUniform(x, o) },
		"Reciprocal":  func(x []float64, o *FitOptions) (any, error) { return FitReciprocal(x, o) },
		"LogNormal":   func(x []float64, o *FitOptions) (any, error) { return FitLogNormal(x, o) },
		"Cauchy":      func(x []float64, o *FitOptions) (any, error) { return FitCauchy(x, o) },
		"Laplace":     func(x []float64, o *FitOptions) (any, error) { return FitLaplace(x, o) },
		"Logistic":    func(x []float64, o *FitOptions) (any, error) { return FitLogistic(x, o) },
		"Weibull":     func(x []float64, o *FitOptions) (any, error) { return FitWeibull(x, o) },
		"Pareto":      func(x []float64, o *FitOptions) (any, error) { return FitPareto(x, o) },
		"Gumbel":      func(x []float64, o *FitOptions) (any, error) { return FitGumbel(x, o) },
		"Triangular":  func(x []float64, o *FitOptions) (any, error) { return FitTriangular(x, o) },
		"HalfNormal":  func(x []float64, o *FitOptions) (any, error) { return FitHalfNormal(x, o) },
		"Rayleigh":    func(x []float64, o *FitOptions) (any, error) { return FitRayleigh(x, o) },
		"Lomax":       func(x []float64, o *FitOptions) (any, error) { return FitLomax(x, o) },
		"VonMises":    func(x []float64, o *FitOptions) (any, error) { return FitVonMises(x, o) },
	}
	for name, f := range fits {
		if _, err := f([]float64{1, 2}, &FitOptions{Fixed: map[string]float64{"Nope": 1}}); !errors.Is(err, ErrParam) {
			t.Errorf("%s: unknown fixed parameter: %v", name, err)
		}
		if _, err := f(nil, nil); !errors.Is(err, ErrData) {
			t.Errorf("%s: no data: %v", name, err)
		}
	}
	x := []float64{1, 2, 3, 4.5}
	bad := []struct {
		name  string
		f     string
		fixed map[string]float64
		data  []float64
		want  error
	}{
		{"expon loc above data", "Exponential", map[string]float64{"Loc": 2}, x, ErrData},
		{"uniform loc above data", "Uniform", map[string]float64{"Loc": 2}, x, ErrData},
		{"uniform scale below range", "Uniform", map[string]float64{"Scale": 1}, x, ErrData},
		{"halfnorm loc above data", "HalfNormal", map[string]float64{"Loc": 2}, x, ErrData},
		{"rayleigh loc at data", "Rayleigh", map[string]float64{"Loc": 1}, x, ErrData},
		{"rayleigh tiny scale", "Rayleigh", map[string]float64{"Scale": 1e-300}, x, ErrFit},
		{"lognorm loc at data", "LogNormal", map[string]float64{"Loc": 1}, x, ErrData},
		{"pareto loc+scale above data", "Pareto", map[string]float64{"Loc": 0, "Scale": 2}, x, ErrData},
		{"reciprocal all fixed", "Reciprocal", map[string]float64{"A": 1, "B": 2, "Loc": 0}, x, ErrParam},
		{"reciprocal a above data", "Reciprocal", map[string]float64{"A": 2, "Loc": 0}, x, ErrData},
		{"reciprocal constant", "Reciprocal", nil, []float64{3, 3, 3}, ErrFit},
		{"logistic constant", "Logistic", nil, []float64{3, 3, 3}, ErrData},
		{"gumbel constant", "Gumbel", nil, []float64{3, 3, 3}, ErrData},
		{"vonmises scale", "VonMises", map[string]float64{"Scale": 2}, x, ErrParam},
	}
	for _, c := range bad {
		if _, err := fits[c.f](c.data, &FitOptions{Fixed: c.fixed}); !errors.Is(err, c.want) {
			t.Errorf("%s: error %v, want %v", c.name, err, c.want)
		}
	}
}

// TestClosedFormFitPaths runs the fits along branches the SciPy fixtures
// do not take, checking only that the result is a valid distribution
// whose likelihood is finite.
func TestClosedFormFitPaths(t *testing.T) {
	skewLeft := []float64{-0.1, -0.3, -0.2, -2.5, -0.05, -0.15, -4, -0.4, -1, -0.02}
	cases := []struct {
		name string
		fit  func() (Continuous, error)
	}{
		{"cauchy one point", func() (Continuous, error) {
			return FitCauchy([]float64{2}, &FitOptions{Fixed: map[string]float64{"Scale": 1}})
		}},
		{"cauchy ties", func() (Continuous, error) { return FitCauchy([]float64{1, 1, 1, 1, 5}, nil) }},
		{"weibull left skew", func() (Continuous, error) {
			return FitWeibull(skewLeft, &FitOptions{Fixed: map[string]float64{"Loc": -5}})
		}},
		{"weibull low outlier", func() (Continuous, error) {
			return FitWeibull([]float64{0.9, 1, 1.1, 1.05, 0.95, 1.02, 0.98, 0.1}, nil)
		}},
		{"logistic wide", func() (Continuous, error) {
			return FitLogistic([]float64{-1000, 1000, -999, 999, 0}, nil)
		}},
		{"logistic narrow", func() (Continuous, error) {
			return FitLogistic([]float64{1e-6, 2e-6, -1e-6, 0, 3e-6}, nil)
		}},
		{"pareto small scale", func() (Continuous, error) {
			return FitPareto([]float64{1e-3, 1.1e-3, 1.5e-3, 2e-3, 4e-3}, nil)
		}},
		{"rayleigh far", func() (Continuous, error) {
			return FitRayleigh([]float64{1000, 1000.5, 1001, 1002, 1000.2}, nil)
		}},
		{"reciprocal fixed a", func() (Continuous, error) {
			return FitReciprocal([]float64{2, 3, 5, 9}, &FitOptions{Fixed: map[string]float64{"A": 0.5}})
		}},
		{"lognorm positive slope", func() (Continuous, error) {
			return FitLogNormal([]float64{1, 1.0001, 1.0002, 5, 50, 500}, nil)
		}},
		{"lognorm constant", func() (Continuous, error) {
			return FitLogNormal([]float64{2, 2, 2, 2.0000001}, nil)
		}},
		{"logistic outlier", func() (Continuous, error) {
			x := []float64{100}
			for i := 0; i < 50; i++ {
				x = append(x, float64(i%7)*1e-3)
			}
			return FitLogistic(x, nil)
		}},
		{"weibull start above the minimum", func() (Continuous, error) {
			return FitWeibull([]float64{0.0786489500736487, 0.5599598032573047, -2.414012293904884, 0.2719371550174548,
				0.2823445865414955, 0.9645738602915266, 0.2367669275839951, 0.5364345041162009, 7.679876045795538}, nil)
		}},
		{"lognorm fixed scale", func() (Continuous, error) {
			return FitLogNormal([]float64{1, 2, 3, 4, 8}, &FitOptions{Fixed: map[string]float64{"Scale": 0.01}})
		}},
	}
	// constant data with Loc fixed: the closed form gives S = 0, and the
	// generic fit takes over (whatever it returns)
	_, _ = FitLogNormal([]float64{2, 2, 2}, &FitOptions{Fixed: map[string]float64{"Loc": 0}})
	for _, c := range cases {
		d, err := c.fit()
		if err != nil {
			t.Errorf("%s: %v", c.name, err)
			continue
		}
		if !finite(d.LogPDF(d.Median())) {
			t.Errorf("%s: %+v", c.name, d)
		}
	}
}

func TestVonMisesEdges(t *testing.T) {
	d := VonMises{Kappa: 2, Scale: 1}
	s := d.std()
	if v := s.lowerTail(dd.F(-4)); v.Hi != 0 {
		t.Errorf("lowerTail(-4) = %v", v)
	}
	if v := s.centralLog(0); !math.IsInf(v, -1) {
		t.Errorf("centralLog(0) = %v", v)
	}
	if v := s.logLowerTail(dd.F(-4)); !math.IsInf(v, -1) {
		t.Errorf("logLowerTail(-4) = %v", v)
	}
	// the CDF counts the turns, as SciPy's
	if got, want := d.CDF(7), 1+d.CDF(7-2*math.Pi); math.Abs(got-want) > 1e-15 {
		t.Errorf("CDF(7) = %v, want %v", got, want)
	}
	if got := d.LogCDF(-7); !math.IsNaN(got) {
		t.Errorf("LogCDF(-7) = %v, want NaN", got)
	}
	for _, z := range []float64{3 * math.Pi, -3 * math.Pi, 5 * math.Pi, -5 * math.Pi, math.Pi, -math.Pi} {
		r, n := vmReduce(z)
		if math.Abs(r.Hi) > math.Pi+1e-15 || math.Abs(r.Hi+2*math.Pi*n-z) > 1e-14 {
			t.Errorf("vmReduce(%v) = %v, %v", z, r, n)
		}
	}
	if npMod(0, 2*math.Pi) != 0 || npMod(-1, 2*math.Pi) != 2*math.Pi-1 {
		t.Error("npMod")
	}
	// the sixth moment by quadrature, against the uniform's π⁶/7 at k = 0
	if got, want := (VonMises{Kappa: 0, Scale: 1}).Moment(6), math.Pow(math.Pi, 6)/7; math.Abs(got-want) > 1e-13*want {
		t.Errorf("Moment(6) = %v, want %v", got, want)
	}
	if got := (VonMises{Kappa: 400, Scale: 1}).Moment(6); !(got > 0 && got < 1e-6) {
		t.Errorf("Moment(6) at kappa 400 = %v", got)
	}
	for _, c := range []struct{ r, want float64 }{{1, 1e16}, {-0.1, 0x1p-1022}, {0, 0x1p-1022}} {
		if got := vmKappa(c.r); got != c.want {
			t.Errorf("vmKappa(%v) = %v, want %v", c.r, got, c.want)
		}
	}
	for _, r := range []float64{1e-10, 1 - 1e-9, 0.5} {
		k := vmKappa(r)
		if got := special.I1e(k) / special.I0e(k); math.Abs(got-r) > 1e-12 {
			t.Errorf("vmKappa(%v) = %v: ratio %v", r, k, got)
		}
	}
	// quantiles far in the tails, where Newton's first steps leave the
	// bracket; below the probability of [-π, -π + ulp] (2e-17 at small
	// Kappa), the quantile is -π itself, the closest float
	for _, k := range []float64{0.01, 3, 1e4} {
		d := VonMises{Kappa: k, Scale: 1}
		for _, p := range []float64{1e-300, 1e-30, 0.4999999} {
			x := d.PPF(p)
			if c := d.CDF(x); math.Abs(c-p) > 1e-6*p && x != -math.Pi {
				t.Errorf("kappa %v: CDF(PPF(%v)) = %v", k, p, c)
			}
		}
	}
}
