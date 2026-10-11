package stats

import (
	"errors"
	"math"
	"testing"

	"github.com/go-numerics/special"
)

// stubUpper is a standard form whose support is bounded above only.
type stubUpper struct{ normalStd }

func (stubUpper) support() (float64, float64) { return math.Inf(-1), 0 }

func TestCoverParent(t *testing.T) {
	// a fixed negative scale makes the starting likelihood zero
	bad := &FitOptions{Fixed: map[string]float64{"Scale": -1}}
	x := []float64{0.2, 0.4, 0.5, 0.7}
	for name, err := range map[string]error{
		"Gamma":        func() error { _, e := FitGamma(x, bad); return e }(),
		"Chi":          func() error { _, e := FitChi(x, bad); return e }(),
		"ChiSquared":   func() error { _, e := FitChiSquared(x, bad); return e }(),
		"InverseGamma": func() error { _, e := FitInverseGamma(x, bad); return e }(),
		"Beta":         func() error { _, e := FitBeta(x, bad); return e }(),
		"StudentT":     func() error { _, e := FitStudentT(x, bad); return e }(),
		"F":            func() error { _, e := FitF(x, bad); return e }(),
	} {
		if !errors.Is(err, ErrFit) {
			t.Errorf("%s with scale -1: %v", name, err)
		}
	}
	// support bounded above only: the start moves the location above the data
	if loc, scale := fitLocScaleSupport([]float64{1, 2, 3}, stubUpper{}); !(loc > 3) || scale != 1 {
		t.Errorf("upper-bounded start %v %v", loc, scale)
	}
	// chi below the square's underflow
	c := Chi{Df: 3, Scale: 1}
	if got := c.SF(1e-200); got != 1 {
		t.Errorf("chi sf(1e-200) = %v", got)
	}
	if got := c.LogCDF(1e-200); !(got < -1000) {
		t.Errorf("chi logcdf(1e-200) = %v", got)
	}
	if got := c.LogSF(1e-200); got != 0 && !(got < 0 && got > -1e-300) {
		t.Errorf("chi logsf(1e-200) = %v", got)
	}
	// log Q below the continued fraction's range
	if got, want := logUpperGamma(5, 3).Float(), math.Log(special.GammaQ(5, 3)); math.Abs(got-want) > 1e-15 {
		t.Errorf("logUpperGamma(5, 3) = %v, want %v", got, want)
	}
	// the quantile inverter from bad guesses and on a two-sided support
	n := normalStd{}
	for _, x0 := range []float64{math.NaN(), 50, -50} {
		got := invertCDF(0.3, false, n.cdf, n.pdf, x0, math.Inf(-1), math.Inf(1))
		if math.Abs(got-special.Ndtri(0.3)) > 1e-15 {
			t.Errorf("invertCDF from %v = %v", x0, got)
		}
	}
	if m := midpoint(math.Inf(-1), -3); m != -6 {
		t.Errorf("midpoint(-inf, -3) = %v", m)
	}
	if m := midpoint(2, 512); math.Abs(m-32) > 1e-13 {
		t.Errorf("midpoint(2, 512) = %v", m)
	}
	// the ncx2 mixture walks down to its largest term
	d := NoncentralChiSquared{Df: 0.5, Nc: 400, Scale: 1}
	for _, x := range []float64{1e-3, 0.5, 30, 380, 2000} {
		if v := d.PDF(x); !(v >= 0) {
			t.Errorf("ncx2 pdf(%v) = %v", x, v)
		}
	}
	if a, b := betaMomentStart([]float64{0.0021155627801495948, 0.01862151275465947, 0.7837613578183948, 0.00015634595800763905, 0.7507675410989773}); a != 1 || b != 1 {
		t.Errorf("betaMomentStart fallback = %v %v", a, b)
	}
	// a multinomial outcome on a zero-probability category
	if m, err := NewMultinomial(3, []float64{0.5, 0.5, 0}); err != nil {
		t.Error(err)
	} else if l, _ := m.LogPMF([]float64{1, 1, 1}); !math.IsInf(l, -1) {
		t.Errorf("multinomial logpmf on a zero-probability category = %v", l)
	}
	// beta moment start without a positive solution
	for _, v := range [][]float64{{0.1, 0.1, 0.1, 0.9}, {0.01, 0.02, 0.5, 0.99, 0.999}, {0.1, 0.9, 0.1, 0.9}, {0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.99}} {
		if a, b := betaMomentStart(v); !(a > 0 && b > 0) {
			t.Errorf("betaMomentStart %v = %v %v", v, a, b)
		}
	}
}
