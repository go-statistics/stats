package stats

import (
	"math"
	"testing"

	"github.com/go-statistics/stats/internal/dd"
)

// TestHangRegressions holds the calls the extreme-parameter sweep found
// not returning (or running for minutes) to an answer within the sweep's
// time bound, and checks the answers.
func TestHangRegressions(t *testing.T) {
	finite := func(v float64) bool { return !math.IsNaN(v) && !math.IsInf(v, 0) }
	for _, c := range []struct {
		name string
		f    func() float64
		ok   func(v float64) bool
	}{
		// entropySum looped on NaN terms (a zero mass times log 0)
		{"Poisson{Mu: 5e-324}.Entropy", Poisson{Mu: 5e-324}.Entropy, func(v float64) bool { return v >= 0 && v < 1e-300 }},
		{"NegativeBinomial{N: 5e-324, P: 0.4}.Entropy", NegativeBinomial{N: 5e-324, P: 0.4}.Entropy, func(v float64) bool { return v >= 0 && v < 1e-300 }},
		// ratioSum stepped j -= 1 from MaxFloat64, where j - 1 == j,
		// through its whole budget of 4·10^6 terms (seconds; minutes
		// under -race). The value is NaN until go-numerics/special's
		// GammaQ(MaxFloat64, x) and BetaInc(a, MaxFloat64, x) are fixed
		// (they overflow to NaN there); only the time is held here.
		{"Poisson{Mu: 3.5}.LogCDF(MaxFloat64)", func() float64 { return Poisson{Mu: 3.5}.LogCDF(math.MaxFloat64) }, func(v float64) bool { return v == 0 || math.IsNaN(v) }},
		{"NegativeBinomial{N: 30.5, P: 0.97}.LogCDF(MaxFloat64)", func() float64 {
			return NegativeBinomial{N: 30.5, P: 0.97}.LogCDF(math.MaxFloat64)
		}, func(v float64) bool { return v == 0 || math.IsNaN(v) }},
		// summed 10^9 terms or more (now entropyIntegral)
		{"NegativeBinomial{N: 5, P: 1e-8}.Entropy", NegativeBinomial{N: 5, P: 1e-8}.Entropy, finite},
		{"NegativeBinomial{N: 5, P: 5e-324}.Entropy", NegativeBinomial{N: 5, P: 5e-324}.Entropy, finite},
		// zetaTailScaled's j++ stopped moving beyond 2^53
		{"Zipf{A: 1e20}.CDF(1e20)", func() float64 { return Zipf{A: 1e20}.CDF(1e20) }, func(v float64) bool { return v == 1 }},
		{"Zipf{A: 1e20}.SF(1e20)", func() float64 { return Zipf{A: 1e20}.SF(1e20) }, func(v float64) bool { return v == 0 }},
		{"Zipf{A: 2^53}.LogSF(2^53)", func() float64 { return Zipf{A: 0x1p53}.LogSF(0x1p53) }, func(v float64) bool { return v < -1e15 }},
		{"Zipf{A: 1.5}.SF(1e300)", func() float64 { return Zipf{A: 1.5}.SF(1e300) }, func(v float64) bool { return v > 0 && v < 1e-149 }},
		// ncx2: the walk through all-zero terms, logSFDeep from j = 0, the
		// quadrature of a density of relative width 1e-50
		{"NoncentralChiSquared{Df: 3, Nc: 1e8}.CDF(MaxFloat64)", func() float64 {
			return NoncentralChiSquared{Df: 3, Nc: 1e8, Scale: 1}.CDF(math.MaxFloat64)
		}, func(v float64) bool { return v == 1 }},
		{"NoncentralChiSquared{Df: 3, Nc: 1e-8}.LogSF(1e20)", func() float64 {
			return NoncentralChiSquared{Df: 3, Nc: 1e-8, Scale: 1}.LogSF(1e20)
		}, func(v float64) bool { return v < -4e19 && v > -6e19 }},
		{"NoncentralChiSquared{Df: 3, Nc: 2}.LogPDF(MaxFloat64)", func() float64 {
			return NoncentralChiSquared{Df: 3, Nc: 2, Scale: 1}.LogPDF(math.MaxFloat64)
		}, func(v float64) bool { return math.IsInf(v, -1) }},
		{"NoncentralChiSquared{Df: 1e100, Nc: 2}.Entropy", func() float64 {
			return NoncentralChiSquared{Df: 1e100, Nc: 2, Scale: 1}.Entropy()
		}, func(v float64) bool { return math.Abs(v-0.5*math.Log(2*math.Pi*math.E*2e100)) < 1e-12*v }},
		{"NoncentralChiSquared{Df: 3, Nc: 1e300}.CDF(1)", func() float64 {
			return NoncentralChiSquared{Df: 3, Nc: 1e300, Scale: 1}.CDF(1)
		}, math.IsNaN},
	} {
		var v float64
		if s := guarded(func() { v = c.f() }); s != "" {
			t.Errorf("%s: %s", c.name, s)
			continue
		}
		if !c.ok(v) {
			t.Errorf("%s = %v", c.name, v)
		}
	}
}

// TestEntropyIntegralAgainstSum checks the Euler-Maclaurin entropy of the
// wide negative binomials against the plain sum where both can run: the
// mode inside the integral's range (N = 5) and at 0 (N = 0.5, the first
// 1024 terms summed), and far above it (N = 1000: the pieces below the mode
// vanish before reaching them).
func TestEntropyIntegralAgainstSum(t *testing.T) {
	for _, d := range []NegativeBinomial{{N: 5, P: 6e-5}, {N: 0.5, P: 2e-5}, {N: 1000, P: 1e-3}} {
		s := d.std()
		mode := 0.0
		if s.n > 1 {
			mode = math.Floor((s.n - 1) * s.q.Float() / s.p)
		}
		sd := math.Sqrt(s.variance())
		if sd <= entropyIntegralMinSD {
			t.Fatalf("%v: sd %v does not take the integral", d, sd)
		}
		got := entropyIntegral(0, mode, sd, s.term)
		want := entropySum(mode, 0, math.Inf(1), s.term)
		if math.Abs(got-want) > 1e-13*want {
			t.Errorf("%v: integral %v, sum %v (rel %.2g)", d, got, want, (got-want)/want)
		}
		if e := d.Entropy(); e != got {
			t.Errorf("%v: Entropy %v, integral %v", d, e, got)
		}
	}
	// the gamma limit below P = 2^-53 against the integral just above it
	lim := NegativeBinomial{N: 5, P: 0x1p-54}.Entropy()
	s := NegativeBinomial{N: 5, P: 0x1p-52}.std()
	near := entropyIntegral(0, math.Floor(4*s.q.Float()/s.p), math.Sqrt(s.variance()), s.term)
	if d := lim - near - math.Log(4); math.Abs(d) > 1e-13*lim { // log(P'/P) = log 4
		t.Errorf("gamma limit %v vs integral %v + log 4: %g", lim, near, d)
	}
}

// TestDefensiveGuards reaches the guards that turn an argument no caller
// passes any more (an unreduced angle, a tangent beyond 1) into NaN rather
// than an index out of range, and the NaN of the noncentral chi-squared
// beyond ncx2MaxMu.
func TestDefensiveGuards(t *testing.T) {
	if s, c := sincosSmall(dd.F(1e20)); !math.IsNaN(s.Hi) || !math.IsNaN(c.Hi) {
		t.Errorf("sincosSmall(1e20) = %v, %v", s, c)
	}
	if v := atanD(dd.F(2)); !math.IsNaN(v.Hi) {
		t.Errorf("atanD(2) = %v", v)
	}
	d := NoncentralChiSquared{Df: 3, Nc: 1e300, Scale: 1}
	for name, v := range map[string]float64{
		"PDF": d.PDF(1), "SF": d.SF(1), "LogSF": d.LogSF(1), "PPF": d.PPF(0.5), "ISF": d.ISF(0.5),
	} {
		if !math.IsNaN(v) {
			t.Errorf("Nc = 1e300: %s = %v", name, v)
		}
	}
}
