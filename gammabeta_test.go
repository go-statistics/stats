package stats

import (
	"errors"
	"math"
	"testing"

	"github.com/go-ndarray/ndarray"
	"github.com/go-ndarray/ndarray/random"
	"github.com/go-numerics/special"
)

func same(a, b float64) bool { return a == b || math.IsNaN(a) && math.IsNaN(b) }

// The densities at the closed ends of the support.
func TestDensityEdges(t *testing.T) {
	inf := math.Inf(1)
	cases := []struct {
		name      string
		got, want float64
	}{
		{"gamma a<1", Gamma{A: 0.5, Scale: 1}.PDF(0), inf},
		{"gamma a=1", Gamma{A: 1, Scale: 1}.PDF(0), 1},
		{"gamma a>1", Gamma{A: 2, Scale: 1}.PDF(0), 0},
		{"gamma log a=1", Gamma{A: 1, Scale: 1}.LogPDF(0), 0},
		{"chi2 df=2", ChiSquared{Df: 2, Scale: 1}.PDF(0), 0.5},
		{"chi2 log df=2", ChiSquared{Df: 2, Scale: 1}.LogPDF(0), math.Log(0.5)},
		{"chi df=1", Chi{Df: 1, Scale: 1}.PDF(0), math.Sqrt(2 / math.Pi)},
		{"chi log df=3", Chi{Df: 3, Scale: 1}.LogPDF(0), math.Inf(-1)},
		{"invgamma 0", InverseGamma{A: 2, Scale: 1}.PDF(0), 0},
		{"invgamma log 0", InverseGamma{A: 2, Scale: 1}.LogPDF(0), math.Inf(-1)},
		{"beta a<1", Beta{A: 0.5, B: 2, Scale: 1}.PDF(0), inf},
		{"beta a=1", Beta{A: 1, B: 3, Scale: 1}.PDF(0), 3},
		{"beta a>1", Beta{A: 2, B: 3, Scale: 1}.PDF(0), 0},
		{"beta b=1 at 1", Beta{A: 4, B: 1, Scale: 1}.PDF(1), 4},
		{"beta log at 0", Beta{A: 1, B: 3, Scale: 1}.LogPDF(0), math.Log(3)},
		{"beta log at 1", Beta{A: 4, B: 1, Scale: 1}.LogPDF(1), math.Log(4)},
		{"F dfn=2", F{Dfn: 2, Dfd: 7, Scale: 1}.PDF(0), 1},
		{"F log dfn=1", F{Dfn: 1, Dfd: 7, Scale: 1}.LogPDF(0), inf},
		{"ncx2 at 0, df=2", NoncentralChiSquared{Df: 2, Nc: 3, Scale: 1}.PDF(0), math.Exp(-1.5) * 0.5},
	}
	for _, c := range cases {
		if !(math.Abs(c.got-c.want) <= 1e-15*math.Abs(c.want)) && !same(c.got, c.want) {
			t.Errorf("%s: %v, want %v", c.name, c.got, c.want)
		}
	}
}

// Shapes at the extremes reach the fallbacks of the density helpers.
func TestDensityExtremes(t *testing.T) {
	// a below 1/MaxFloat64: Γ(a) overflows; x^(a-1)e^-x/Γ(a) ≈ a/x
	a := 1e-310
	if got := (Gamma{A: a, Scale: 1}).PDF(0.5); math.Abs(got/(a*math.Exp(-0.5)/0.5)-1) > 1e-12 {
		t.Errorf("gamma pdf at tiny a = %v", got)
	}
	if got := (Gamma{A: a, Scale: 1}).LogPDF(0.5); math.Abs(got-math.Log(a*math.Exp(-0.5)/0.5)) > 1e-12 {
		t.Errorf("gamma logpdf at tiny a = %v", got)
	}
	// B(a, b) far below the normal range: the log form
	d := Beta{A: 29, B: 1e30, Scale: 1}
	x := 29e-30
	want := 28*math.Log(x) + (1e30-1)*math.Log1p(-x) - special.LogBeta(29, 1e30)
	if got := d.LogPDF(x); math.Abs(got-want) > 1e-9*math.Abs(want) {
		t.Errorf("beta logpdf at extreme shapes = %v, want %v", got, want)
	}
	// the Stirling form with x near 0 (log(1+u) taken directly)
	if got, want := (Beta{A: 40, B: 50, Scale: 1}).LogPDF(1e-200), 39*math.Log(1e-200)-special.LogBeta(40, 50); math.Abs(got-want) > 1e-13*math.Abs(want) {
		t.Errorf("beta logpdf near 0 = %v, want %v", got, want)
	}
}

func TestMomentsBranches(t *testing.T) {
	inf, nan := math.Inf(1), math.NaN()
	cases := []struct {
		name      string
		got, want float64
	}{
		{"t var df<=1", StudentT{Df: 1, Scale: 1}.Var(), nan},
		{"t var 1<df<=2", StudentT{Df: 1.5, Scale: 1}.Var(), inf},
		{"t var inf", StudentT{Df: inf, Scale: 1}.Var(), 1},
		{"t kurt inf", StudentT{Df: inf, Scale: 1}.Kurtosis(), 0},
		{"t kurt 2<df<=4", StudentT{Df: 3, Scale: 1}.Kurtosis(), inf},
		{"t kurt df<=2", StudentT{Df: 2, Scale: 1}.Kurtosis(), nan},
		{"t moment odd beyond df", StudentT{Df: 2, Scale: 1}.Moment(5), nan},
		{"t moment inf df", StudentT{Df: inf, Scale: 1}.Moment(4), 3},
		{"t moment 6", StudentT{Df: 10, Scale: 1}.Moment(6), 15 * 1000 / (8 * 6 * 4.0)},
		{"invgamma mean a<=1", InverseGamma{A: 1, Scale: 1}.Mean(), inf},
		{"invgamma var a<=2", InverseGamma{A: 2, Scale: 1}.Var(), inf},
		{"invgamma skew", InverseGamma{A: 3, Scale: 1}.Skew(), nan},
		{"invgamma kurt", InverseGamma{A: 4, Scale: 1}.Kurtosis(), nan},
		{"F mean dfd<=2", F{Dfn: 3, Dfd: 2, Scale: 1}.Mean(), inf},
		{"F var dfd<=4", F{Dfn: 3, Dfd: 4, Scale: 1}.Var(), inf},
		{"F skew dfd<=6", F{Dfn: 3, Dfd: 6, Scale: 1}.Skew(), nan},
		{"F kurt dfd<=8", F{Dfn: 3, Dfd: 8, Scale: 1}.Kurtosis(), nan},
		{"chi2 entropy large", ChiSquared{Df: 400, Scale: 1}.Entropy(), 0},
		{"chi entropy large", Chi{Df: 400, Scale: 1}.Entropy(), 0},
	}
	for _, c := range cases {
		if c.want == 0 { // only finiteness is checked
			if !finite(c.got) {
				t.Errorf("%s = %v", c.name, c.got)
			}
			continue
		}
		if !(math.Abs(c.got-c.want) <= 1e-13*math.Abs(c.want)) && !same(c.got, c.want) {
			t.Errorf("%s = %v, want %v", c.name, c.got, c.want)
		}
	}
	// entropies against mpmath (SciPy's own formulas are off by 3e-9 at
	// (5e6, 6e6) and return 0 at (3, 2e7))
	for _, c := range []struct{ a, b, want float64 }{
		{5e6, 6e6, -7.3850610106085932}, {3, 2e7, -14.963664521155243}, {2e7, 3, -14.963664521155243},
		{3, 1e9, -18.8756873305834}, {300, 40, -2.6351941308233846}, {0.2, 80, -6.8867031357892482},
		{40, 40, -1.4715504090199423},
	} {
		if got := (Beta{A: c.a, B: c.b, Scale: 1}).Entropy(); math.Abs(got-c.want) > 4e-16*math.Abs(c.want)+1e-15 {
			t.Errorf("beta(%v, %v) entropy = %.17g, mpmath %.17g", c.a, c.b, got, c.want)
		}
	}
	for _, c := range []struct{ a, g, ig float64 }{
		{0.5, 0.090609929913988347, 4.0176299819568353}, {29, 3.0909926633661488, -3.6089180847635217},
		{30, 3.1083331192877876, -3.6605431460832622}, {47, 3.3368823057260029, -4.3420608564431408},
		{300, 4.2697177330852444, -7.1345120310440302}, {1e6, 8.3266934788533931, -19.304326637074988},
	} {
		// below a = 30 the formula is SciPy's, log Γ(a) - (a-1)ψ(a) + a,
		// which cancels a few digits; above, the expansion is exact
		tol := 4e-16
		if c.a < stirlingMin {
			tol = 1e-14
		}
		if got := (Gamma{A: c.a, Scale: 1}).Entropy(); math.Abs(got-c.g) > tol*math.Abs(c.g) {
			t.Errorf("gamma(%v) entropy = %.17g, mpmath %.17g", c.a, got, c.g)
		}
		if got := (InverseGamma{A: c.a, Scale: 1}).Entropy(); math.Abs(got-c.ig) > tol*math.Abs(c.ig) {
			t.Errorf("invgamma(%v) entropy = %.17g, mpmath %.17g", c.a, got, c.ig)
		}
	}
}

func TestFitErrors(t *testing.T) {
	good := []float64{1, 2, 3, 4.5}
	fits := map[string]func(x []float64, o *FitOptions) error{
		"Normal":       func(x []float64, o *FitOptions) error { _, err := FitNormal(x, o); return err },
		"Gamma":        func(x []float64, o *FitOptions) error { _, err := FitGamma(x, o); return err },
		"ChiSquared":   func(x []float64, o *FitOptions) error { _, err := FitChiSquared(x, o); return err },
		"Chi":          func(x []float64, o *FitOptions) error { _, err := FitChi(x, o); return err },
		"InverseGamma": func(x []float64, o *FitOptions) error { _, err := FitInverseGamma(x, o); return err },
		"Beta":         func(x []float64, o *FitOptions) error { _, err := FitBeta(x, o); return err },
		"StudentT":     func(x []float64, o *FitOptions) error { _, err := FitStudentT(x, o); return err },
		"F":            func(x []float64, o *FitOptions) error { _, err := FitF(x, o); return err },
	}
	for name, fit := range fits {
		if err := fit(good, &FitOptions{Fixed: map[string]float64{"Nope": 1}}); !errors.Is(err, ErrParam) {
			t.Errorf("%s: unknown parameter: %v", name, err)
		}
		if err := fit(nil, nil); !errors.Is(err, ErrData) {
			t.Errorf("%s: no data: %v", name, err)
		}
		if err := fit([]float64{1, math.NaN()}, nil); !errors.Is(err, ErrData) {
			t.Errorf("%s: NaN data: %v", name, err)
		}
	}
	if _, err := FitNormal(good, &FitOptions{Fixed: map[string]float64{"Loc": 0, "Scale": 1}}); !errors.Is(err, ErrParam) {
		t.Errorf("all fixed: %v", err)
	}
	if _, err := FitGamma(good, &FitOptions{Fixed: map[string]float64{"Loc": 2}}); !errors.Is(err, ErrData) {
		t.Errorf("gamma data below a fixed loc: %v", err)
	}
	if _, err := FitBeta(good, &FitOptions{Fixed: map[string]float64{"Loc": 0, "Scale": 2}}); !errors.Is(err, ErrData) {
		t.Errorf("beta data outside (loc, loc+scale): %v", err)
	}
}

// The numerical fits converge to a local optimum at least as good as the
// starting point, and the closed and equation-solving forms satisfy their
// equations.
func TestFitsNumerical(t *testing.T) {
	g := random.DefaultRNG(5)
	draw := func(a *ndarray.Array, err error) []float64 {
		v, _ := ndarray.Data[float64](a)
		return v
	}
	x := draw(g.Beta(2, 3, 300))
	d, err := FitBeta(x, nil)
	if err != nil || math.Abs(d.A-2) > 1.5 || d.Loc > 0.1 {
		t.Errorf("FitBeta free: %+v %v", d, err)
	}
	// Gamma with fixed loc and scale: ψ(a) = mean log(x/scale)
	y := draw(g.StandardGamma(3, 400))
	gd, err := FitGamma(y, &FitOptions{Fixed: map[string]float64{"Loc": 0, "Scale": 1}})
	if err != nil {
		t.Fatal(err)
	}
	logs := make([]float64, len(y))
	for i, v := range y {
		logs[i] = math.Log(v)
	}
	if r := special.Digamma(gd.A) - mean(logs); math.Abs(r) > 1e-13 {
		t.Errorf("gamma ψ(a) residual %v", r)
	}
	// digammaInv on a negative argument (a < 1/2)
	if a := digammaInv(-5); math.Abs(special.Digamma(a)+5) > 1e-12 {
		t.Errorf("digammaInv(-5) = %v", a)
	}
	// the gamma shape equation with a tiny s (a large)
	if a := solveLogMinusDigamma(1e-6); math.Abs(logMinusDigamma(a)-1e-6) > 1e-21 {
		t.Errorf("solveLogMinusDigamma = %v", a)
	}
	// solveDigammaDiff from a bad start
	if u := solveDigammaDiff(2, -1, math.Inf(1)); math.Abs(special.Digamma(u)-special.Digamma(u+2)+1) > 1e-13 {
		t.Errorf("solveDigammaDiff = %v", u)
	}
	// generic numerical fits, from SciPy's starts
	for name, fit := range map[string]func() (Continuous, error){
		"Chi":          func() (Continuous, error) { return FitChi(draw(g.ChiSquare(4, 60)), nil) },
		"ChiSquared":   func() (Continuous, error) { return FitChiSquared(draw(g.ChiSquare(4, 60)), nil) },
		"InverseGamma": func() (Continuous, error) { return FitInverseGamma(draw(g.StandardGamma(2, 60)), nil) },
		"StudentT":     func() (Continuous, error) { return FitStudentT(draw(g.StandardT(4, 60)), nil) },
		"F":            func() (Continuous, error) { return FitF(draw(g.F(5, 9, 60)), nil) },
	} {
		d, err := fit()
		if err != nil || d.Validate() != nil {
			t.Errorf("%s: %v %v", name, d, err)
		}
	}
	// a start with zero likelihood is reported
	_, err = fitMLE(mleProblem{names: []string{"A"}, positive: []bool{true}, start: []float64{1},
		loglik: func([]float64) float64 { return math.Inf(-1) }}, nil)
	if !errors.Is(err, ErrFit) {
		t.Errorf("fitMLE with zero likelihood: %v", err)
	}
}

func TestLogLikAndStarts(t *testing.T) {
	if !math.IsNaN(contLogLik([]float64{1}, gammaStd{a: 1}, 0, -1)) {
		t.Error("contLogLik with scale < 0")
	}
	if !math.IsInf(contLogLik([]float64{-1}, gammaStd{a: 1}, 0, 1), -1) {
		t.Error("contLogLik outside the support")
	}
	// fitLocScaleSupport: finite support, one-sided support, upper bound
	v := []float64{3, 4, 5, 9}
	if loc, scale := fitLocScaleSupport(v, betaStd{a: 50, b: 50}); !(loc < 3 && loc+scale > 9) {
		t.Errorf("beta start %v %v", loc, scale)
	}
	if loc, scale := fitLocScaleSupport([]float64{-5, 1, 2}, gammaStd{a: 0.1}); !(loc < -5) || scale != 1 {
		t.Errorf("gamma start %v %v", loc, scale)
	}
	if loc, scale := fitLocScaleSupport([]float64{1, 1, 1}, normalStd{}); loc != 1 || scale != 1 {
		t.Errorf("degenerate start %v %v", loc, scale)
	}
	if loc, _ := fitLocScaleSupport([]float64{1, 2}, tStd{df: 1}); loc != 0 {
		t.Errorf("start with infinite moments: loc %v", loc)
	}
	if got := kurtosisOf([]float64{1, 2, 3, 4}); math.Abs(got+1.36) > 1e-12 {
		t.Errorf("kurtosisOf = %v", got)
	}
	if a, b := betaMomentStart([]float64{1, 1, 1, 1}); a != 1 || b != 1 {
		t.Errorf("betaMomentStart degenerate = %v %v", a, b)
	}
}

func TestArrayInput(t *testing.T) {
	if floats[*ndarray.Array](nil) != nil {
		t.Error("nil array")
	}
	a, _ := ndarray.FromSlice([]int64{1, 2, 3}, 3)
	if v := floats(a); len(v) != 3 || v[2] != 3 {
		t.Errorf("int64 array: %v", v)
	}
	d, err := FitNormal(a, nil)
	if err != nil || d.Loc != 2 {
		t.Errorf("FitNormal on an int64 array: %v %v", d, err)
	}
}

func TestNormalExtremes(t *testing.T) {
	if got := (Normal{Scale: 1}).LogPDF(1e160); !math.IsInf(got, -1) {
		t.Errorf("logpdf(1e160) = %v", got)
	}
}
