package stats

import (
	"errors"
	"fmt"
	"math"
	"testing"

	"github.com/go-ndarray/ndarray"
	"github.com/go-ndarray/ndarray/random"
)

type kdeFixture struct {
	Cases []struct {
		Name       string   `json:"name"`
		Weights    mvFloats `json:"weights"`
		BW         string   `json:"bw"`
		BWValue    *float64 `json:"bw_value"`
		Dataset    mvFloats `json:"dataset"`
		D, N, M    int
		Factor     mvFloat    `json:"factor"`
		Neff       mvFloat    `json:"neff"`
		Covariance mvFloats   `json:"covariance"`
		ChoCov     mvFloats   `json:"cho_cov"`
		InvCov     mvFloats   `json:"inv_cov"`
		LogDet     mvFloat    `json:"log_det"`
		Scotts     mvFloat    `json:"scotts"`
		Silverman  mvFloat    `json:"silverman"`
		Points     mvFloats   `json:"points"`
		PDF        mvFloats   `json:"pdf"`
		LogPDF     mvFloats   `json:"logpdf"`
		IGMean     mvFloats   `json:"ig_mean"`
		IGCov      mvFloats   `json:"ig_cov"`
		IG         mvFloat    `json:"integrate_gaussian"`
		Box1D      []mvFloats `json:"box1d"`
		Box        *struct {
			Lo, Hi mvFloats
			Value  mvFloat
		} `json:"box"`
		Resample []struct {
			Seed  uint64   `json:"seed"`
			Size  int      `json:"size"`
			Shape []int    `json:"shape"`
			Hex   mvFloats `json:"hex"`
		} `json:"resample"`
	} `json:"cases"`
	Pairs []struct {
		A, B  string
		Value mvFloat  `json:"value"`
		BData mvFloats `json:"b_data"`
	} `json:"pairs"`
	Box3 struct {
		Lo, Hi mvFloats
		Value  mvFloat
	} `json:"box3"`
	Marginal struct {
		Dims   []int    `json:"dims"`
		PDF    mvFloats `json:"pdf"`
		Points mvFloats `json:"points"`
		Factor mvFloat  `json:"factor"`
	} `json:"marginal"`
}

func kdeBandwidth(bw string, v *float64) KDEBandwidth {
	switch bw {
	case "silverman":
		return KDEBandwidth{Rule: Silverman}
	case "scalar":
		return KDEBandwidth{Factor: *v}
	case "half_scott":
		return KDEBandwidth{Func: func(k *GaussianKDE) float64 { return k.ScottsFactor() / 2 }}
	}
	return KDEBandwidth{}
}

func TestGaussianKDEParity(t *testing.T) {
	var fx kdeFixture
	loadMV(t, "kde", &fx)
	var attr, pdf, lpdf, integ, box mvMax
	var draws, draws1 mvDraws
	var resAbs mvMax
	sizeDiff := 0
	kdes := map[string]*GaussianKDE{}
	for ci, c := range fx.Cases {
		o := &GaussianKDEOptions{Bandwidth: kdeBandwidth(c.BW, c.BWValue), Weights: c.Weights.f()}
		if c.Weights == nil {
			o.Weights = nil
		}
		var k *GaussianKDE
		var err error
		if c.D == 1 {
			k, err = NewGaussianKDE(c.Dataset.f(), o)
		} else {
			k, err = NewGaussianKDE(mvArr(c.Dataset.f(), c.D, c.N), o)
		}
		if err != nil {
			t.Fatal(err)
		}
		if c.Weights == nil && c.BW == "scott" {
			kdes[c.Name] = k
		}
		if c.Name == "d2" && c.Weights != nil && c.BW == "scott" {
			kdes["d2w"] = k
		}
		w := fmt.Sprintf("case %d (%s, w=%v, %s)", ci, c.Name, c.Weights != nil, c.BW)
		cov, _ := ndarray.Data[float64](k.Covariance())
		cho, _ := ndarray.Data[float64](k.ChoCov())
		inv, _ := ndarray.Data[float64](k.InvCov())
		for _, p := range [][2][]float64{{cov, c.Covariance.f()}, {cho, c.ChoCov.f()}, {inv, c.InvCov.f()},
			{{k.Factor(), k.Neff(), k.LogDet(), k.ScottsFactor(), k.SilvermanFactor()},
				{float64(c.Factor), float64(c.Neff), float64(c.LogDet), float64(c.Scotts), float64(c.Silverman)}}} {
			for i := range p[1] {
				attr.add(mvRel(p[0][i], p[1][i]), w)
			}
		}
		pts := mvArr(c.Points.f(), c.D, c.M)
		got, err := k.Evaluate(pts)
		if err != nil {
			t.Fatal(err)
		}
		lg, _ := k.LogPDF(pts)
		for i := range got {
			if float64(c.PDF[i]) > 1e-250 {
				pdf.add(mvRel(got[i], float64(c.PDF[i])), fmt.Sprint(w, " point ", i))
			}
			lpdf.add(mvRel(lg[i], float64(c.LogPDF[i])), fmt.Sprint(w, " point ", i))
		}
		ig, err := k.IntegrateGaussian(c.IGMean.f(), mvArr(c.IGCov.f(), c.D, c.D))
		if err != nil {
			t.Fatal(err)
		}
		integ.add(mvRel(ig, float64(c.IG)), w)
		for _, b := range c.Box1D {
			v, err := k.IntegrateBox1D(float64(b[0]), float64(b[1]))
			if err != nil {
				t.Fatal(err)
			}
			box.add(math.Abs(v-float64(b[2])), fmt.Sprint(w, " box ", b.f()))
		}
		if c.Box != nil {
			v, err := k.IntegrateBox(c.Box.Lo.f(), c.Box.Hi.f(), nil)
			if err != nil {
				t.Fatal(err)
			}
			box.add(math.Abs(v-float64(c.Box.Value)), fmt.Sprint(w, " box"))
		}
		for _, r := range c.Resample {
			a, err := k.Resample(random.DefaultRNG(r.Seed), r.Size)
			if err != nil {
				t.Fatal(err)
			}
			v, _ := ndarray.Data[float64](a)
			if r.Size == 0 {
				// The default size is int(neff). SciPy's neff = 1/(w·w)
				// comes from the platform's BLAS dot (49.99999999999999
				// for 50 equal weights with Accelerate); ours is
				// correctly rounded, so int(neff) = n for equal weights.
				if s := a.Shape(); s[1] != int(k.Neff()) {
					t.Errorf("%s: default resample size %d, neff %v", w, s[1], k.Neff())
				}
				if a.Shape()[1] != r.Shape[1] {
					sizeDiff++
				}
				continue
			}
			for i := range v {
				rd := math.Abs(v[i]-float64(r.Hex[i])) / k.cho[0]
				resAbs.add(rd, w)
			}
			if c.D == 1 {
				draws1.add(v, r.Hex.f())
			} else {
				draws.add(v, r.Hex.f())
			}
		}
	}
	t.Logf("attributes (covariance, cho_cov, inv_cov, factor, neff, log_det): max rel diff %.3g (%s)", attr.v, attr.where)
	t.Logf("evaluate: max rel diff %.3g (%s)", pdf.v, pdf.where)
	t.Logf("logpdf: max rel diff %.3g (%s)", lpdf.v, lpdf.where)
	t.Logf("integrate_gaussian: max rel diff %.3g (%s)", integ.v, integ.where)
	t.Logf("integrate_box_1d / integrate_box: max abs diff %.3g (%s)", box.v, box.where)
	t.Logf("resample d=1: %d values, %d differ, max %g ULP; d>1: %d values, %d differ, max %g ULP; max |diff|/bandwidth %.3g (%s); default size differs from SciPy's in %d cases",
		draws1.n, draws1.diff, draws1.maxULP, draws.n, draws.diff, draws.maxULP, resAbs.v, resAbs.where, sizeDiff)
	if attr.v > 1e-13 || pdf.v > 2e-13 || lpdf.v > 1e-13 || integ.v > 1e-13 || box.v > 1e-14 {
		t.Errorf("parity: attributes %g, pdf %g, logpdf %g, integrate_gaussian %g, box %g", attr.v, pdf.v, lpdf.v, integ.v, box.v)
	}
	if resAbs.v > 1e-13 {
		t.Errorf("resample: |diff| %g bandwidths at %s", resAbs.v, resAbs.where)
	}
	for _, p := range fx.Pairs {
		a, b := kdes[p.A], kdes[p.B]
		if p.BData != nil {
			var err error
			b, err = NewGaussianKDE(mvArr(p.BData.f(), 2, len(p.BData)/2), nil)
			if err != nil {
				t.Fatal(err)
			}
		}
		v, err := a.IntegrateKDE(b)
		if err != nil {
			t.Fatal(err)
		}
		v2, _ := b.IntegrateKDE(a)
		if mvRel(v, float64(p.Value)) > 1e-13 || mvRel(v2, float64(p.Value)) > 1e-13 {
			t.Errorf("integrate_kde %s %s: %g %g, SciPy %g", p.A, p.B, v, v2, float64(p.Value))
		}
	}
	k3 := kdes["d3"]
	v, err := k3.IntegrateBox(fx.Box3.Lo.f(), fx.Box3.Hi.f(), &MultivariateNormalCDFOptions{RNG: random.DefaultRNG(4)})
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("integrate_box d=3: %g, SciPy %g", v, float64(fx.Box3.Value))
	if math.Abs(v-float64(fx.Box3.Value)) > 1e-5 {
		t.Errorf("integrate_box d=3: %g, SciPy %g", v, float64(fx.Box3.Value))
	}
	mg, err := k3.Marginal(fx.Marginal.Dims...)
	if err != nil {
		t.Fatal(err)
	}
	got, _ := mg.Evaluate(mvArr(fx.Marginal.Points.f(), 2, 3))
	for i := range got {
		if mvRel(got[i], float64(fx.Marginal.PDF[i])) > 1e-13 || mvRel(mg.Factor(), float64(fx.Marginal.Factor)) > 1e-15 {
			t.Errorf("marginal: %v, SciPy %v", got, fx.Marginal.PDF.f())
		}
	}
}

// kdeExact is the density by the same terms as the windowed sum (relative
// to the nearest point) but over every point, so that the comparison
// measures only the omitted tail.
func kdeExact(k *GaussianKDE, x float64) float64 {
	z := x / k.cho[0]
	a0 := math.Inf(1)
	for _, v := range k.white {
		u := v - z
		a0 = math.Min(a0, u*u/2)
	}
	var s, c float64
	for i, w := range k.ew {
		u := k.white[i] - z
		s, c = neumaier(s, c, w*math.Exp(a0-u*u/2))
	}
	logNorm := -0.5*math.Log(2*math.Pi) - math.Log(k.cho[0])
	return math.Exp(-a0+logNorm) * (s + c)
}

// TestGaussianKDETailCut measures the one-dimensional windowed sum against
// the full sum: the omitted terms must stay below an ULP or two.
func TestGaussianKDETailCut(t *testing.T) {
	g := random.DefaultRNG(5)
	a, _ := g.StandardNormal(3000)
	x, _ := ndarray.Data[float64](a)
	wa, _ := g.Uniform(0, 1, 3000)
	w, _ := ndarray.Data[float64](wa)
	worst := 0.0
	for _, o := range []*GaussianKDEOptions{nil, {Weights: w}, {Bandwidth: KDEBandwidth{Factor: 0.01}}} {
		k, err := NewGaussianKDE(x, o)
		if err != nil {
			t.Fatal(err)
		}
		pts := make([]float64, 401)
		for i := range pts {
			pts[i] = -8 + 0.04*float64(i)
		}
		got, _ := k.EvaluateSlice(pts)
		for i, p := range pts {
			worst = math.Max(worst, mvULP(got[i], kdeExact(k, p)))
		}
	}
	t.Logf("windowed vs full compensated sum: max %g ULP", worst)
	if worst > 4 {
		t.Errorf("windowed sum is %g ULP from the full sum", worst)
	}
}

func TestGaussianKDEEdges(t *testing.T) {
	x := []float64{1, 2, 4, 7, 11}
	k, err := NewGaussianKDE(x, nil)
	if err != nil {
		t.Fatal(err)
	}
	if k.D() != 1 || k.N() != 5 || k.Weights()[0] != 0.2 {
		t.Error(k.D(), k.N(), k.Weights())
	}
	if d, _ := ndarray.Data[float64](k.Dataset()); d[3] != 7 {
		t.Error(d)
	}
	// Far from the data: the density underflows, its logarithm does not.
	lp, _ := k.LogPDFSlice([]float64{1e4, math.Inf(1), math.NaN()})
	if !(lp[0] < -1e6 && lp[0] > -1e9) || !math.IsInf(lp[1], -1) || !math.IsNaN(lp[2]) {
		t.Errorf("far logpdf: %v", lp)
	}
	p, _ := k.EvaluateSlice([]float64{1e4, math.Inf(-1)})
	if p[0] != 0 || p[1] != 0 {
		t.Errorf("far pdf: %v", p)
	}
	// A 0-d point and a single point of a 2-D estimate given as 1-D.
	if v, err := k.Evaluate(ndarray.Scalar(3.0)); err != nil || len(v) != 1 {
		t.Error(v, err)
	}
	k2, err := NewGaussianKDE(mvArr([]float64{1, 2, 3, 4, 2, 1, 4, 3}, 2, 4), &GaussianKDEOptions{Weights: []float64{1, 0, 2, 1}})
	if err != nil {
		t.Fatal(err)
	}
	one, err := k2.EvaluateSlice([]float64{2, 2})
	if err != nil || len(one) != 1 {
		t.Error(one, err)
	}
	col, _ := k2.Evaluate(mvArr([]float64{2, 2}, 2, 1))
	if col[0] != one[0] {
		t.Error(col, one)
	}
	lfar, _ := k2.LogPDF(mvArr([]float64{1e3, 1e3}, 2, 1))
	if !(lfar[0] < -1e3) || math.IsInf(lfar[0], 0) {
		t.Error(lfar)
	}
	lnan, _ := k2.LogPDF(mvArr([]float64{math.NaN(), 1}, 2, 1))
	if !math.IsNaN(lnan[0]) {
		t.Error(lnan)
	}
	for _, c := range []struct {
		name string
		err  error
		f    func() error
	}{
		{"evaluate nil", ErrData, func() error { _, e := k.Evaluate(nil); return e }},
		{"evaluate 3-D", ErrData, func() error { _, e := k.Evaluate(mvArr([]float64{1}, 1, 1, 1)); return e }},
		{"evaluate wrong dim", ErrData, func() error { _, e := k2.EvaluateSlice([]float64{1, 2, 3}); return e }},
		{"box1d in 2-D", ErrData, func() error { _, e := k2.IntegrateBox1D(0, 1); return e }},
		{"box wrong dim", ErrParam, func() error { _, e := k2.IntegrateBox([]float64{0}, []float64{1, 1}, nil); return e }},
		{"gaussian mean", ErrParam, func() error { _, e := k2.IntegrateGaussian([]float64{0}, nil); return e }},
		{"gaussian cov", ErrParam, func() error { _, e := k2.IntegrateGaussian([]float64{0, 0}, mvArr([]float64{1, 2, 3}, 3)); return e }},
		{"gaussian not pd", ErrParam, func() error {
			_, e := k2.IntegrateGaussian([]float64{0, 0}, mvArr([]float64{-50, 0, 0, -50}, 2, 2))
			return e
		}},
		{"kde dims", ErrParam, func() error { _, e := k.IntegrateKDE(k2); return e }},
		{"resample size", ErrParam, func() error { _, e := k.Resample(random.DefaultRNG(0), -1); return e }},
		{"marginal", ErrParam, func() error { _, e := k2.Marginal(0, -2); return e }},
		{"marginal range", ErrParam, func() error { _, e := k2.Marginal(2); return e }},
		{"bandwidth", ErrParam, func() error { return k.SetBandwidth(KDEBandwidth{Rule: 7}) }},
		{"bandwidth factor", ErrParam, func() error { return k.SetBandwidth(KDEBandwidth{Factor: -1}) }},
		{"one value", ErrData, func() error { _, e := NewGaussianKDE([]float64{1}, nil); return e }},
		{"3-D data", ErrData, func() error { _, e := NewGaussianKDE(mvArr(make([]float64, 8), 2, 2, 2), nil); return e }},
		{"d > n", ErrData, func() error { _, e := NewGaussianKDE(mvArr([]float64{1, 2, 3, 4, 5, 6}, 3, 2), nil); return e }},
		{"singular", ErrData, func() error { _, e := NewGaussianKDE(mvArr([]float64{1, 2, 3, 2, 4, 6}, 2, 3), nil); return e }},
		{"weights length", ErrParam, func() error {
			_, e := NewGaussianKDE(x, &GaussianKDEOptions{Weights: []float64{1}})
			return e
		}},
		{"weights negative", ErrParam, func() error {
			_, e := NewGaussianKDE(x, &GaussianKDEOptions{Weights: []float64{1, 1, 1, 1, -1}})
			return e
		}},
		{"weights zero", ErrParam, func() error {
			_, e := NewGaussianKDE(x, &GaussianKDEOptions{Weights: make([]float64, 5)})
			return e
		}},
		{"resample weights", ErrParam, func() error {
			kk, _ := NewGaussianKDE(x, nil)
			kk.weights[0] = 5 // not a probability vector any more
			_, e := kk.Resample(random.DefaultRNG(0), 3)
			return e
		}},
	} {
		if err := c.f(); !errors.Is(err, c.err) {
			t.Errorf("%s: %v, want %v", c.name, err, c.err)
		}
	}
	// SetBandwidth keeps the estimate consistent.
	if err := k.SetBandwidth(KDEBandwidth{Rule: Silverman}); err != nil || k.Factor() != k.SilvermanFactor() {
		t.Error(err, k.Factor())
	}
}

// BenchmarkGaussianKDE1D evaluates 10⁴ points on a 10⁴-point
// one-dimensional estimate (SciPy 1.18.1 takes about 1.5 s for this on
// the same machine, single-threaded C).
func BenchmarkGaussianKDE1D(b *testing.B) {
	g := random.DefaultRNG(1)
	a, _ := g.StandardNormal(10000)
	x, _ := ndarray.Data[float64](a)
	k, _ := NewGaussianKDE(x, nil)
	pa, _ := g.Uniform(-4, 4, 10000)
	pts, _ := ndarray.Data[float64](pa)
	b.ResetTimer()
	for b.Loop() {
		_, _ = k.EvaluateSlice(pts)
	}
}

// BenchmarkGaussianKDE2D evaluates 10³ points on a 10³-point
// two-dimensional estimate.
func BenchmarkGaussianKDE2D(b *testing.B) {
	g := random.DefaultRNG(1)
	a, _ := g.StandardNormal(2, 1000)
	k, _ := NewGaussianKDE(a, nil)
	pa, _ := g.Uniform(-3, 3, 2, 1000)
	b.ResetTimer()
	for b.Loop() {
		_, _ = k.Evaluate(pa)
	}
}

func TestGaussianKDEBadBandwidth(t *testing.T) {
	if _, err := NewGaussianKDE([]float64{1, 2, 3}, &GaussianKDEOptions{Bandwidth: KDEBandwidth{Factor: math.NaN()}}); !errors.Is(err, ErrParam) {
		t.Error(err)
	}
}
