package stats

import (
	"math"
	"testing"

	"github.com/go-ndarray/ndarray"
)

// Branches of the hypothesis tests that the SciPy fixtures reach only
// with impractically large inputs, and the helpers' edge cases.

func TestClip01(t *testing.T) {
	for _, c := range [][2]float64{{-1e-17, 0}, {1.5, 1}, {0.25, 0.25}} {
		if got := clip01(c[0]); got != c[1] {
			t.Errorf("clip01(%v) = %v, want %v", c[0], got, c[1])
		}
	}
	if !math.IsNaN(clip01(math.NaN())) {
		t.Error("clip01(NaN) is not NaN")
	}
}

func TestBinarySearchExactHit(t *testing.T) {
	a := func(k float64) float64 { return 2 * k }
	for _, c := range [][2]float64{{6, 3}, {7, 3}, {-1, -1}, {100, 10}} {
		if got := binarySearch(a, c[0], 0, 10); got != c[1] {
			t.Errorf("binarySearch(2k, %v) = %v, want %v", c[0], got, c[1])
		}
	}
}

func TestSmirnovSpecialArguments(t *testing.T) {
	if !math.IsNaN(smirnovSF(10, math.NaN())) {
		t.Error("smirnov(10, NaN) is not NaN")
	}
	if smirnovSF(10, 0) != 1 || smirnovSF(10, -0.5) != 1 {
		t.Error("smirnov(10, x <= 0) != 1")
	}
	if smirnovSF(10, 1) != 0 || smirnovSF(10, 2) != 0 {
		t.Error("smirnov(10, x >= 1) != 0")
	}
}

func TestKolmogorovSpecialArguments(t *testing.T) {
	if !math.IsNaN(kolmogorovSF(math.NaN())) {
		t.Error("kolmogorov(NaN) is not NaN")
	}
	// exp(-π²/(8x²)) underflows to 0 just above the cut-off 0.0406...
	for _, x := range []float64{0.04062, 0.04068} {
		if got := kolmogorovSF(x); got != 1 {
			t.Errorf("kolmogorov(%v) = %v, want 1", x, got)
		}
	}
	if !math.IsNaN(kstwoSF(0, 0.5)) {
		t.Error("kstwo.sf(0.5, n=0) is not NaN")
	}
}

// interleaved returns two samples of n values each whose ECDFs differ by
// at most 1/n.
func interleaved(n int) (a, b []float64) {
	for i := 0; i < n; i++ {
		a = append(a, float64(2*i))
		b = append(b, float64(2*i+1))
	}
	return a, b
}

func TestKS2SampExactFallbacks(t *testing.T) {
	ks := func(a, b []float64, alt Alternative, m Method) KSResult {
		t.Helper()
		r, err := KSTest2Samp(a, b, &KSOptions{Alternative: alt, Method: m})
		if err != nil {
			t.Fatal(err)
		}
		return r
	}
	// D = 1/n with n1 = n2: the alternating sum of the exact two-sided
	// p-value rounds above 1, and SciPy falls back to the asymptotic
	// distribution.
	a, b := interleaved(5)
	if e, as := ks(a, b, TwoSided, MethodExact), ks(a, b, TwoSided, MethodAsymptotic); e.PValue != as.PValue {
		t.Errorf("n1 = n2 = 5, D = 1/5: exact %v, want the asymptotic fallback %v", e.PValue, as.PValue)
	}
	// lcm(n1, n2) beyond int32: the exact method is not attempted.
	big1, big2 := make([]float64, 46349), make([]float64, 46348)
	for i := range big1 {
		big1[i] = float64(i)
	}
	for i := range big2 {
		big2[i] = float64(i) + 0.5
	}
	if e, as := ks(big1, big2, TwoSided, MethodExact), ks(big1, big2, TwoSided, MethodAsymptotic); e.PValue != as.PValue {
		t.Errorf("lcm beyond int32: exact %v, want the asymptotic %v", e.PValue, as.PValue)
	}
	// One-sided with C(n1+n2, n1) overflowing: asymptotic again.
	c, d := make([]float64, 1100), make([]float64, 1000)
	for i := range c {
		c[i] = float64(i)
	}
	for i := range d {
		d[i] = float64(i) + 50.5
	}
	if e, as := ks(c, d, Greater, MethodExact), ks(c, d, Greater, MethodAsymptotic); e.PValue != as.PValue {
		t.Errorf("binomial overflow: exact %v, want the asymptotic %v", e.PValue, as.PValue)
	}
	if got := outerProbInside(3, 2, 1, 1); got != 1 {
		t.Errorf("outerProbInside(3, 2, 1, 1) = %v, want 1", got)
	}
}

func TestContingencyNaN(t *testing.T) {
	tab, _ := ndarray.FromData([]float64{1, math.NaN(), 3, 4}, 2, 2)
	r, err := Chi2Contingency(tab, nil)
	if err != nil || !math.IsNaN(r.Statistic) || !math.IsNaN(r.PValue) {
		t.Errorf("Chi2Contingency with a NaN: %+v, %v", r, err)
	}
	p, err := PowerDivergence([]float64{1, math.NaN(), 3}, 1, &ChiSquareOptions{Expected: []float64{1, 2, 3}})
	if err != nil || !math.IsNaN(p.Statistic) || !math.IsNaN(p.PValue) {
		t.Errorf("PowerDivergence with a NaN: %+v, %v", p, err)
	}
}

func TestAndersonNaN(t *testing.T) {
	r, err := AndersonDarling([]float64{1, 2, math.NaN(), 4, 3}, ADNorm)
	if err != nil || !math.IsNaN(r.Statistic) || !math.IsNaN(r.PValue) {
		t.Errorf("AndersonDarling with a NaN: %+v, %v", r, err)
	}
}
