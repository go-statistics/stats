package stats

import (
	"errors"
	"math"
	"testing"

	"github.com/go-ndarray/ndarray"
)

// Input validation, nil options and the result forms of the hypothesis
// tests (the values themselves are checked against SciPy by
// TestScipyTests).

func wantErr(t *testing.T, name string, err, kind error) {
	t.Helper()
	if !errors.Is(err, kind) {
		t.Errorf("%s: error %v, want one wrapping %v", name, err, kind)
	}
}

func TestTestsInvalidOptions(t *testing.T) {
	x := []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
	y := []float64{2, 1, 4, 3, 6, 5, 8, 7, 10, 9}
	bad := Alternative(7)
	var err error
	_, err = TTest1Samp(x, 0, &TTestOptions{Alternative: bad})
	wantErr(t, "TTest1Samp", err, ErrParam)
	_, err = TTestRel(x, y, &TTestOptions{Alternative: bad})
	wantErr(t, "TTestRel", err, ErrParam)
	_, err = TTestInd(x, y, &TTestIndOptions{Alternative: bad})
	wantErr(t, "TTestInd", err, ErrParam)
	_, err = TTestInd(x, y, &TTestIndOptions{Trim: 0.5})
	wantErr(t, "TTestInd trim", err, ErrParam)
	_, err = FisherExact([2][2]int64{{1, 2}, {3, 4}}, &FisherExactOptions{Alternative: bad})
	wantErr(t, "FisherExact", err, ErrParam)
	_, err = Pearsonr(x, y, &CorrelationOptions{Alternative: bad})
	wantErr(t, "Pearsonr", err, ErrParam)
	_, err = Spearmanr(x, y, &CorrelationOptions{Alternative: bad})
	wantErr(t, "Spearmanr", err, ErrParam)
	_, err = Linregress(x, y, &CorrelationOptions{Alternative: bad})
	wantErr(t, "Linregress", err, ErrParam)
	_, err = Kendalltau(x, y, &KendallOptions{Alternative: bad})
	wantErr(t, "Kendalltau alternative", err, ErrParam)
	_, err = Kendalltau(x, y, &KendallOptions{Variant: 5})
	wantErr(t, "Kendalltau variant", err, ErrParam)
	_, err = Kendalltau(x, y, &KendallOptions{Method: MethodApprox})
	wantErr(t, "Kendalltau method", err, ErrParam)
	_, err = KSTest1Samp(x, Normal{Scale: 1}.CDF, &KSOptions{Alternative: bad})
	wantErr(t, "KSTest1Samp alternative", err, ErrParam)
	_, err = KSTest1Samp(x, Normal{Scale: 1}.CDF, &KSOptions{Method: 9})
	wantErr(t, "KSTest1Samp method", err, ErrParam)
	_, err = KSTest2Samp(x, y, &KSOptions{Alternative: bad})
	wantErr(t, "KSTest2Samp alternative", err, ErrParam)
	_, err = KSTest2Samp(x, y, &KSOptions{Method: MethodApprox})
	wantErr(t, "KSTest2Samp method", err, ErrParam)
	_, err = MannWhitneyU(x, y, &MannWhitneyUOptions{Alternative: bad})
	wantErr(t, "MannWhitneyU alternative", err, ErrParam)
	_, err = MannWhitneyU(x, y, &MannWhitneyUOptions{Method: MethodApprox})
	wantErr(t, "MannWhitneyU method", err, ErrParam)
	_, err = Wilcoxon(x, y, &WilcoxonOptions{Alternative: bad})
	wantErr(t, "Wilcoxon alternative", err, ErrParam)
	_, err = Wilcoxon(x, y, &WilcoxonOptions{ZeroMethod: 4})
	wantErr(t, "Wilcoxon zero method", err, ErrParam)
	_, err = Wilcoxon(x, y, &WilcoxonOptions{Method: MethodApprox})
	wantErr(t, "Wilcoxon method", err, ErrParam)
	_, err = Wilcoxon(x, y[:3], nil)
	wantErr(t, "Wilcoxon lengths", err, ErrData)
	_, err = SkewTest(x, &SkewTestOptions{Alternative: bad})
	wantErr(t, "SkewTest", err, ErrParam)
	_, err = KurtosisTest(x, &SkewTestOptions{Alternative: bad})
	wantErr(t, "KurtosisTest", err, ErrParam)
	_, err = RankData(x, RankMethod(9))
	wantErr(t, "RankData", err, ErrParam)
	_, err = AndersonDarling(x, ADDist(9))
	wantErr(t, "AndersonDarling dist", err, ErrParam)
	_, err = AndersonDarling([]float64{}, ADNorm)
	wantErr(t, "AndersonDarling empty", err, ErrData)
	_, err = AndersonDarling([]float64{1, math.NaN(), 3}, ADGumbelR)
	wantErr(t, "AndersonDarling NaN gumbel", err, ErrData)
	_, err = AndersonDarling([]float64{2, 2, 2, 2}, ADGumbelL)
	wantErr(t, "AndersonDarling constant gumbel", err, ErrFit)
	_, err = Levene(&LeveneOptions{Center: 7}, x, y)
	wantErr(t, "Levene center", err, ErrParam)
	_, err = Levene(&LeveneOptions{Center: CenterTrimmed, ProportionToCut: 0.6}, x, y)
	wantErr(t, "Levene cut", err, ErrParam)
	_, err = Levene[[]float64](nil, x)
	wantErr(t, "Levene groups", err, ErrData)
	_, err = FOneway[[]float64](nil, x)
	wantErr(t, "FOneway groups", err, ErrData)
	_, err = Bartlett(x)
	wantErr(t, "Bartlett groups", err, ErrData)
	_, err = KruskalWallis(x)
	wantErr(t, "KruskalWallis groups", err, ErrData)
	_, err = PowerDivergence(x, 1, &ChiSquareOptions{Expected: y[:3]})
	wantErr(t, "PowerDivergence lengths", err, ErrData)
	_, err = PowerDivergence(x, math.NaN(), nil)
	wantErr(t, "PowerDivergence lambda", err, ErrParam)
	_, err = Chi2Contingency(nil, nil)
	wantErr(t, "Chi2Contingency nil", err, ErrData)
	_, err = FisherExact([2][2]int64{{1, -2}, {3, 4}}, nil)
	wantErr(t, "FisherExact negative", err, ErrData)
	r, _ := TTest1Samp(x, 0, nil)
	_, err = r.ConfidenceInterval(1.5)
	wantErr(t, "TTestResult.ConfidenceInterval", err, ErrParam)
	p, _ := Pearsonr(x, y, nil)
	_, err = p.ConfidenceInterval(-0.1)
	wantErr(t, "PearsonResult.ConfidenceInterval", err, ErrParam)
}

func TestTestsNilOptions(t *testing.T) {
	x := []float64{1.2, 2.5, 2.9, 4.1, 5.3, 6.6, 7.2, 8.0, 9.9, 10.4}
	y := []float64{2.2, 1.1, 4.7, 3.3, 6.2, 5.1, 8.8, 7.4, 10.1, 9.3}
	z := []float64{0.5, 3.1, 2.2, 4.4, 1.9}
	check := func(name string, stat, p float64, err error) {
		t.Helper()
		if err != nil || !(p >= 0 && p <= 1) || math.IsNaN(stat) {
			t.Errorf("%s: statistic %v, p-value %v, error %v", name, stat, p, err)
		}
	}
	r1, err := TTest1Samp(x, 5, nil)
	check("TTest1Samp", r1.Statistic, r1.PValue, err)
	r2, err := TTestRel(x, y, nil)
	check("TTestRel", r2.Statistic, r2.PValue, err)
	r3, err := TTestInd(x, y, nil)
	check("TTestInd", r3.Statistic, r3.PValue, err)
	c1, err := ChiSquare(x, nil)
	check("ChiSquare", c1.Statistic, c1.PValue, err)
	tab, _ := ndarray.FromData([]float64{12, 5, 7, 9}, 2, 2)
	c2, err := Chi2Contingency(tab, nil)
	check("Chi2Contingency", c2.Statistic, c2.PValue, err)
	f, err := FisherExact([2][2]int64{{8, 2}, {1, 5}}, nil)
	check("FisherExact", f.Statistic, f.PValue, err)
	p, err := Pearsonr(x, y, nil)
	check("Pearsonr", p.Statistic, p.PValue, err)
	s, err := Spearmanr(x, y, nil)
	check("Spearmanr", s.Statistic, s.PValue, err)
	k, err := Kendalltau(x, y, nil)
	check("Kendalltau", k.Statistic, k.PValue, err)
	l, err := Linregress(x, y, nil)
	check("Linregress", l.Slope, l.PValue, err)
	ks, err := KSTest1Samp(x, Normal{Loc: 5, Scale: 3}.CDF, nil)
	check("KSTest1Samp", ks.Statistic, ks.PValue, err)
	ks2, err := KSTest2Samp(x, z, nil)
	check("KSTest2Samp", ks2.Statistic, ks2.PValue, err)
	m, err := MannWhitneyU(x, z, nil)
	check("MannWhitneyU", m.Statistic, m.PValue, err)
	w, err := Wilcoxon(x, y, nil)
	check("Wilcoxon", w.Statistic, w.PValue, err)
	sk, err := SkewTest(x, nil)
	check("SkewTest", sk.Statistic, sk.PValue, err)
	ku, err := KurtosisTest(x, nil)
	check("KurtosisTest", ku.Statistic, ku.PValue, err)
	fo, err := FOneway(nil, x, y, z)
	check("FOneway", fo.Statistic, fo.PValue, err)
	lv, err := Levene(nil, x, y, z)
	check("Levene", lv.Statistic, lv.PValue, err)
}

func TestMethodString(t *testing.T) {
	for m, s := range map[Method]string{MethodAuto: "auto", MethodExact: "exact",
		MethodAsymptotic: "asymptotic", MethodApprox: "approx", Method(9): "invalid"} {
		if m.String() != s {
			t.Errorf("Method(%d).String() = %q, want %q", int(m), m.String(), s)
		}
	}
}

func TestRankDataArray(t *testing.T) {
	a, _ := ndarray.FromData([]float64{3, 1, 2, 1, 5, 4}, 2, 3)
	r, err := RankData(a, RankAverage)
	if err != nil {
		t.Fatal(err)
	}
	got, _ := ndarray.Data[float64](r)
	want := []float64{4, 1.5, 3, 1.5, 6, 5}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("RankData = %v, want %v", got, want)
		}
	}
	if s := r.Shape(); len(s) != 2 || s[0] != 2 || s[1] != 3 {
		t.Errorf("RankData shape %v, want [2 3]", s)
	}
}
