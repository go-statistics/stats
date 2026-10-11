package stats

import (
	"errors"
	"math"
	"math/rand/v2"
	"slices"
	"testing"

	"github.com/go-ndarray/ndarray"
)

// Argument validation and the corners the NumPy/SciPy fixtures do not
// reach.

func TestDescribeErrors(t *testing.T) {
	nan, inf := math.NaN(), math.Inf(1)
	x := []float64{1, 2, 3, 4}
	none := []float64{}
	one := 1.0
	type check struct {
		name string
		err  error
		want error
	}
	e := func(_ any, err error) error { return err }
	e3 := func(_, _ any, err error) error { return err }
	checks := []check{
		{"Quantile method", e(Quantile(x, 0.5, &QuantileOptions{Method: 99})), ErrParam},
		{"Quantile q", e(Quantile(x, 1.5, nil)), ErrParam},
		{"Quantile q NaN", e(Quantile(x, nan, nil)), ErrParam},
		{"Percentile p", e(Percentile(x, -1, nil)), ErrParam},
		{"Quantile empty", e(Quantile(none, 0.5, nil)), ErrData},
		{"Quantile weights method", e(Quantile(x, 0.5, &QuantileOptions{Weights: []float64{1, 1, 1, 1}})), ErrParam},
		{"Quantile weights length", e(Quantile(x, 0.5, &QuantileOptions{Method: QuantileInvertedCDF, Weights: []float64{1}})), ErrData},
		{"Quantile weights negative", e(Quantile(x, 0.5, &QuantileOptions{Method: QuantileInvertedCDF, Weights: []float64{1, -1, 1, 1}})), ErrData},
		{"Quantile weights zero", e(Quantile(x, 0.5, &QuantileOptions{Method: QuantileInvertedCDF, Weights: []float64{0, 0, 0, 0}})), ErrData},
		{"Median empty", e(Median(none, nil)), ErrData},
		{"Mean empty", e(Mean(none, nil)), ErrData},
		{"Mean all NaN", e(Mean([]float64{nan}, &MeanOptions{OmitNaN: true})), ErrData},
		{"Mean weights length", e(Mean(x, &MeanOptions{Weights: []float64{1}})), ErrData},
		{"Mean weights zero", e(Mean(x, &MeanOptions{Weights: []float64{0, 0, 0, 0}})), ErrData},
		{"Var empty", e(Var(none, nil)), ErrData},
		{"Var weights zero", e(Var(x, &MeanOptions{Weights: []float64{0, 0, 0, 0}})), ErrData},
		{"Std empty", e(Std(none, nil)), ErrData},
		{"Average empty", e3(Average(none, nil)), ErrData},
		{"Moment order", e(Moment(x, -1, nil)), ErrParam},
		{"Moment empty", e(Moment([]float64{nan}, 2, &MomentOptions{OmitNaN: true})), ErrData},
		{"Skew empty", e(Skew(none, nil)), ErrData},
		{"Kurtosis empty", e(Kurtosis(none, nil)), ErrData},
		{"Describe empty", e(Describe(none, nil)), ErrData},
		{"SEM one", e(SEM([]float64{1}, nil)), ErrData},
		{"Variation empty", e(Variation(none, nil)), ErrData},
		{"ZScore empty", e(ZScore(none, nil)), ErrData},
		{"ZMap all NaN", e(ZMap(x, []float64{nan}, &DdofOptions{OmitNaN: true})), ErrData},
		{"GMean empty", e(GMean(none, nil)), ErrData},
		{"GMean weights", e(GMean(x, &PowerMeanOptions{Weights: []float64{1}})), ErrData},
		{"HMean empty", e(HMean(none, nil)), ErrData},
		{"PMean p", e(PMean(x, inf, nil)), ErrParam},
		{"PMean empty", e(PMean(none, 2, nil)), ErrData},
		{"Mode empty", e(Mode(none, nil)), ErrData},
		{"IQR range NaN", e(IQR(x, &IQROptions{Range: [2]float64{nan, 50}})), ErrParam},
		{"IQR range", e(IQR(x, &IQROptions{Range: [2]float64{-1, 50}})), ErrParam},
		{"IQR method", e(IQR(x, &IQROptions{Method: -1})), ErrParam},
		{"IQR empty", e(IQR(none, nil)), ErrData},
		{"MAD empty", e(MedianAbsDeviation(none, nil)), ErrData},
		{"TMean empty", e(TMean(none, nil)), ErrData},
		{"TMean limits", e(TMean(x, &TrimmedOptions{Limits: &Limits{Lower: 10, Upper: 20}})), ErrData},
		{"TVar empty", e(TVar([]float64{nan}, &TrimmedOptions{OmitNaN: true})), ErrData},
		{"TStd empty", e(TStd(none, nil)), ErrData},
		{"TSem empty", e(TSem(none, nil)), ErrData},
		{"TMin empty", e(TMin([]float64{nan}, 0, &BoundOptions{OmitNaN: true})), ErrData},
		{"TMin limit", e(TMin(x, 10, nil)), ErrData},
		{"TMax limit", e(TMax(x, 1, &BoundOptions{Exclusive: true})), ErrData},
		{"TrimMean empty", e(TrimMean(none, 0.1)), ErrData},
		{"TrimMean proportion", e(TrimMean(x, 0.8)), ErrParam},
		{"TrimBoth proportion", e(TrimBoth(x, 0.5)), ErrParam},
		{"Trim1 proportion", e(Trim1(x, -0.5, TailRight)), ErrParam},
		{"Entropy base", e(Entropy(x, &EntropyOptions{Base: -1})), ErrParam},
		{"Entropy qk", e(Entropy(x, &EntropyOptions{QK: []float64{1}})), ErrData},
		{"DifferentialEntropy base", e(DifferentialEntropy(x, &DiffEntropyOptions{Base: nan})), ErrParam},
		{"DifferentialEntropy method", e(DifferentialEntropy(x, &DiffEntropyOptions{Method: 9})), ErrParam},
		{"DifferentialEntropy small", e(DifferentialEntropy(x, nil)), ErrData},
		{"Histogram edges", e(Histogram(x, &HistogramOptions{Edges: []float64{1}})), ErrParam},
		{"Histogram edges order", e(Histogram(x, &HistogramOptions{Edges: []float64{2, 1}})), ErrParam},
		{"Histogram rule", e(Histogram(x, &HistogramOptions{Rule: 42})), ErrParam},
		{"Histogram bins", e(Histogram(x, &HistogramOptions{Bins: -2})), ErrParam},
		{"Histogram range order", e(Histogram(x, &HistogramOptions{Range: &[2]float64{2, 1}})), ErrParam},
		{"Histogram range inf", e(Histogram(x, &HistogramOptions{Range: &[2]float64{0, inf}})), ErrParam},
		{"Histogram rule weights", e(Histogram(x, &HistogramOptions{Rule: BinsAuto, Weights: []float64{1, 1, 1, 1}})), ErrParam},
		{"Histogram too many bins", e(Histogram([]float64{1e300}, &HistogramOptions{Bins: 1000, Range: &[2]float64{0, 5e-324}})), ErrParam},
		{"HistogramBinEdges weights", e(HistogramBinEdges(x, &HistogramOptions{Weights: []float64{1}})), ErrData},
		{"CumFreq bins", e(CumFreq(x, &FreqOptions{NumBins: -1})), ErrParam},
		{"CumFreq weights", e(CumFreq(x, &FreqOptions{Weights: []float64{1}})), ErrData},
		{"CumFreq NaN", e(CumFreq([]float64{nan}, nil)), ErrData},
		{"CumFreq negative weight", e(CumFreq(x, &FreqOptions{Weights: []float64{1, -1, 1, 1}})), ErrData},
		{"RelFreq limits", e(RelFreq(x, &FreqOptions{Limits: &[2]float64{1, 1}})), ErrParam},
		{"BinnedStatistic statistic", e(BinnedStatistic(x, x, &BinnedOptions{Statistic: 7})), ErrParam},
		{"BinnedStatistic values", e(BinnedStatistic(x, x[:2], nil)), ErrData},
		{"BinnedStatistic NaN", e(BinnedStatistic([]float64{nan}, []float64{1}, nil)), ErrData},
		{"BinnedStatistic bins", e(BinnedStatistic(x, x, &BinnedOptions{Bins: -1})), ErrParam},
		{"BinnedStatistic range", e(BinnedStatistic(x, x, &BinnedOptions{Range: &[2]float64{2, 1}})), ErrParam},
		{"BinnedStatistic edges", e(BinnedStatistic(x, x, &BinnedOptions{Edges: []float64{1}})), ErrParam},
		{"BinnedStatistic equal edges", e(BinnedStatistic(x, x, &BinnedOptions{Edges: []float64{1, 1, 2}})), ErrParam},
		{"BinnedStatistic infinite edges", e(BinnedStatistic(x, x, &BinnedOptions{Edges: []float64{0, inf}})), ErrParam},
	}
	for _, c := range checks {
		if !errors.Is(c.err, c.want) {
			t.Errorf("%s: error %v, want %v", c.name, c.err, c.want)
		}
	}
	_ = one
}

func TestDescribeCorners(t *testing.T) {
	nan, inf := math.NaN(), math.Inf(1)
	x := []float64{1, 2, 3, 4}
	eq := func(name string, got, want float64) {
		t.Helper()
		if !(got == want || (got != got && want != want)) {
			t.Errorf("%s = %v, want %v", name, got, want)
		}
	}
	v := func(f float64, _ error) float64 { return f }
	for m := QuantileLinear; m <= QuantileNearest; m++ {
		if m.String() != quantileMethodNames[m] {
			t.Errorf("method %d named %q", m, m.String())
		}
	}
	if QuantileMethod(-1).String() != "invalid" {
		t.Error("invalid method name")
	}
	eq("Percentile", v(Percentile(x, 50, nil)), 2.5)
	eq("Quantile NaN", v(Quantile([]float64{1, nan}, 0.5, nil)), nan)
	q, _ := Quantiles([]float64{3, nan, 1, 2}, []float64{0.5}, &QuantileOptions{Method: QuantileInvertedCDF, Weights: []float64{1, 1, 1, 1}, OmitNaN: true})
	eq("weighted nanquantile", q[0], 2)
	// weighted variance: frequency weights
	eq("Var weights", v(Var([]float64{1, 3}, &MeanOptions{Weights: []float64{1, 3}})), 0.75)
	eq("Var weights omit", v(Var([]float64{1, nan, 3}, &MeanOptions{Weights: []float64{1, 2, 3}, OmitNaN: true})), 0.75)
	eq("Mean weights omit", v(Mean([]float64{1, 2, 3}, &MeanOptions{Weights: []float64{1, nan, 3}, OmitNaN: true})), 2.5)
	c := 2.0
	eq("Moment center", v(Moment([]float64{1, nan, 3}, 2, &MomentOptions{Center: &c, OmitNaN: true})), 1)
	eq("Moment NaN", v(Moment([]float64{1, nan, 3}, 2, nil)), nan)
	eq("Kurtosis Pearson", v(Kurtosis([]float64{1, 2, 3, 4, 10}, &ShapeOptions{Pearson: true}))-3, v(Kurtosis([]float64{1, 2, 3, 4, 10}, nil)))
	d1 := 1.0
	eq("Variation ddof=n", v(Variation([]float64{2, 4}, &DdofOptions{Ddof: &[]float64{2}[0]})), inf)
	eq("Variation ddof=n constant", v(Variation([]float64{2, 2}, &DdofOptions{Ddof: &[]float64{2}[0]})), nan)
	eq("SEM omit", v(SEM([]float64{1, nan, 3}, &DdofOptions{Ddof: &d1, OmitNaN: true})), 1)
	eq("GMean omit", v(GMean([]float64{4, nan, 1}, &PowerMeanOptions{OmitNaN: true, Weights: []float64{1, 1, 1}})), 2)
	eq("HMean negative", v(HMean([]float64{-1, 1}, nil)), nan)
	eq("PMean negative", v(PMean([]float64{-1, 1}, 2, nil)), nan)
	eq("PMean small p negative", v(PMean([]float64{-1, 1}, 1e-7, nil)), nan)
	eq("crPow -2^3", crPow(-2, 3), -8)
	eq("crPow 2^-3", crPow(2, -3), 0.125)
	eq("crPow huge", crPow(1e200, 3), inf)
	eq("crPow 0", crPow(0, 3), 0)
	eq("crLog2 8", crLog2(8), 3)
	// entropy corners: zeros, infinite sums
	eq("Entropy zero", v(Entropy([]float64{0, 1, 1}, nil)), math.Ln2)
	eq("Entropy negative", v(Entropy([]float64{-1, 1, 1}, nil)), math.Inf(-1))
	eq("Entropy NaN", v(Entropy([]float64{nan, 1}, nil)), nan)
	eq("rel entropy zero", v(Entropy([]float64{0, 1}, &EntropyOptions{QK: []float64{1, 1}})), math.Ln2)
	eq("rel entropy q zero", v(Entropy([]float64{1, 1}, &EntropyOptions{QK: []float64{0, 1}})), inf)
	eq("rel entropy NaN", v(Entropy([]float64{nan, 1}, &EntropyOptions{QK: []float64{1, 1}})), nan)
	eq("rel entropy overflow", v(Entropy([]float64{1e308, 1e308}, &EntropyOptions{QK: []float64{1e308, 1e308}})), 0)
	eq("rel entropy empty", v(Entropy([]float64{}, &EntropyOptions{QK: []float64{}})), 0)
	eq("DifferentialEntropy NaN", v(DifferentialEntropy([]float64{1, 2, 3, 4, 5, nan}, &DiffEntropyOptions{WindowLength: 1})), nan)
	de, err := DifferentialEntropy([]float64{1, 2, 3, 4, 5, nan}, &DiffEntropyOptions{WindowLength: 1, OmitNaN: true})
	if err != nil || math.IsNaN(de) {
		t.Errorf("DifferentialEntropy omit: %v %v", de, err)
	}
	// histograms: empty data, degenerate range
	h, err := Histogram([]float64{}, nil)
	if err != nil || len(h.Counts) != 10 || h.Edges[10] != 1 {
		t.Errorf("Histogram of nothing: %v %v", h, err)
	}
	h, err = Histogram([]float64{1, 1, 2}, &HistogramOptions{Bins: 2, Range: &[2]float64{1, 1}})
	if err != nil || h.Counts[0] != 0 || h.Counts[1] != 2 {
		t.Errorf("Histogram degenerate range: %v %v", h, err)
	}
	f, err := CumFreq([]float64{}, nil)
	if err != nil || f.LowerLimit != 0 || len(f.Freq) != 10 {
		t.Errorf("CumFreq of nothing: %v %v", f, err)
	}
	// binned statistic with the default bins and a range
	b, err := BinnedStatistic(x, x, nil)
	if err != nil || len(b.Statistic) != 10 {
		t.Errorf("BinnedStatistic: %v %v", b, err)
	}
	b, err = BinnedStatistic(x, []float64{1, nan, 3, 4}, &BinnedOptions{Statistic: BinMin, Range: &[2]float64{0, 5}, Bins: 1})
	if err != nil || b.Statistic[0] != 1 {
		t.Errorf("BinnedStatistic min: %v %v", b, err)
	}
	b, _ = BinnedStatistic(x, []float64{1, nan, 3, 4}, &BinnedOptions{Statistic: BinMax, Bins: 1})
	eq("binned max NaN", b.Statistic[0], nan)
	b, _ = BinnedStatistic(x, []float64{nan, 1, 3, 4}, &BinnedOptions{Statistic: BinMin, Bins: 1})
	eq("binned min NaN first", b.Statistic[0], 1)
	b, _ = BinnedStatistic(x, []float64{nan, 2, 3, 4}, &BinnedOptions{Statistic: BinMedian, Bins: 1})
	eq("binned median NaN", b.Statistic[0], 3.5)
	// trimming corners
	r, _ := TrimBoth([]float64{}, 0.1)
	if len(r) != 0 {
		t.Error("TrimBoth of nothing")
	}
	r, _ = Trim1(x, 1, TailLeft)
	if len(r) != 0 {
		t.Error("Trim1 of everything")
	}
	eq("TrimMean NaN", v(TrimMean([]float64{1, nan}, 0)), nan)
	eq("TMean omit", v(TMean([]float64{1, nan, 3}, &TrimmedOptions{OmitNaN: true})), 2)
	eq("TMin omit", v(TMin([]float64{1, nan, 3}, 2, &BoundOptions{OmitNaN: true})), 3)
	// the array form
	a, _ := ndarray.FromData([]float64{4, 1, 3, 2}, 2, 2)
	z, _ := ZScore(a, nil)
	if s := z.Shape(); len(s) != 2 {
		t.Errorf("ZScore of a 2x2 array has shape %v", s)
	}
	tb, _ := TrimBoth(a, 0.25)
	if tb.Size() != 2 {
		t.Errorf("TrimBoth of an array: %v", tb.Shape())
	}
	if e, _ := HistogramBinEdges(x, nil); len(e) != 11 {
		t.Errorf("HistogramBinEdges: %v", e)
	}
	eq("Median array", v(Median(a, nil)), 2.5)
}

// TestSelection checks the order statistics the selection leaves in place
// against a sort, on random, tied, sorted, reversed and organ-pipe data,
// and the sorting fallback of floydRivest.
func TestSelection(t *testing.T) {
	r := rand.New(rand.NewPCG(1, 2))
	for _, n := range []int{17, 100, 601, 1000, 5000, 20000} {
		gens := map[string]func(i int) float64{
			"random":   func(int) float64 { return r.NormFloat64() },
			"ties":     func(int) float64 { return float64(r.IntN(3)) },
			"sorted":   func(i int) float64 { return float64(i) },
			"reversed": func(i int) float64 { return float64(n - i) },
			"pipe":     func(i int) float64 { return float64(min(i, n-i)) },
		}
		for name, g := range gens {
			v := make([]float64, n)
			for i := range v {
				v[i] = g(i)
			}
			s := slices.Clone(v)
			slices.Sort(s)
			ks := uniqueSorted([]int{0, 1, n / 3, n / 2, n/2 + 1, n - 2, n - 1, r.IntN(n)})
			a := orderStats(v, ks)
			for _, k := range ks {
				if a[k] != s[k] {
					t.Fatalf("%s n=%d: order statistic %d = %v, want %v", name, n, k, a[k], s[k])
				}
				for i := range a {
					if (i < k && a[i] > a[k]) || (i > k && a[i] < a[k]) {
						t.Fatalf("%s n=%d: not partitioned around %d", name, n, k)
					}
				}
			}
		}
	}
	a := []float64{3, 1, 2, 5, 4}
	floydRivest(a, 0, 4, 2, 0)
	if !slices.IsSorted(a) {
		t.Errorf("floydRivest without budget: %v", a)
	}
}
