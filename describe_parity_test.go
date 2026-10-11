package stats

import (
	"bytes"
	"compress/gzip"
	"crypto/sha256"
	_ "embed"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"math"
	"os"
	"slices"
	"sort"
	"strings"
	"testing"

	"github.com/go-ndarray/ndarray"
	"github.com/go-ndarray/ndarray/random"
)

// The NumPy/SciPy fixtures of the descriptive statistics
// (testdata/gen/scipy_describe.py).

//go:embed testdata/scipy/describe_cases.json.gz
var describeFixture []byte

var describeTable = flag.String("describe.table", "", "write the per-function comparison table (markdown) to this file")

type describeCase struct {
	Fn    string          `json:"fn"`
	Data  string          `json:"data"`
	Args  map[string]any  `json:"args"`
	Error bool            `json:"error"`
	Msg   string          `json:"msg"`
	Exact bool            `json:"exact"`
	Want  json.RawMessage `json:"want"`
	MP    map[string]jf   `json:"mp"`
}

type describeFile struct {
	NumPy    string                    `json:"numpy"`
	SciPy    string                    `json:"scipy"`
	Datasets map[string]map[string]any `json:"datasets"`
	Cases    []describeCase            `json:"cases"`
}

func loadDescribe(t testing.TB) (describeFile, map[string][]float64) {
	t.Helper()
	gz, err := gzip.NewReader(bytes.NewReader(describeFixture))
	if err != nil {
		t.Fatal(err)
	}
	var df describeFile
	if err := json.NewDecoder(gz).Decode(&df); err != nil {
		t.Fatal(err)
	}
	data := map[string][]float64{}
	for name, spec := range df.Datasets {
		v := genDescribeData(t, spec)
		h := sha256.New()
		for _, t := range v {
			u := math.Float64bits(t)
			if t != t {
				u = 0x7ff8000000000000 // NumPy's NaN
			}
			_ = binary.Write(h, binary.LittleEndian, u)
		}
		if got := hex.EncodeToString(h.Sum(nil)); got != spec["sha256"] {
			t.Fatalf("dataset %s: regenerated data differ from NumPy's", name)
		}
		data[name] = v
	}
	return df, data
}

func numF(a any) float64 {
	switch v := a.(type) {
	case float64:
		return v
	case string:
		var f jf
		if err := f.UnmarshalJSON([]byte(v)); err != nil {
			panic(err)
		}
		return float64(f)
	}
	panic(fmt.Sprintf("not a number: %v", a))
}

// genDescribeData mirrors gen() of scipy_describe.py.
func genDescribeData(t testing.TB, s map[string]any) []float64 {
	if vals, ok := s["values"]; ok {
		var out []float64
		for _, x := range vals.([]any) {
			out = append(out, numF(x))
		}
		return out
	}
	n := int(numF(s["n"]))
	rng := random.DefaultRNG(uint64(numF(s["seed"])))
	scale, shift := 1.0, 0.0
	if v, ok := s["scale"]; ok {
		scale = numF(v)
	}
	if v, ok := s["shift"]; ok {
		shift = numF(v)
	}
	draw := func(k int) []float64 {
		a, err := rng.Random(n * k)
		if err != nil {
			t.Fatal(err)
		}
		v, _ := ndarray.Data[float64](a)
		return v
	}
	x := make([]float64, n)
	switch s["kind"] {
	case "unif":
		u := draw(1)
		for i := range x {
			x[i] = float64(u[i]*scale) + shift
		}
	case "ih":
		u := draw(4)
		for i := range x {
			r := u[4*i : 4*i+4]
			sm := r[0] + r[1]
			sm += r[2]
			sm += r[3]
			x[i] = float64((sm-2.0)*scale) + shift
		}
	case "ties":
		u := draw(1)
		k := numF(s["k"])
		for i := range x {
			x[i] = float64(math.Floor(u[i]*k)*scale) + shift
		}
	case "heavy":
		u := draw(2)
		c := numF(s["c"])
		for i := range x {
			v := 1.0 / (u[2*i] + c)
			if u[2*i+1] < 0.5 {
				v = -v
			}
			x[i] = float64(v*scale) + shift
		}
	case "skewed":
		u := draw(2)
		for i := range x {
			a, b := u[2*i], u[2*i+1]
			x[i] = float64(float64(float64(a*a)*b)*scale) + shift
		}
	case "const":
		for i := range x {
			x[i] = numF(s["c"])
		}
	default:
		t.Fatalf("unknown dataset kind %v", s["kind"])
	}
	if at, ok := s["nan_at"]; ok {
		for _, i := range at.([]any) {
			x[int(numF(i))] = math.NaN()
		}
	}
	return x
}

// describeStat accumulates the comparison of one function.
type describeStat struct {
	cases, values, identical, closerExact, errNaN int
	maxRel                                        float64
	exact                                         bool
}

func relDiff(got, want float64) float64 {
	switch {
	case got == want || (got != got && want != want):
		return 0
	case got != got || want != want || math.IsInf(got, 0) || math.IsInf(want, 0):
		return math.Inf(1)
	}
	return math.Abs(got-want) / math.Max(math.Abs(got), math.Abs(want))
}

func sameBits(got, want float64) bool {
	if got != got || want != want {
		return got != got && want != want
	}
	return math.Float64bits(got) == math.Float64bits(want)
}

// describeTol is the relative tolerance of the SciPy comparisons (default
// 1e-13): the moments and logarithms go through pow/log/exp, where SciPy
// calls the C library and we round correctly, and SciPy's own arithmetic
// (pairwise sums of rounded products) loses digits on cancellation; the
// cases beyond the tolerance must then be closer to the exact (mpmath)
// value than SciPy.
var describeTol = map[string]float64{
	// differential entropy: sums of up to 10^5 logarithms of spacings
	"differential_entropy": 1e-12,
	// trimmed mean: NumPy sums the middle part in the order its
	// partition left it, which no other selection reproduces
	"trim_mean": 1e-14,
	// see TestDescribeParity
	"histogram": 1e-14,
}

func TestDescribeParity(t *testing.T) {
	df, data := loadDescribe(t)
	stats := map[string]*describeStat{}
	fails := 0
	for _, c := range df.Cases {
		key := c.Fn
		if _, ok := c.Args["edges"]; ok && c.Fn == "histogram" && c.Args["weights"] != nil {
			// NumPy sums the weights in the order of its (unstable)
			// argsort: ties among the data may be summed in another order
			key, c.Exact = "histogram_weighted_edges", false
		}
		st := stats[key]
		if st == nil {
			st = &describeStat{exact: c.Exact}
			stats[key] = st
		}
		st.cases++
		x := data[c.Data]
		got, err := runDescribe(c, x, data)
		name := fmt.Sprintf("%s(%s, %v)", c.Fn, c.Data, c.Args)
		if c.Error {
			if err == nil {
				t.Errorf("%s: no error; NumPy/SciPy raise %s", name, c.Msg)
				fails++
			}
			continue
		}
		want := decodeWant(c.Want)
		if err != nil {
			if allNaN(want) {
				st.errNaN++
				continue
			}
			t.Errorf("%s: %v; want %v", name, err, want)
			fails++
			continue
		}
		if !errors.Is(compareDescribe(t, name, c, got, want, st, len(x)), nil) {
			fails++
		}
		if fails > 60 {
			t.Fatal("too many failures")
		}
		// the array form gives the same results
		if len(x) <= 9 {
			got2, err2 := runDescribeArray(c, x, data)
			if (err2 == nil) != (err == nil) || fmt.Sprint(got2) != fmt.Sprint(got) {
				t.Errorf("%s: array form %v (%v), slice form %v", name, got2, err2, got)
			}
		}
	}
	names := make([]string, 0, len(stats))
	for k := range stats {
		names = append(names, k)
	}
	sort.Strings(names)
	var b strings.Builder
	fmt.Fprintf(&b, "| function | reference | cases | values | bit-identical | max rel. diff | closer to exact than SciPy | error where SciPy gives NaN |\n|---|---|---|---|---|---|---|---|\n")
	for _, k := range names {
		s := stats[k]
		ref := "SciPy " + df.SciPy
		if s.exact {
			ref = "NumPy " + df.NumPy + " (bit for bit)"
		}
		fmt.Fprintf(&b, "| %s | %s | %d | %d | %d | %.3g | %d | %d |\n", k, ref, s.cases, s.values, s.identical, s.maxRel, s.closerExact, s.errNaN)
	}
	t.Log("\n" + b.String())
	if *describeTable != "" {
		if err := os.WriteFile(*describeTable, []byte(b.String()), 0o644); err != nil {
			t.Fatal(err)
		}
	}
}

// decodeWant turns a fixture result into a float64, a []float64 or a
// map of those.
func decodeWant(raw json.RawMessage) any {
	var a any
	if err := json.Unmarshal(raw, &a); err != nil {
		panic(err)
	}
	var conv func(any) any
	conv = func(a any) any {
		switch v := a.(type) {
		case []any:
			out := make([]float64, len(v))
			for i, e := range v {
				out[i] = numF(e)
			}
			return out
		case map[string]any:
			out := map[string]any{}
			for k, e := range v {
				out[k] = conv(e)
			}
			return out
		}
		return numF(a)
	}
	return conv(a)
}

func allNaN(w any) bool {
	switch v := w.(type) {
	case float64:
		return v != v
	case []float64:
		for _, e := range v {
			if e == e {
				return false
			}
		}
		return true
	case map[string]any:
		for k, e := range v {
			if k != "nobs" && k != "count" && !allNaN(e) {
				return false
			}
		}
		return true
	}
	return false
}

// mpFor returns the exact value of a scalar statistic, when the fixture
// has it.
func mpFor(c describeCase, key string, n int) (float64, bool) {
	if c.MP == nil {
		return 0, false
	}
	pick := func(k string) (float64, bool) {
		v, ok := c.MP[k]
		return float64(v), ok
	}
	if v, ok := c.MP["value"]; ok {
		return float64(v), true
	}
	bias := c.Args["bias"] != false
	switch c.Fn {
	case "skew":
		if bias {
			return pick("skew")
		}
	case "kurtosis":
		if bias && c.Args["fisher"] != false {
			return pick("kurt")
		}
	case "moment":
		if _, ok := c.Args["center"]; !ok {
			return pick(fmt.Sprintf("m%d", int(numF(c.Args["order"]))))
		}
	case "describe":
		switch key {
		case "mean":
			return pick("mean")
		case "skewness":
			if bias {
				return pick("skew")
			}
		case "kurtosis":
			if bias {
				return pick("kurt")
			}
		case "variance":
			m2, ok := pick("m2")
			ddof := numF(c.Args["ddof"])
			return m2 * float64(n) / (float64(n) - ddof), ok
		}
	}
	return 0, false
}

func compareDescribe(t *testing.T, name string, c describeCase, got, want any, st *describeStat, n int) error {
	t.Helper()
	var bad error
	cmp := func(key string, g, w float64) {
		st.values++
		if sameBits(g, w) {
			st.identical++
			return
		}
		if c.Exact {
			bad = fmt.Errorf("mismatch")
			t.Errorf("%s[%s] = %v (%x), NumPy %v (%x)", name, key, g, math.Float64bits(g), w, math.Float64bits(w))
			return
		}
		r := relDiff(g, w)
		st.maxRel = math.Max(st.maxRel, r)
		tol := 1e-13
		if v, ok := describeTol[c.Fn]; ok {
			tol = v
		}
		if r <= tol {
			return
		}
		if ex, ok := mpFor(c, key, n); ok && math.Abs(g-ex) <= math.Abs(w-ex) {
			st.closerExact++
			return
		}
		bad = fmt.Errorf("mismatch")
		t.Errorf("%s[%s] = %v, SciPy %v (rel. diff %.3g)", name, key, g, w, r)
	}
	var walk func(key string, g, w any)
	walk = func(key string, g, w any) {
		switch wv := w.(type) {
		case float64:
			gv, ok := g.(float64)
			if !ok {
				t.Errorf("%s[%s]: got %T, want a number", name, key, g)
				bad = fmt.Errorf("type")
				return
			}
			cmp(key, gv, wv)
		case []float64:
			gv, ok := g.([]float64)
			if !ok || len(gv) != len(wv) {
				t.Errorf("%s[%s]: got %v, want %v", name, key, g, wv)
				bad = fmt.Errorf("len")
				return
			}
			for i := range wv {
				cmp(fmt.Sprintf("%s%d", key, i), gv[i], wv[i])
			}
		case map[string]any:
			gm, ok := g.(map[string]any)
			if !ok {
				t.Errorf("%s: got %T, want a map", name, g)
				bad = fmt.Errorf("type")
				return
			}
			for k, e := range wv {
				gk, ok := gm[k]
				if !ok {
					t.Errorf("%s: no %q in the result", name, k)
					bad = fmt.Errorf("key")
					continue
				}
				walk(k, gk, e)
			}
		}
	}
	walk("", got, want)
	return bad
}

func runDescribeArray(c describeCase, x []float64, data map[string][]float64) (any, error) {
	a, err := ndarray.FromData(slices.Clone(x), len(x))
	if err != nil {
		return nil, err
	}
	return describeCall(c, a, data)
}

func runDescribe(c describeCase, x []float64, data map[string][]float64) (any, error) {
	return describeCall(c, slices.Clone(x), data)
}

// describeCall dispatches a fixture case to the Go function on either form
// of the data.
func describeCall[D Data](c describeCase, x D, data map[string][]float64) (any, error) {
	args := c.Args
	has := func(k string) bool { v, ok := args[k]; return ok && v != nil }
	f := func(k string) float64 { return numF(args[k]) }
	fs := func(k string) []float64 {
		var out []float64
		for _, e := range args[k].([]any) {
			out = append(out, numF(e))
		}
		return out
	}
	pf := func(k string) *float64 {
		if !has(k) {
			return nil
		}
		v := f(k)
		return &v
	}
	b := func(k string, def bool) bool {
		if v, ok := args[k].(bool); ok {
			return v
		}
		return def
	}
	weights := func() []float64 {
		if !has("weights") {
			return nil
		}
		return data[args["weights"].(string)]
	}
	method := func() QuantileMethod {
		m, _ := args["method"].(string)
		for i, s := range quantileMethodNames {
			if s == m {
				return QuantileMethod(i)
			}
		}
		return QuantileLinear
	}
	scalar := func(v float64, err error) (any, error) { return v, err }
	vec := func(v []float64, err error) (any, error) { return v, err }
	asVec := func(v D, err error) (any, error) { return floats(v), err }
	sortedVec := func(v D, err error) (any, error) {
		s := slices.Clone(floats(v))
		slices.SortFunc(s, func(p, q float64) int {
			switch {
			case lessNaN(p, q):
				return -1
			case lessNaN(q, p):
				return 1
			}
			return 0
		})
		return s, err
	}
	switch c.Fn {
	case "mean":
		return scalar(Mean(x, nil))
	case "nanmean":
		return scalar(Mean(x, &MeanOptions{OmitNaN: true}))
	case "var":
		return scalar(Var(x, &MeanOptions{Ddof: f("ddof")}))
	case "std":
		return scalar(Std(x, &MeanOptions{Ddof: f("ddof")}))
	case "nanvar":
		return scalar(Var(x, &MeanOptions{Ddof: f("ddof"), OmitNaN: true}))
	case "median":
		return scalar(Median(x, nil))
	case "nanmedian":
		return scalar(Median(x, &MedianOptions{OmitNaN: true}))
	case "average":
		a, s, err := Average(x, &MeanOptions{Weights: weights()})
		return []float64{a, s}, err
	case "quantile_w":
		w := weights()
		if args["weights"] == "int" {
			n := len(floats(x))
			if n == 1000 {
				w = data["intweights1000"]
			} else {
				w = mapF(data["weights100000"], func(t float64) float64 { return math.Floor(t * 4) })
			}
		}
		return vec(Quantiles(x, fs("q"), &QuantileOptions{Method: QuantileInvertedCDF, Weights: w}))
	case "quantile":
		return vec(Quantiles(x, fs("q"), &QuantileOptions{Method: method()}))
	case "nanquantile":
		return vec(Quantiles(x, fs("q"), &QuantileOptions{Method: method(), OmitNaN: true}))
	case "percentile":
		return vec(Percentiles(x, fs("q"), &QuantileOptions{Method: method()}))
	case "quantile1":
		return scalar(Quantile(x, f("q"), &QuantileOptions{Method: method()}))
	case "histogram", "histogram_bin_edges":
		o := &HistogramOptions{Density: b("density", false), Weights: weights()}
		if has("bins") {
			o.Bins = int(f("bins"))
		}
		if has("edges") {
			o.Edges = fs("edges")
		}
		if has("range") {
			r := fs("range")
			o.Range = &[2]float64{r[0], r[1]}
		}
		if has("rule") {
			for r, s := range []string{"", "auto", "fd", "doane", "scott", "stone", "rice", "sturges", "sqrt"} {
				if s == args["rule"] {
					o.Rule = BinRule(r)
				}
			}
		}
		if c.Fn == "histogram_bin_edges" {
			return vec(HistogramBinEdges(x, o))
		}
		h, err := Histogram(x, o)
		return map[string]any{"counts": h.Counts, "edges": h.Edges}, err
	case "describe", "describe_omit":
		o := &ShapeOptions{Unbiased: !b("bias", true), Ddof: pf("ddof"), OmitNaN: c.Fn == "describe_omit"}
		r, err := Describe(x, o)
		return map[string]any{"nobs": float64(r.NObs), "min": r.Min, "max": r.Max, "mean": r.Mean,
			"variance": r.Variance, "skewness": r.Skewness, "kurtosis": r.Kurtosis}, err
	case "skew", "skew_omit":
		return scalar(Skew(x, &ShapeOptions{Unbiased: !b("bias", true), OmitNaN: c.Fn == "skew_omit"}))
	case "kurtosis":
		return scalar(Kurtosis(x, &ShapeOptions{Unbiased: !b("bias", true), Pearson: !b("fisher", true)}))
	case "moment":
		return scalar(Moment(x, int(f("order")), &MomentOptions{Center: pf("center")}))
	case "mode", "mode_omit":
		r, err := Mode(x, &NaNOptions{OmitNaN: c.Fn == "mode_omit"})
		return map[string]any{"mode": r.Mode, "count": float64(r.Count)}, err
	case "sem", "sem_omit":
		return scalar(SEM(x, &DdofOptions{Ddof: pf("ddof"), OmitNaN: c.Fn == "sem_omit"}))
	case "variation", "variation_omit":
		return scalar(Variation(x, &DdofOptions{Ddof: pf("ddof"), OmitNaN: c.Fn == "variation_omit"}))
	case "zscore", "zscore_omit":
		return asVec(ZScore(x, &DdofOptions{Ddof: pf("ddof"), OmitNaN: c.Fn == "zscore_omit"}))
	case "zmap":
		return vec(ZMap([]float64{-1.0, 0.0, 0.5, 2.25, 1e3}, x, &DdofOptions{Ddof: pf("ddof")}))
	case "iqr", "iqr_omit":
		o := &IQROptions{Method: method(), OmitNaN: c.Fn == "iqr_omit"}
		if has("rng") {
			r := fs("rng")
			o.Range = [2]float64{r[0], r[1]}
		}
		switch s := args["scale"].(type) {
		case string:
			o.Scale = IQRNormal
		case float64:
			o.Scale = s
		}
		return scalar(IQR(x, o))
	case "mad", "mad_omit":
		o := &MADOptions{OmitNaN: c.Fn == "mad_omit"}
		if args["scale"] == "normal" {
			o.Scale = MADNormal
		}
		if args["center"] == "mean" {
			o.Center = mean
		}
		return scalar(MedianAbsDeviation(x, o))
	case "tmean", "tvar", "tstd", "tsem":
		o := &TrimmedOptions{Ddof: pf("ddof")}
		if has("limits") {
			l := fs("limits")
			inc := args["inclusive"].([]any)
			o.Limits = &Limits{Lower: l[0], Upper: l[1], LowerExclusive: inc[0] == false, UpperExclusive: inc[1] == false}
		}
		switch c.Fn {
		case "tmean":
			return scalar(TMean(x, o))
		case "tvar":
			return scalar(TVar(x, o))
		case "tstd":
			return scalar(TStd(x, o))
		}
		return scalar(TSem(x, o))
	case "tmin", "tmax":
		o := &BoundOptions{Exclusive: has("limit") && !b("inclusive", true)}
		if c.Fn == "tmin" {
			l := math.Inf(-1)
			if has("limit") {
				l = f("limit")
			}
			return scalar(TMin(x, l, o))
		}
		l := math.Inf(1)
		if has("limit") {
			l = f("limit")
		}
		return scalar(TMax(x, l, o))
	case "trim_mean":
		return scalar(TrimMean(x, f("p")))
	case "trimboth":
		return sortedVec(TrimBoth(x, f("p")))
	case "trim1":
		tail := TailRight
		if args["tail"] == "left" {
			tail = TailLeft
		}
		return sortedVec(Trim1(x, f("p"), tail))
	case "gmean", "gmean_omit":
		return scalar(GMean(x, &PowerMeanOptions{Weights: weights(), OmitNaN: c.Fn == "gmean_omit"}))
	case "hmean":
		return scalar(HMean(x, &PowerMeanOptions{Weights: weights()}))
	case "pmean":
		return scalar(PMean(x, f("p"), &PowerMeanOptions{Weights: weights()}))
	case "entropy":
		o := &EntropyOptions{}
		if has("base") {
			o.Base = f("base")
		}
		if has("qk") {
			o.QK = slices.Clone(floats(x))
			slices.Reverse(o.QK)
		}
		return scalar(Entropy(x, o))
	case "differential_entropy":
		o := &DiffEntropyOptions{}
		for i, s := range []string{"auto", "vasicek", "van es", "ebrahimi", "correa"} {
			if s == args["method"] {
				o.Method = DiffEntropyMethod(i)
			}
		}
		if has("window_length") {
			o.WindowLength = int(f("window_length"))
		}
		if has("base") {
			o.Base = f("base")
		}
		return scalar(DifferentialEntropy(x, o))
	case "cumfreq", "relfreq":
		o := &FreqOptions{NumBins: int(f("numbins")), Weights: weights()}
		if has("limits") {
			l := fs("limits")
			o.Limits = &[2]float64{l[0], l[1]}
		}
		fn := CumFreq[D]
		if c.Fn == "relfreq" {
			fn = RelFreq[D]
		}
		r, err := fn(x, o)
		return map[string]any{"freq": r.Freq, "lower": r.LowerLimit, "binsize": r.BinSize, "extra": float64(r.ExtraPoints)}, err
	case "binned_statistic":
		v := floats(x)
		vals := make([]float64, len(v))
		for i, t := range v {
			vals[i] = float64(t*2.0) + 1.0
		}
		o := &BinnedOptions{}
		for i, s := range []string{"mean", "std", "median", "count", "sum", "min", "max"} {
			if s == args["statistic"] {
				o.Statistic = BinStatistic(i)
			}
		}
		switch bv := args["bins"].(type) {
		case float64:
			o.Bins = int(bv)
		default:
			o.Edges = fs("bins")
		}
		r, err := BinnedStatistic(x, vals, o)
		bn := make([]float64, len(r.BinNumber))
		for i, k := range r.BinNumber {
			bn[i] = float64(k)
		}
		if len(v) > 129 {
			bn = []float64{}
		}
		return map[string]any{"statistic": r.Statistic, "edges": r.Edges, "binnumber": bn}, err
	}
	return nil, fmt.Errorf("unknown fixture function %q", c.Fn)
}
