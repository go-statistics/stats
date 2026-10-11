package stats

import (
	"compress/gzip"
	"embed"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"math"
	"os"
	"sort"
	"strings"
	"testing"

	"github.com/go-ndarray/ndarray"
)

// The SciPy fixtures of the hypothesis tests (testdata/gen/scipy_tests.py):
// every case is replayed through the Go function and each output field
// compared with SciPy's.

//go:embed testdata/scipy/tests_*.json.gz
var testsFS embed.FS

var testsReport = flag.String("tests.report", "", "write the per-function maximum relative differences against SciPy (markdown) to this file")

type testCase struct {
	Fn    string                     `json:"fn"`
	Data  []int                      `json:"data"`
	Opts  map[string]any             `json:"opts"`
	Out   map[string]json.RawMessage `json:"out"`
	Error bool                       `json:"error"`
	Crash bool                       `json:"crash"`

	file string
	data [][]float64
}

type testsFile struct {
	Datasets [][]jf    `json:"datasets"`
	Cases    []testCase `json:"cases"`
}

func loadTestCases(t *testing.T) []testCase {
	t.Helper()
	names, err := testsFS.ReadDir("testdata/scipy")
	if err != nil {
		t.Fatal(err)
	}
	var all []testCase
	for _, e := range names {
		if !strings.HasPrefix(e.Name(), "tests_") {
			continue
		}
		fh, err := testsFS.Open("testdata/scipy/" + e.Name())
		if err != nil {
			t.Fatal(err)
		}
		zr, err := gzip.NewReader(fh)
		if err != nil {
			t.Fatal(err)
		}
		b, err := io.ReadAll(zr)
		fh.Close()
		if err != nil {
			t.Fatal(err)
		}
		var f testsFile
		if err := json.Unmarshal(b, &f); err != nil {
			t.Fatalf("%s: %v", e.Name(), err)
		}
		ds := make([][]float64, len(f.Datasets))
		for i, d := range f.Datasets {
			ds[i] = make([]float64, len(d))
			for k, v := range d {
				ds[i][k] = float64(v)
			}
		}
		for _, c := range f.Cases {
			c.file = e.Name()
			for _, i := range c.Data {
				c.data = append(c.data, ds[i])
			}
			all = append(all, c)
		}
	}
	return all
}

func (c testCase) str(k string) string {
	s, _ := c.Opts[k].(string)
	return s
}

func (c testCase) num(k string) float64 {
	switch v := c.Opts[k].(type) {
	case float64:
		return v
	case string:
		f, _ := jfParse(v)
		return f
	}
	return 0
}

func (c testCase) flag(k string) bool {
	b, _ := c.Opts[k].(bool)
	return b
}

func jfParse(s string) (float64, error) {
	var f jf
	err := f.UnmarshalJSON([]byte(`"` + s + `"`))
	return float64(f), err
}

func (c testCase) alt() Alternative {
	switch c.str("alternative") {
	case "less":
		return Less
	case "greater":
		return Greater
	}
	return TwoSided
}

func (c testCase) method() Method {
	switch c.str("method") {
	case "exact":
		return MethodExact
	case "asymptotic", "asymp":
		return MethodAsymptotic
	case "approx":
		return MethodApprox
	}
	return MethodAuto
}

// outputs is what a Go function produced, by SciPy's field names.
type outputs map[string]any // float64 or []float64

// testRunners maps each fixture function to its Go counterpart.
var testRunners = map[string]func(c testCase) (outputs, error){}

func registerTestRunner(fn string, r func(c testCase) (outputs, error)) {
	testRunners[fn] = r
}

// testTol is the relative tolerance per function and field (default
// 1e-12); entries are justified where they are set.
var testTol = map[string]float64{}

func tolFor(fn, field string) float64 {
	if v, ok := testTol[fn+"."+field]; ok {
		return v
	}
	if v, ok := testTol[fn]; ok {
		return v
	}
	return 1e-12
}

// relDiff is |got-want|/|want|, 0 when both are equal (including NaN and
// infinities of the same sign), +Inf when one is NaN or infinite and the
// other is not; when want is 0 it is |got| (an absolute difference).
func relDiff(got, want float64) float64 {
	switch {
	case got == want || (got != got && want != want):
		return 0
	case got != got || want != want || math.IsInf(got, 0) || math.IsInf(want, 0):
		return math.Inf(1)
	case want == 0:
		return math.Abs(got)
	}
	return math.Abs(got-want) / math.Abs(want)
}

type diffStat struct {
	n       int
	max     float64
	where   string
	got     float64
	want    float64
	tol     float64
	failing int
}

func TestScipyTests(t *testing.T) {
	cases := loadTestCases(t)
	if len(cases) == 0 {
		t.Fatal("no fixtures")
	}
	stats := map[string]*diffStat{}
	for i, c := range cases {
		run, ok := testRunners[c.Fn]
		if !ok {
			t.Errorf("%s: no Go runner for %s", c.file, c.Fn)
			continue
		}
		name := fmt.Sprintf("%s#%d %s %v", c.file, i, c.Fn, c.Opts)
		out, err := run(c)
		if c.Error {
			if err == nil {
				t.Errorf("%s: SciPy raised ValueError, Go returned no error (%v)", name, out)
			} else if !errors.Is(err, ErrData) && !errors.Is(err, ErrParam) {
				t.Errorf("%s: error %v wraps neither ErrData nor ErrParam", name, err)
			}
			continue
		}
		if err != nil {
			t.Errorf("%s: unexpected error %v", name, err)
			continue
		}
		if c.Crash {
			// SciPy fails with ZeroDivisionError; the Go function returns NaN.
			if p, _ := out["pvalue"].(float64); p == p {
				t.Errorf("%s: SciPy crashes here, expected a NaN p-value, got %v", name, p)
			}
			continue
		}
		for field, raw := range c.Out {
			g, ok := out[field]
			if !ok {
				t.Errorf("%s: Go result has no field %q", name, field)
				continue
			}
			var want []float64
			var w jf
			if json.Unmarshal(raw, &w) == nil {
				want = []float64{float64(w)}
			} else {
				var ws []jf
				if err := json.Unmarshal(raw, &ws); err != nil {
					t.Fatalf("%s: field %s: %v", name, field, err)
				}
				for _, v := range ws {
					want = append(want, float64(v))
				}
			}
			var got []float64
			switch v := g.(type) {
			case float64:
				got = []float64{v}
			case []float64:
				got = v
			}
			if len(got) != len(want) {
				t.Errorf("%s: field %s: %d values, SciPy has %d", name, field, len(got), len(want))
				continue
			}
			key := c.Fn + "." + field
			if strings.HasPrefix(field, "ci_") {
				key = c.Fn + ".ci"
			}
			st := stats[key]
			if st == nil {
				st = &diffStat{tol: tolFor(c.Fn, field)}
				stats[key] = st
			}
			for k := range got {
				d := relDiff(got[k], want[k])
				st.n++
				if d > st.max || (st.where == "" && d >= st.max) {
					st.max, st.where, st.got, st.want = d, fmt.Sprintf("%s [%d]", name, k), got[k], want[k]
				}
				if d > st.tol {
					st.failing++
					if st.failing <= 5 {
						t.Errorf("%s: field %s[%d]: got %v, SciPy %v (rel. diff %.3g > %g)", name, field, k, got[k], want[k], d, st.tol)
					}
				}
			}
		}
	}
	keys := make([]string, 0, len(stats))
	for k := range stats {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	var b strings.Builder
	fmt.Fprintf(&b, "| function.field | values | max rel. diff | tolerance | worst case |\n|---|---:|---:|---:|---|\n")
	for _, k := range keys {
		s := stats[k]
		fmt.Fprintf(&b, "| %s | %d | %.3g | %g | %s: %v vs %v |\n", k, s.n, s.max, s.tol, s.where, s.got, s.want)
		if s.failing > 5 {
			t.Errorf("%s: %d values outside tolerance", k, s.failing)
		}
	}
	if *testsReport != "" {
		if err := os.WriteFile(*testsReport, []byte(b.String()), 0o644); err != nil {
			t.Fatal(err)
		}
	}
}

// TestScipyTestsCovered checks that every registered runner has fixtures.
func TestScipyTestsCovered(t *testing.T) {
	seen := map[string]bool{}
	for _, c := range loadTestCases(t) {
		seen[c.Fn] = true
	}
	for fn := range testRunners {
		if !seen[fn] {
			t.Errorf("runner %s has no fixture", fn)
		}
	}
}

// arr turns a dataset into a 1-d array, to exercise the *ndarray.Array
// form of Data in some runners.
func arr(v []float64) *ndarray.Array {
	a, _ := ndarray.FromData(append([]float64(nil), v...), len(v))
	return a
}
