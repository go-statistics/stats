package stats

import (
	"compress/gzip"
	"embed"
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"reflect"
	"sort"
	"strconv"
	"strings"
	"testing"

	"github.com/go-ndarray/ndarray"
	"github.com/go-ndarray/ndarray/random"
)

// The SciPy parity fixtures (testdata/gen/parity.py): parameterised
// instances of each distribution with SciPy's values in the bulk, its
// moments, fits and random draws.

//go:embed testdata/scipy/dist_*.json.gz
var parityFS embed.FS

var rvsTable = flag.String("rvs.table", "", "write the Rvs bit-identity table (markdown) to this file")

// distEntry is filled by the generated zz_dist_<name>_test.go files.
type distEntry struct {
	typ       reflect.Type
	kind      string
	fit       func(x []float64, o *FitOptions) (any, error)
	construct func(p map[string]float64) (any, error)
}

var distTypes = map[string]distEntry{}

// jf is a float64 in a fixture: a JSON number, or "nan", "inf", "-inf".
type jf float64

func (f *jf) UnmarshalJSON(b []byte) error {
	s := strings.Trim(string(b), `"`)
	v, err := strconv.ParseFloat(s, 64)
	*f = jf(v)
	return err
}

// jmap is a parameter set in a fixture.
type jmap map[string]jf

func (m jmap) floats() map[string]float64 {
	out := map[string]float64{}
	for k, v := range m {
		out[k] = float64(v)
	}
	return out
}

type parityDraws struct {
	Seed  uint64   `json:"seed"`
	Size  any      `json:"size"`
	Shape []int    `json:"shape"`
	Hex   []string `json:"hex"`
	Int   []int64  `json:"int"`
}

type parityInstance struct {
	Params    jmap            `json:"params"`
	Mean      jf              `json:"mean"`
	Var       jf              `json:"var"`
	Std       jf              `json:"std"`
	Skew      jf              `json:"skew"`
	Kurtosis  jf              `json:"kurtosis"`
	Median    jf              `json:"median"`
	Entropy   jf              `json:"entropy"`
	Moments   []jf            `json:"moments"`
	Interval  []jf            `json:"interval"`
	Support   []jf            `json:"support"`
	Points    []map[string]jf `json:"points"`
	Quantiles []map[string]jf `json:"quantiles"`
	Rvs       []parityDraws   `json:"rvs"`
	MPMath    map[string]jf   `json:"mpmath"`
}

type parityFit struct {
	Data   []jf               `json:"data"`
	Fixed  map[string]float64 `json:"fixed"`
	Params map[string]jf      `json:"params"`
	NNLF   jf                 `json:"nnlf"`
}

type parityFile struct {
	SciPy     string           `json:"scipy"`
	Go        string           `json:"go"`
	Kind      string           `json:"kind"`
	Instances []parityInstance `json:"instances"`
	Invalid   []jmap           `json:"invalid"`
	Fit       []parityFit      `json:"fit"`
	FitKind   string           `json:"fit_kind"`
}

func loadParity(t testing.TB) []parityFile {
	t.Helper()
	ents, err := parityFS.ReadDir("testdata/scipy")
	if err != nil {
		t.Fatal(err)
	}
	var out []parityFile
	for _, e := range ents {
		if !strings.HasPrefix(e.Name(), "dist_") {
			continue
		}
		f, err := parityFS.Open("testdata/scipy/" + e.Name())
		if err != nil {
			t.Fatal(err)
		}
		gz, err := gzip.NewReader(f)
		if err != nil {
			t.Fatal(err)
		}
		var pf parityFile
		if err := json.NewDecoder(gz).Decode(&pf); err != nil {
			t.Fatalf("%s: %v", e.Name(), err)
		}
		f.Close()
		out = append(out, pf)
	}
	return out
}

// Tolerances of the parity comparison, relative. SciPy is not the
// reference for accuracy (the ULP fixtures are, against mpmath); these only
// guard against a formula that is wrong, so they absorb SciPy's own
// rounding: parityRtol for the moments, entropy, median, interval and
// support (per Go type: parityTol), parityPointRtol for the function values
// and quantiles at the fixture's points (per type: parityPointTol).
const (
	parityRtol      = 1e-11
	parityPointRtol = 1e-9
)

var parityPointTol = map[string]float64{}

var parityTol = map[string]float64{}

// rvsULP is the largest difference, in ULPs, allowed between a draw and
// SciPy's for the same seed (0: bit-identical), per Go type.
var rvsULP = map[string]float64{}

func closeTo(got, want, rtol float64) bool {
	switch {
	case math.IsNaN(want):
		return math.IsNaN(got)
	case math.IsInf(want, 0):
		return got == want
	case got == want:
		return true
	}
	return math.Abs(got-want) <= rtol*math.Max(math.Abs(want), 1e-300)
}

func asCont(t *testing.T, v any) Continuous {
	t.Helper()
	c, ok := v.(Continuous)
	if !ok {
		t.Fatalf("%T is not Continuous", v)
	}
	return c
}

// distMethods returns the scalar methods common to both kinds, by fixture
// key.
func distFns(v any) map[string]func(float64) float64 {
	m := map[string]func(float64) float64{}
	switch d := v.(type) {
	case Continuous:
		m["pdf"], m["logpdf"], m["cdf"], m["sf"], m["logcdf"], m["logsf"] = d.PDF, d.LogPDF, d.CDF, d.SF, d.LogCDF, d.LogSF
		m["ppf"], m["isf"] = d.PPF, d.ISF
	case Discrete:
		m["pmf"], m["logpmf"], m["cdf"], m["sf"], m["logcdf"], m["logsf"] = d.PMF, d.LogPMF, d.CDF, d.SF, d.LogCDF, d.LogSF
		m["ppf"], m["isf"] = d.PPF, d.ISF
	}
	return m
}

type momenter interface {
	Mean() float64
	Var() float64
	Std() float64
	Skew() float64
	Kurtosis() float64
	Median() float64
	Entropy() float64
	Moment(n int) float64
	Interval(c float64) (float64, float64)
	Support() (float64, float64)
	Rvs(g *random.Generator, size ...int) (*ndarray.Array, error)
	Validate() error
}

func TestParity(t *testing.T) {
	for _, pf := range loadParity(t) {
		t.Run(pf.SciPy, func(t *testing.T) {
			e, ok := distTypes[pf.Go]
			if !ok {
				t.Fatalf("no Go type %s registered", pf.Go)
			}
			if e.kind != pf.Kind {
				t.Fatalf("kind %s, fixture says %s", e.kind, pf.Kind)
			}
			rtol, prtol := parityRtol, parityPointRtol
			if v, ok := parityTol[pf.Go]; ok {
				rtol = v
			}
			if v, ok := parityPointTol[pf.Go]; ok {
				prtol = v
			}
			for i, in := range pf.Instances {
				v, err := e.construct(in.Params.floats())
				if err != nil {
					t.Fatalf("instance %v: %v", in.Params, err)
				}
				name := fmt.Sprintf("%s%v", pf.Go, in.Params)
				d := v.(momenter)
				checkTol := func(what string, got float64, want jf, tol float64) {
					if !closeTo(got, float64(want), tol) {
						t.Errorf("%s %s = %v, SciPy %v", name, what, got, float64(want))
					}
				}
				check := func(what string, got float64, want jf) {
					if exact, ok := in.MPMath[what]; ok {
						// SciPy's value is known to be inaccurate here: hold
						// ours to mpmath's instead.
						if !closeTo(got, float64(exact), 1e-14) {
							t.Errorf("%s %s = %v, mpmath %v (SciPy %v)", name, what, got, float64(exact), float64(want))
						}
						return
					}
					checkTol(what, got, want, rtol)
				}
				check("mean", d.Mean(), in.Mean)
				check("var", d.Var(), in.Var)
				check("std", d.Std(), in.Std)
				check("skew", d.Skew(), in.Skew)
				check("kurtosis", d.Kurtosis(), in.Kurtosis)
				check("median", d.Median(), in.Median)
				check("entropy", d.Entropy(), in.Entropy)
				for n, m := range in.Moments {
					check(fmt.Sprintf("moment(%d)", n), d.Moment(n), m)
				}
				lo, hi := d.Interval(0.95)
				check("interval.lo", lo, in.Interval[0])
				check("interval.hi", hi, in.Interval[1])
				lo, hi = d.Support()
				check("support.lo", lo, in.Support[0])
				check("support.hi", hi, in.Support[1])
				fns := distFns(v)
				for _, pt := range in.Points {
					x := float64(pt["x"])
					for k, want := range pt {
						if k == "x" {
							continue
						}
						checkTol(fmt.Sprintf("%s(%v)", k, x), fns[k](x), want, prtol)
					}
				}
				for _, q := range in.Quantiles {
					p := float64(q["p"])
					checkTol(fmt.Sprintf("ppf(%v)", p), fns["ppf"](p), q["ppf"], prtol)
					checkTol(fmt.Sprintf("isf(%v)", p), fns["isf"](p), q["isf"], prtol)
				}
				_ = i
			}
			for _, p := range pf.Invalid {
				if _, err := e.construct(p.floats()); err == nil {
					t.Errorf("%s%v: constructor accepted invalid parameters", pf.Go, p)
				}
			}
		})
	}
}

// rvsCompare compares the draws for one instance and seed list with
// SciPy's: the number of values, the number that differ, and the largest
// difference in ULPs.
func rvsCompare(t *testing.T, d momenter, in parityInstance) (n, diff int, maxULP float64) {
	var g *random.Generator
	var seed uint64 = math.MaxUint64
	for _, r := range in.Rvs {
		if r.Seed != seed {
			seed = r.Seed
			g = random.DefaultRNG(seed)
		}
		var size []int
		switch s := r.Size.(type) {
		case float64:
			size = []int{int(s)}
		case []any:
			for _, x := range s {
				size = append(size, int(x.(float64)))
			}
		}
		a, err := d.Rvs(g, size...)
		if err != nil {
			t.Fatalf("Rvs: %v", err)
		}
		if !reflect.DeepEqual(append([]int{}, a.Shape()...), append([]int{}, r.Shape...)) {
			t.Fatalf("Rvs shape %v, SciPy %v", a.Shape(), r.Shape)
		}
		if r.Int != nil {
			got, ok := ndarray.Data[int64](a)
			if !ok {
				t.Fatalf("Rvs dtype %v, SciPy int64", a.DType())
			}
			for i, w := range r.Int {
				n++
				if got[i] != w {
					diff++
					maxULP = math.Max(maxULP, math.Abs(float64(got[i]-w)))
				}
			}
			continue
		}
		got, _ := ndarray.Data[float64](a)
		for i, h := range r.Hex {
			w, err := strconv.ParseFloat(h, 64)
			if err != nil {
				t.Fatal(err)
			}
			n++
			if math.Float64bits(got[i]) != math.Float64bits(w) {
				diff++
				maxULP = math.Max(maxULP, math.Abs(got[i]-w)/ulp(w))
			}
		}
	}
	return n, diff, maxULP
}

func TestRvsParity(t *testing.T) {
	type row struct {
		name           string
		n, diff        int
		maxULP         float64
		instances, bad int
	}
	var rows []row
	for _, pf := range loadParity(t) {
		e := distTypes[pf.Go]
		r := row{name: pf.SciPy + " (" + pf.Go + ")"}
		allowed := rvsULP[pf.Go]
		for _, in := range pf.Instances {
			v, err := e.construct(in.Params.floats())
			if err != nil {
				t.Fatal(err)
			}
			n, diff, mu := rvsCompare(t, v.(momenter), in)
			r.n += n
			r.diff += diff
			r.maxULP = math.Max(r.maxULP, mu)
			r.instances++
			if diff > 0 {
				r.bad++
			}
			if mu > allowed {
				t.Errorf("%s%v: %d of %d draws differ from SciPy's, by up to %.3g ulp (allowed %g)", pf.Go, in.Params, diff, n, mu, allowed)
			}
		}
		rows = append(rows, r)
	}
	if *rvsTable == "" {
		return
	}
	sort.Slice(rows, func(i, j int) bool { return rows[i].name < rows[j].name })
	var b strings.Builder
	b.WriteString("<!-- generated by: go test -run TestRvsParity -rvs.table=docs/rvs-table.md -->\n\n")
	b.WriteString("| distribution | draws compared | differing | max difference (ulp, or integer units) | bit-identical |\n")
	b.WriteString("|---|--:|--:|--:|:-:|\n")
	for _, r := range rows {
		ident := "yes"
		if r.diff > 0 {
			ident = "no"
		}
		fmt.Fprintf(&b, "| %s | %d | %d | %s | %s |\n", r.name, r.n, r.diff, fmtULP(r.maxULP), ident)
	}
	if err := os.WriteFile(*rvsTable, []byte(b.String()), 0o644); err != nil {
		t.Fatal(err)
	}
}

func TestFitParity(t *testing.T) {
	for _, pf := range loadParity(t) {
		if len(pf.Fit) == 0 {
			continue
		}
		e := distTypes[pf.Go]
		if e.fit == nil {
			t.Errorf("%s: fixture has fits but Fit%s is not registered", pf.SciPy, pf.Go)
			continue
		}
		for _, c := range pf.Fit {
			x := make([]float64, len(c.Data))
			for i, v := range c.Data {
				x[i] = float64(v)
			}
			got, err := e.fit(x, &FitOptions{Fixed: c.Fixed})
			if err != nil {
				t.Errorf("%s fit (fixed %v): %v", pf.Go, c.Fixed, err)
				continue
			}
			gv := reflect.ValueOf(got)
			nll := 0.0
			switch d := got.(type) {
			case Continuous:
				for _, v := range x {
					nll -= d.LogPDF(v)
				}
			case Discrete:
				for _, v := range x {
					nll -= d.LogPMF(v)
				}
			}
			switch pf.FitKind {
			case "exact":
				for k, want := range c.Params {
					g := gv.FieldByName(k)
					if g.Float() != float64(want) {
						t.Errorf("%s fit %s = %v, SciPy %v (fixed %v)", pf.Go, k, g.Float(), float64(want), c.Fixed)
					}
				}
			case "mle":
				// Our optimum must be at least as good as SciPy's.
				if nll > float64(c.NNLF)+1e-9*math.Abs(float64(c.NNLF)) {
					t.Errorf("%s fit: -loglik %v worse than SciPy's %v (ours %v, SciPy %v)", pf.Go, nll, float64(c.NNLF), got, c.Params)
				}
			default:
				t.Errorf("%s: unknown fit_kind %q", pf.SciPy, pf.FitKind)
			}
		}
	}
}
