package bench

import (
	"context"
	"crypto/sha256"
	"embed"
	"encoding/binary"
	"encoding/hex"
	"flag"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"testing"

	"github.com/go-ndarray/ndarray"
	"github.com/go-ndarray/ndarray/random"
	"github.com/go-scikit/refbench"
	"github.com/go-scikit/refbench/rbtest"
	. "github.com/go-statistics/stats"
)

// The speed comparison with scipy.stats 1.18.1 (docs/perf.md).
//
//	cd bench
//	go test -run TestPerf -cpu 1 .                   CI mode: allocations, table, inventory
//	go test -run TestPerf -cpu 1 -perf.run .         measure (refuses at a 1-minute load of 8 or more)
//	go test -run TestPerf -perf.record .             allocation counts only (a deferred report)
//	go test -run TestDataParity -perf.parity .       both sides get the same bytes (needs uv)
//
// It replaces the harness of TestPerfReport (one python -c process, the Go
// and SciPy calls timed back to back): the cases, their SciPy expressions and
// their data are those of that harness, run by refbench (one core on both sides,
// interleaved rounds, a load gate). What is compared.
//
//   - A distribution's elementwise row ("norm.pdf") times a Go loop over n
//     points calling the scalar function, against ONE ufunc call over the same
//     n points (the result array allocated by the call, as the Go side allocates
//     its own). The array API of this package is measured by the "Map/..." rows.
//   - Every other row is the call a user makes, on the same data.
//   - n is the size of the previous harness, capped (ncap_test.go): at most
//     100000 and so that one Go call takes at most 8 ms, because the allocation
//     count of the CI mode makes 51 calls. Both sides use the same n.
//   - Parity is "none": the oracle of the values is the ulp fixtures
//     (docs/accuracy.md), SciPy's own errors reaching 1e15 ulp on some functions.

var parityFlag = flag.Bool("perf.parity", false, "TestDataParity: start the SciPy worker through uv")

const gold = 0.6180339887498949

// points is lo + (hi-lo)·frac(i·gold): the same IEEE operations as pyPoints (the
// explicit float64 conversion forbids a fused multiply-add).
func points(n int, lo, hi float64) []float64 {
	v := make([]float64, n)
	for i := range v {
		f := math.Mod(float64(i)*gold, 1)
		v[i] = lo + float64((hi-lo)*f)
	}
	return v
}

func floored(x []float64) []float64 {
	for i := range x {
		x[i] = math.Floor(x[i])
	}
	return x
}

func normalDraws(n int, seed uint64) []float64 {
	a, _ := random.DefaultRNG(seed).StandardNormal(n)
	v, _ := ndarray.Data[float64](a)
	return v
}

func positive(n int, seed uint64) []float64 {
	x := normalDraws(n, seed)
	for i := range x {
		x[i] = math.Abs(x[i]) + 0.5
	}
	return x
}

func counts(n int) []float64 { return floored(points(n, 50, 150)) }

func arr(x []float64, shape ...int) *ndarray.Array {
	a, err := ndarray.FromData(x, shape...)
	if err != nil {
		panic(err)
	}
	return a
}

var sink float64

// row is one case: its Go side and its SciPy side on the same data.
type row struct {
	name, fn, ref string
	n             int
	// serial, when set, is the Go loop over n points calling the scalar
	// function (an elementwise row); call is then not used.
	serial func(n int) func()
	// call builds the user's call on data of size n (outside the timed body).
	call func(n int) func()
	// py is the SciPy side: the argument tuple and the callable.
	py pyCase
	// data builds the arguments as the Go side sees them, to hash against the
	// Python arguments (nil: see why).
	data func(n int) [][]float64
	why  string
}

type pyCase struct{ setup, call string }

func (r row) size() int {
	if m, ok := nCap[r.name]; ok {
		return m
	}
	// logcdf and logsf cost what cdf and sf cost.
	for _, s := range [][2]string{{".logcdf", ".cdf"}, {".logsf", ".sf"}} {
		if p, ok := strings.CutSuffix(r.name, s[0]); ok {
			if m, ok := nCap[p+s[1]]; ok {
				return m
			}
		}
	}
	return r.n
}

// ---- the SciPy side: Python expressions for the data (the same IEEE operations as points and normalDraws).

func pyPoints(lo, hi string) string {
	return fmt.Sprintf("((%s) + ((%s) - (%s)) * np.mod(np.arange(n, dtype=np.float64) * 0.6180339887498949, 1.0))", lo, hi, lo)
}

func pyNormal(seed int) string {
	return fmt.Sprintf("np.random.default_rng(%d).standard_normal(n)", seed)
}

func pyPositive(seed int) string { return "(np.abs(" + pyNormal(seed) + ") + 0.5)" }

var (
	pyCounts = "np.floor(" + pyPoints("50", "150") + ")"
	pyN1     = "(" + pyNormal(1) + ",)"
	pyN2     = "(" + pyNormal(1) + ", " + pyNormal(2) + ")"
	pyN3     = "(" + pyNormal(1) + ", " + pyNormal(2) + ", " + pyNormal(3) + ")"
	pyMVN    = "st.multivariate_normal(np.array([0, 1, -1, 0.5, 2.0]), np.array([[2, 0.5, 0.3, 0.1, 0], [0.5, 1.5, 0.2, 0, 0.1], [0.3, 0.2, 1, 0.4, 0.2], [0.1, 0, 0.4, 2.5, 0.3], [0, 0.1, 0.2, 0.3, 1.2]]))"
	pyMVNX   = pyMVN + ".rvs(size=n, random_state=np.random.default_rng(8))"
	pyAlpha  = "np.array([1.5, 2, 3.5, 0.8, 4])"
	pyMNP    = "np.array([0.1, 0.2, 0.3, 0.15, 0.25])"
)

// pyNum formats a float64 so that Python reads the same double (an int where it
// is integral: hypergeom's shapes must be integers).
func pyNum(v float64) string {
	if v == math.Trunc(v) && math.Abs(v) < 1<<53 {
		return strconv.FormatInt(int64(v), 10)
	}
	return strconv.FormatFloat(v, 'g', -1, 64)
}

func pyShapes(a []float64) string {
	s := make([]string, len(a))
	for i, v := range a {
		s[i] = pyNum(v)
	}
	return strings.Join(s, ", ")
}

func pyWithShapes(head string, a []float64) string {
	if len(a) == 0 {
		return head
	}
	return head + ", " + pyShapes(a)
}

func sep(a []float64) string {
	if len(a) == 0 {
		return ""
	}
	return ", "
}

// ---- the distributions.

// perfDist is one distribution of the generic rows: the Go value, SciPy's name
// and shape arguments, the x range of the elementwise rows, and whether it is
// discrete (x is then floor(points)).
type perfDist struct {
	d        any
	name     string
	args     []float64
	lo, hi   float64
	discrete bool
}

var perfDists = []perfDist{
	{Beta{A: 2.5, B: 7, Scale: 1}, "beta", []float64{2.5, 7}, 1e-6, 1 - 1e-6, false},
	{Cauchy{Scale: 1}, "cauchy", nil, -50, 50, false},
	{Chi{Df: 4.5, Scale: 1}, "chi", []float64{4.5}, 1e-3, 6, false},
	{ChiSquared{Df: 12, Scale: 1}, "chi2", []float64{12}, 1e-3, 100, false},
	{Exponential{Scale: 1}, "expon", nil, 0, 20, false},
	{F{Dfn: 5, Dfd: 12, Scale: 1}, "f", []float64{5, 12}, 1e-3, 10, false},
	{Gamma{A: 4.5, Scale: 1}, "gamma", []float64{4.5}, 1e-3, 30, false},
	{Gumbel{Scale: 1}, "gumbel_r", nil, -3, 10, false},
	{HalfNormal{Scale: 1}, "halfnorm", nil, 0, 6, false},
	{InverseGamma{A: 3.5, Scale: 1}, "invgamma", []float64{3.5}, 1e-3, 10, false},
	{Laplace{Scale: 1}, "laplace", nil, -15, 15, false},
	{Logistic{Scale: 1}, "logistic", nil, -20, 20, false},
	{LogNormal{S: 0.8, Scale: 1}, "lognorm", []float64{0.8}, 1e-3, 20, false},
	{Lomax{C: 2.5, Scale: 1}, "lomax", []float64{2.5}, 0, 30, false},
	{NoncentralChiSquared{Df: 4, Nc: 3.5, Scale: 1}, "ncx2", []float64{4, 3.5}, 1e-3, 40, false},
	{Normal{Scale: 1}, "norm", nil, -10, 10, false},
	{Pareto{B: 2.5, Scale: 1}, "pareto", []float64{2.5}, 1, 50, false},
	{Rayleigh{Scale: 1}, "rayleigh", nil, 0, 6, false},
	{Reciprocal{A: 0.5, B: 20, Scale: 1}, "reciprocal", []float64{0.5, 20}, 0.5, 20, false},
	{StudentT{Df: 7.5, Scale: 1}, "t", []float64{7.5}, -20, 20, false},
	{Triangular{C: 0.3, Scale: 1}, "triang", []float64{0.3}, 0, 1, false},
	{Uniform{Scale: 1}, "uniform", nil, 0, 1, false},
	{VonMises{Kappa: 2.5, Scale: 1}, "vonmises", []float64{2.5}, -math.Pi, math.Pi, false},
	{Weibull{C: 1.7, Scale: 1}, "weibull_min", []float64{1.7}, 0, 6, false},
	{Bernoulli{P: 0.3}, "bernoulli", []float64{0.3}, 0, 2, true},
	{Binomial{N: 30, P: 0.3}, "binom", []float64{30, 0.3}, 0, 31, true},
	{DiscreteUniform{Low: 0, High: 50}, "randint", []float64{0, 50}, -1, 51, true},
	{Geometric{P: 0.2}, "geom", []float64{0.2}, 1, 60, true},
	{Hypergeometric{Total: 100, Good: 30, Draws: 20}, "hypergeom", []float64{100, 30, 20}, 0, 21, true},
	{NegativeBinomial{N: 5.5, P: 0.4}, "nbinom", []float64{5.5, 0.4}, 0, 60, true},
	{Poisson{Mu: 7.5}, "poisson", []float64{7.5}, 0, 30, true},
	{Zipf{A: 2.5}, "zipf", []float64{2.5}, 1, 200, true},
}

type rvser interface {
	Rvs(g *random.Generator, size ...int) (*ndarray.Array, error)
}

// fits maps SciPy's name to this package's fit of that family.
var fits = map[string]func(x []float64){
	"beta":        func(x []float64) { FitBeta(x, nil) },
	"cauchy":      func(x []float64) { FitCauchy(x, nil) },
	"chi":         func(x []float64) { FitChi(x, nil) },
	"chi2":        func(x []float64) { FitChiSquared(x, nil) },
	"expon":       func(x []float64) { FitExponential(x, nil) },
	"f":           func(x []float64) { FitF(x, nil) },
	"gamma":       func(x []float64) { FitGamma(x, nil) },
	"gumbel_r":    func(x []float64) { FitGumbel(x, nil) },
	"halfnorm":    func(x []float64) { FitHalfNormal(x, nil) },
	"invgamma":    func(x []float64) { FitInverseGamma(x, nil) },
	"laplace":     func(x []float64) { FitLaplace(x, nil) },
	"logistic":    func(x []float64) { FitLogistic(x, nil) },
	"lognorm":     func(x []float64) { FitLogNormal(x, nil) },
	"lomax":       func(x []float64) { FitLomax(x, nil) },
	"norm":        func(x []float64) { FitNormal(x, nil) },
	"pareto":      func(x []float64) { FitPareto(x, nil) },
	"rayleigh":    func(x []float64) { FitRayleigh(x, nil) },
	"reciprocal":  func(x []float64) { FitReciprocal(x, nil) },
	"t":           func(x []float64) { FitStudentT(x, nil) },
	"triang":      func(x []float64) { FitTriangular(x, nil) },
	"uniform":     func(x []float64) { FitUniform(x, nil) },
	"vonmises":    func(x []float64) { FitVonMises(x, nil) },
	"weibull_min": func(x []float64) { FitWeibull(x, nil) },
}

var fitFuncs = map[string]string{
	"beta": "FitBeta", "cauchy": "FitCauchy", "chi": "FitChi", "chi2": "FitChiSquared", "expon": "FitExponential",
	"f": "FitF", "gamma": "FitGamma", "gumbel_r": "FitGumbel", "halfnorm": "FitHalfNormal", "invgamma": "FitInverseGamma",
	"laplace": "FitLaplace", "logistic": "FitLogistic", "lognorm": "FitLogNormal", "lomax": "FitLomax", "norm": "FitNormal",
	"pareto": "FitPareto", "rayleigh": "FitRayleigh", "reciprocal": "FitReciprocal", "t": "FitStudentT",
	"triang": "FitTriangular", "uniform": "FitUniform", "vonmises": "FitVonMises", "weibull_min": "FitWeibull",
}

func typeName(d any) string { return reflect.TypeOf(d).Name() }

// The data of the fit rows are 10000 draws of the distribution itself (seed 7),
// written by TestGenFitData to testdata/fit_<name>.f64 and read by BOTH sides:
// for 16 of the families SciPy's rvs and this package's differ in the last bit
// of a transcendental function (the Go side uses correctly rounded ones), and a
// fit timed on two different samples is not a comparison.
//
//go:embed testdata/fit_*.f64
var fitFiles embed.FS

const fitN = 10_000

func fitData(name string, n int) []float64 {
	b, err := fitFiles.ReadFile("testdata/fit_" + name + ".f64")
	if err != nil {
		panic(err)
	}
	x := make([]float64, n)
	for i := range x {
		x[i] = math.Float64frombits(binary.LittleEndian.Uint64(b[8*i:]))
	}
	return x
}

var genFlag = flag.Bool("perf.gen", false, "TestGenFitData: regenerate testdata/fit_*.f64")

func TestGenFitData(t *testing.T) {
	if !*genFlag {
		t.Skip("set -perf.gen to regenerate testdata/fit_*.f64")
	}
	for _, pd := range perfDists {
		r, ok := pd.d.(rvser)
		if _, fit := fits[pd.name]; !ok || !fit {
			continue
		}
		a, err := r.Rvs(random.DefaultRNG(7), fitN)
		if err != nil {
			t.Fatal(err)
		}
		x, _ := ndarray.Data[float64](a)
		b := make([]byte, 8*len(x))
		for i, v := range x {
			binary.LittleEndian.PutUint64(b[8*i:], math.Float64bits(v))
		}
		if err := os.WriteFile("testdata/fit_"+pd.name+".f64", b, 0o644); err != nil {
			t.Fatal(err)
		}
	}
}

func distRows() []row {
	var rs []row
	for _, pd := range perfDists {
		pd := pd
		tn := typeName(pd.d)
		type methodSet map[string]func(float64) float64
		var methods methodSet
		var order []string
		var goNames map[string]string
		switch d := pd.d.(type) {
		case Continuous:
			methods = methodSet{"pdf": d.PDF, "logpdf": d.LogPDF, "cdf": d.CDF, "sf": d.SF, "ppf": d.PPF, "isf": d.ISF, "logcdf": d.LogCDF, "logsf": d.LogSF}
			order = []string{"pdf", "logpdf", "cdf", "sf", "ppf", "isf", "logcdf", "logsf"}
			goNames = map[string]string{"pdf": "PDF", "logpdf": "LogPDF", "cdf": "CDF", "sf": "SF", "ppf": "PPF", "isf": "ISF", "logcdf": "LogCDF", "logsf": "LogSF"}
		case Discrete:
			methods = methodSet{"pmf": d.PMF, "logpmf": d.LogPMF, "cdf": d.CDF, "sf": d.SF, "ppf": d.PPF, "isf": d.ISF, "logcdf": d.LogCDF, "logsf": d.LogSF}
			order = []string{"pmf", "logpmf", "cdf", "sf", "ppf", "isf", "logcdf", "logsf"}
			goNames = map[string]string{"pmf": "PMF", "logpmf": "LogPMF", "cdf": "CDF", "sf": "SF", "ppf": "PPF", "isf": "ISF", "logcdf": "LogCDF", "logsf": "LogSF"}
		}
		for _, m := range order {
			f := methods[m]
			n, lo, hi, disc := 100_000, pd.lo, pd.hi, pd.discrete
			if m == "ppf" || m == "isf" {
				n, lo, hi, disc = 20_000, 1e-9, 1-1e-9, false
			}
			xs := func(n int) []float64 {
				x := points(n, lo, hi)
				if disc {
					floored(x)
				}
				return x
			}
			px := pyPoints(strconv.FormatFloat(lo, 'g', -1, 64), strconv.FormatFloat(hi, 'g', -1, 64))
			if disc {
				px = "np.floor(" + px + ")"
			}
			name := pd.name + "." + m
			ref := "scipy.stats." + pd.name + "." + m + " (one ufunc call over n points)"
			rs = append(rs, row{
				name: name, fn: tn + "." + goNames[m], ref: ref, n: n,
				serial: func(n int) func() {
					x := xs(n)
					out := make([]float64, n)
					return func() {
						for i, v := range x {
							out[i] = f(v)
						}
						sink = out[0]
					}
				},
				py:   pyCase{"(" + px + ",)", "lambda x: st." + pd.name + "." + m + "(" + pyWithShapes("x", pd.args) + ")"},
				data: func(n int) [][]float64 { return [][]float64{xs(n)} },
			})
			// The array API: one Map row per distribution (the cdf).
			if m == "cdf" {
				rs = append(rs, row{
					name: "Map/" + name, fn: "Map", ref: ref, n: n,
					call: func(n int) func() {
						xa := arr(xs(n), n)
						return func() { r := Map(xa, f); sink = float64(r.Size()) }
					},
					py:   pyCase{"(" + px + ",)", "lambda x: st." + pd.name + "." + m + "(" + pyWithShapes("x", pd.args) + ")"},
					data: func(n int) [][]float64 { return [][]float64{xs(n)} },
				})
			}
		}
		r, ok := pd.d.(rvser)
		if !ok {
			continue
		}
		rs = append(rs, row{
			name: pd.name + ".rvs", fn: tn + ".Rvs", n: 100_000,
			ref: "scipy.stats." + pd.name + ".rvs (size=n)",
			call: func(n int) func() {
				return func() { a, _ := r.Rvs(random.DefaultRNG(1), n); sink = float64(a.Size()) }
			},
			py:  pyCase{"()", "lambda: st." + pd.name + ".rvs(" + pyShapes(pd.args) + sep(pd.args) + "size=n, random_state=np.random.default_rng(1))"},
			why: "no arguments: the draws are the output (the stream is checked bit for bit by the rvs fixtures)",
		})
		if ent, ok := pd.d.(interface{ Entropy() float64 }); ok {
			rs = append(rs, row{
				name: pd.name + ".entropy", fn: tn + ".Entropy", n: 1,
				ref: "scipy.stats." + pd.name + "(...).entropy() (100 calls)",
				call: func(int) func() {
					return func() {
						for range 100 {
							sink = ent.Entropy()
						}
					}
				},
				py:  pyCase{"()", "lambda: [st." + pd.name + "(" + pyShapes(pd.args) + ").entropy() for _ in range(100)]"},
				why: "no arguments: a scalar per call",
			})
		}
		if fit, ok := fits[pd.name]; ok {
			rs = append(rs, row{
				name: pd.name + ".fit", fn: fitFuncs[pd.name], n: 10_000,
				ref: "scipy.stats." + pd.name + ".fit",
				call: func(n int) func() {
					x := fitData(pd.name, n)
					return func() { fit(x) }
				},
				py:   pyCase{`(np.fromfile("testdata/fit_` + pd.name + `.f64", dtype="<f8")[:n],)`, "st." + pd.name + ".fit"},
				data: func(n int) [][]float64 { return [][]float64{fitData(pd.name, n)} },
			})
		}
	}
	return rs
}

// ---- the rest of the public surface: named rows.

// Constructors of the common shapes of data. Each takes the call on the arrays.
func on1(name, fn string, n int, py pyCase, f func(a []float64)) row {
	return row{name: name, fn: fn, n: n, py: py,
		call: func(n int) func() { a := normalDraws(n, 1); return func() { f(a) } },
		data: func(n int) [][]float64 { return [][]float64{normalDraws(n, 1)} }}
}

func on2(name, fn string, n int, py pyCase, f func(a, b []float64)) row {
	return row{name: name, fn: fn, n: n, py: py,
		call: func(n int) func() { a, b := normalDraws(n, 1), normalDraws(n, 2); return func() { f(a, b) } },
		data: func(n int) [][]float64 { return [][]float64{normalDraws(n, 1), normalDraws(n, 2)} }}
}

func on3(name, fn string, n int, py pyCase, f func(a, b, c []float64)) row {
	return row{name: name, fn: fn, n: n, py: py,
		call: func(n int) func() {
			a, b, c := normalDraws(n, 1), normalDraws(n, 2), normalDraws(n, 3)
			return func() { f(a, b, c) }
		},
		data: func(n int) [][]float64 {
			return [][]float64{normalDraws(n, 1), normalDraws(n, 2), normalDraws(n, 3)}
		}}
}

func onPos(name, fn string, n int, py pyCase, f func(a []float64)) row {
	return row{name: name, fn: fn, n: n, py: py,
		call: func(n int) func() { a := positive(n, 1); return func() { f(a) } },
		data: func(n int) [][]float64 { return [][]float64{positive(n, 1)} }}
}

func onCounts(name, fn string, n int, py pyCase, f func(a []float64)) row {
	return row{name: name, fn: fn, n: n, py: py,
		call: func(n int) func() { a := counts(n); return func() { f(a) } },
		data: func(n int) [][]float64 { return [][]float64{counts(n)} }}
}

// object is a row whose SciPy arguments include an object (a frozen
// distribution, a fitted KDE): the hash of its arguments is not defined, the
// parameters and seeds are equal by construction.
func object(r row) row {
	r.data = nil
	r.why = "SciPy argument is an object (frozen distribution or KDE); same parameters and seeds by construction"
	return r
}

func perfMVN(n int) (*MultivariateNormal, *ndarray.Array) {
	cov := arr([]float64{
		2, 0.5, 0.3, 0.1, 0,
		0.5, 1.5, 0.2, 0, 0.1,
		0.3, 0.2, 1, 0.4, 0.2,
		0.1, 0, 0.4, 2.5, 0.3,
		0, 0.1, 0.2, 0.3, 1.2,
	}, 5, 5)
	mv, _ := NewMultivariateNormal([]float64{0, 1, -1, 0.5, 2}, cov, nil)
	x, _ := mv.Rvs(random.DefaultRNG(8), n)
	return mv, x
}

func namedRows() []row {
	const m, k = 100_000, 100_000
	pn := func(s string) pyCase { return pyCase{pyN1, s} }
	pp := func(s string) pyCase { return pyCase{"(" + pyPositive(1) + ",)", s} }
	p2 := func(s string) pyCase { return pyCase{pyN2, s} }
	p3 := func(s string) pyCase { return pyCase{pyN3, s} }
	lim := &TrimmedOptions{Limits: &Limits{Lower: -1, Upper: 1}}
	rs := []row{
		on1("mean", "Mean", m, pn("np.mean"), func(a []float64) { Mean(a, nil) }),
		on1("var", "Var", m, pn("np.var"), func(a []float64) { Var(a, nil) }),
		on1("std", "Std", m, pn("np.std"), func(a []float64) { Std(a, nil) }),
		on1("median", "Median", m, pn("np.median"), func(a []float64) { Median(a, nil) }),
		on1("quantile", "Quantile", m, pn("lambda a: np.quantile(a, 0.3)"), func(a []float64) { Quantile(a, 0.3, nil) }),
		{name: "average", fn: "Average", n: m, py: pyCase{"(" + pyNormal(1) + ", " + pyPositive(2) + ")", "lambda a, w: np.average(a, weights=w)"},
			call: func(n int) func() {
				a, w := normalDraws(n, 1), positive(n, 2)
				return func() { Average(a, &MeanOptions{Weights: w}) }
			},
			data: func(n int) [][]float64 { return [][]float64{normalDraws(n, 1), positive(n, 2)} }},
		on1("moment", "Moment", m, pn("lambda a: st.moment(a, 3)"), func(a []float64) { Moment(a, 3, nil) }),
		on1("skew", "Skew", m, pn("st.skew"), func(a []float64) { Skew(a, nil) }),
		on1("kurtosis", "Kurtosis", m, pn("st.kurtosis"), func(a []float64) { Kurtosis(a, nil) }),
		on1("sem", "SEM", m, pn("st.sem"), func(a []float64) { SEM(a, nil) }),
		onPos("variation", "Variation", m, pp("st.variation"), func(a []float64) { Variation(a, nil) }),
		on1("zscore", "ZScore", m, pn("st.zscore"), func(a []float64) { ZScore(a, nil) }),
		on2("zmap", "ZMap", m, p2("st.zmap"), func(a, b []float64) { ZMap(a, b, nil) }),
		onPos("gmean", "GMean", m, pp("st.gmean"), func(a []float64) { GMean(a, nil) }),
		onPos("hmean", "HMean", m, pp("st.hmean"), func(a []float64) { HMean(a, nil) }),
		onPos("pmean", "PMean", m, pp("lambda a: st.pmean(a, 2.5)"), func(a []float64) { PMean(a, 2.5, nil) }),
		on1("iqr", "IQR", m, pn("st.iqr"), func(a []float64) { IQR(a, nil) }),
		on1("mad", "MedianAbsDeviation", m, pn("st.median_abs_deviation"), func(a []float64) { MedianAbsDeviation(a, nil) }),
		{name: "mode", fn: "Mode", n: m, py: pyCase{"(np.round(" + pyNormal(1) + " * 10),)", "st.mode"},
			call: func(n int) func() {
				a := normalDraws(n, 1)
				for i := range a {
					a[i] = math.RoundToEven(a[i] * 10)
				}
				return func() { Mode(a, nil) }
			},
			data: func(n int) [][]float64 {
				a := normalDraws(n, 1)
				for i := range a {
					a[i] = math.RoundToEven(a[i] * 10)
				}
				return [][]float64{a}
			}},
		on1("tmean", "TMean", m, pn("lambda a: st.tmean(a, (-1, 1))"), func(a []float64) { TMean(a, lim) }),
		on1("tvar", "TVar", m, pn("lambda a: st.tvar(a, (-1, 1))"), func(a []float64) { TVar(a, lim) }),
		on1("tstd", "TStd", m, pn("lambda a: st.tstd(a, (-1, 1))"), func(a []float64) { TStd(a, lim) }),
		on1("tsem", "TSem", m, pn("lambda a: st.tsem(a, (-1, 1))"), func(a []float64) { TSem(a, lim) }),
		on1("tmin", "TMin", m, pn("lambda a: st.tmin(a, -1)"), func(a []float64) { TMin(a, -1, nil) }),
		on1("tmax", "TMax", m, pn("lambda a: st.tmax(a, 1)"), func(a []float64) { TMax(a, 1, nil) }),
		on1("trim_mean", "TrimMean", m, pn("lambda a: st.trim_mean(a, 0.1)"), func(a []float64) { TrimMean(a, 0.1) }),
		on1("trimboth", "TrimBoth", m, pn("lambda a: st.trimboth(a, 0.1)"), func(a []float64) { TrimBoth(a, 0.1) }),
		on1("trim1", "Trim1", m, pn("lambda a: st.trim1(a, 0.1, 'right')"), func(a []float64) { Trim1(a, 0.1, TailRight) }),
		on1("rankdata", "RankData", m, pn("st.rankdata"), func(a []float64) { RankData(a, RankAverage) }),
		on1("histogram", "Histogram", m, pn("lambda a: np.histogram(a, 100)"), func(a []float64) { Histogram(a, &HistogramOptions{Bins: 100}) }),
		on1("histogram_bin_edges", "HistogramBinEdges", m, pn("lambda a: np.histogram_bin_edges(a, 'auto')"), func(a []float64) { HistogramBinEdges(a, &HistogramOptions{Rule: BinsAuto}) }),
		on1("cumfreq", "CumFreq", m, pn("lambda a: st.cumfreq(a, 100)"), func(a []float64) { CumFreq(a, &FreqOptions{NumBins: 100}) }),
		on1("relfreq", "RelFreq", m, pn("lambda a: st.relfreq(a, 100)"), func(a []float64) { RelFreq(a, &FreqOptions{NumBins: 100}) }),
		on2("binned_statistic", "BinnedStatistic", m, p2("lambda a, b: st.binned_statistic(a, b, 'mean', 50)"), func(a, b []float64) {
			BinnedStatistic(a, b, &BinnedOptions{Statistic: BinMean, Bins: 50})
		}),
		onPos("entropy", "Entropy", m, pp("st.entropy"), func(a []float64) { Entropy(a, nil) }),
		on1("differential_entropy", "DifferentialEntropy", k, pn("st.differential_entropy"), func(a []float64) { DifferentialEntropy(a, nil) }),
		on1("describe", "Describe", m, pn("st.describe"), func(a []float64) { Describe(a, nil) }),
		on1("percentile", "Percentiles", m, pn("lambda a: np.percentile(a, [5, 25, 50, 75, 95])"), func(a []float64) {
			Percentiles(a, []float64{5, 25, 50, 75, 95}, nil)
		}),

		on2("ttest_rel", "TTestRel", m, p2("st.ttest_rel"), func(a, b []float64) { TTestRel(a, b, nil) }),
		{name: "ttest_ind", fn: "TTestInd", n: m, py: pyCase{"(" + pyNormal(1) + ", " + pyNormal(2) + " + 0.01)", "st.ttest_ind"},
			call: func(n int) func() {
				a, b := normalDraws(n, 1), normalDraws(n, 2)
				for i := range b {
					b[i] += 0.01
				}
				return func() { TTestInd(a, b, nil) }
			},
			data: func(n int) [][]float64 {
				a, b := normalDraws(n, 1), normalDraws(n, 2)
				for i := range b {
					b[i] += 0.01
				}
				return [][]float64{a, b}
			}},
		on1("ttest_1samp", "TTest1Samp", m, pn("lambda a: st.ttest_1samp(a, 0.0)"), func(a []float64) { TTest1Samp(a, 0, nil) }),
		on2("spearmanr", "Spearmanr", m, p2("st.spearmanr"), func(a, b []float64) { Spearmanr(a, b, nil) }),
		on2("pearsonr", "Pearsonr", m, p2("st.pearsonr"), func(a, b []float64) { Pearsonr(a, b, nil) }),
		on2("linregress", "Linregress", m, p2("st.linregress"), func(a, b []float64) { Linregress(a, b, nil) }),
		on2("kendalltau", "Kendalltau", k, p2("st.kendalltau"), func(a, b []float64) { Kendalltau(a, b, nil) }),
		on2("mannwhitneyu", "MannWhitneyU", k, p2("st.mannwhitneyu"), func(a, b []float64) { MannWhitneyU(a, b, nil) }),
		on2("wilcoxon", "Wilcoxon", k, p2("st.wilcoxon"), func(a, b []float64) { Wilcoxon(a, b, nil) }),
		on3("kruskal", "KruskalWallis", k, p3("st.kruskal"), func(a, b, c []float64) { KruskalWallis(a, b, c) }),
		on3("f_oneway", "FOneway", m, p3("st.f_oneway"), func(a, b, c []float64) { FOneway(nil, a, b, c) }),
		on3("bartlett", "Bartlett", m, p3("st.bartlett"), func(a, b, c []float64) { Bartlett(a, b, c) }),
		on3("levene", "Levene", m, p3("st.levene"), func(a, b, c []float64) { Levene(nil, a, b, c) }),
		on1("ks_1samp", "KSTest1Samp", k, pn("lambda a: st.ks_1samp(a, st.norm.cdf)"), func(a []float64) {
			KSTest1Samp(a, Normal{Scale: 1}.CDF, nil)
		}),
		on2("ks_2samp", "KSTest2Samp", k, p2("st.ks_2samp"), func(a, b []float64) { KSTest2Samp(a, b, nil) }),
		on1("shapiro", "ShapiroWilk", 5000, pn("st.shapiro"), func(a []float64) { ShapiroWilk(a) }),
		on1("anderson", "AndersonDarling", k, pn("lambda a: st.anderson(a, 'norm')"), func(a []float64) { AndersonDarling(a, ADNorm) }),
		on1("skewtest", "SkewTest", m, pn("st.skewtest"), func(a []float64) { SkewTest(a, nil) }),
		on1("kurtosistest", "KurtosisTest", m, pn("st.kurtosistest"), func(a []float64) { KurtosisTest(a, nil) }),
		on1("normaltest", "Normaltest", m, pn("st.normaltest"), func(a []float64) { Normaltest(a) }),
		on1("jarque_bera", "JarqueBera", m, pn("st.jarque_bera"), func(a []float64) { JarqueBera(a) }),
		onCounts("chisquare", "ChiSquare", m, pyCase{"(" + pyCounts + ",)", "st.chisquare"}, func(a []float64) { ChiSquare(a, nil) }),
		onCounts("power_divergence", "PowerDivergence", m, pyCase{"(" + pyCounts + ",)", "lambda a: st.power_divergence(a, lambda_='cressie-read')"},
			func(a []float64) { PowerDivergence(a, 2.0/3, nil) }),
		{name: "chi2_contingency", fn: "Chi2Contingency", n: 10_000, py: pyCase{"(" + pyCounts + ".reshape(100, n // 100),)", "st.chi2_contingency"},
			call: func(n int) func() {
				t := arr(counts(n), 100, n/100)
				return func() { Chi2Contingency(t, nil) }
			},
			data: func(n int) [][]float64 { return [][]float64{counts(n)} }},
		{name: "fisher_exact", fn: "FisherExact", n: 1, py: pyCase{"()", "lambda: st.fisher_exact([[1200, 800], [900, 1100]])"},
			call: func(int) func() { return func() { FisherExact([2][2]int64{{1200, 800}, {900, 1100}}, nil) } },
			why:  "no array arguments"},
		{name: "bootstrap", fn: "Bootstrap", n: 1000, py: pyCase{pyN1, "lambda a: st.bootstrap((a,), np.mean, n_resamples=9999, rng=np.random.default_rng(3))"},
			call: func(n int) func() {
				a := normalDraws(n, 1)
				mean := func(s ...[]float64) float64 { v, _ := Mean(s[0], nil); return v }
				return func() { Bootstrap([][]float64{a}, mean, random.DefaultRNG(3), nil) }
			},
			data: func(n int) [][]float64 { return [][]float64{normalDraws(n, 1)} }},
	}
	// Densities and kernels, and the multivariate families.
	rs = append(rs,
		row{name: "kde.evaluate", fn: "GaussianKDE.EvaluateSlice", n: 10_000,
			py: pyCase{"(" + pyNormal(1) + ", " + pyPoints("-4", "4") + ")", "lambda d, x: st.gaussian_kde(d)(x)"},
			call: func(n int) func() {
				d, x := normalDraws(n, 1), points(n, -4, 4)
				return func() { k, _ := NewGaussianKDE(d, nil); k.EvaluateSlice(x) }
			},
			data: func(n int) [][]float64 { return [][]float64{normalDraws(n, 1), points(n, -4, 4)} }},
		row{name: "kde.logpdf", fn: "GaussianKDE.LogPDFSlice", n: 10_000,
			py: pyCase{"(" + pyNormal(1) + ", " + pyPoints("-4", "4") + ")", "lambda d, x: st.gaussian_kde(d).logpdf(x)"},
			call: func(n int) func() {
				d, x := normalDraws(n, 1), points(n, -4, 4)
				return func() { k, _ := NewGaussianKDE(d, nil); k.LogPDFSlice(x) }
			},
			data: func(n int) [][]float64 { return [][]float64{normalDraws(n, 1), points(n, -4, 4)} }},
		row{name: "kde.evaluate2d", fn: "GaussianKDE.Evaluate", n: 10_000,
			py: pyCase{"(np.vstack([" + pyNormal(1) + ", " + pyNormal(2) + "]), np.vstack([" + pyPoints("-4", "4") + ", " + pyPoints("-3", "3") + "]))", "lambda d, x: st.gaussian_kde(d)(x)"},
			call: func(n int) func() {
				da := arr(append(normalDraws(n, 1), normalDraws(n, 2)...), 2, n)
				xa := arr(append(points(n, -4, 4), points(n, -3, 3)...), 2, n)
				return func() { k, _ := NewGaussianKDE(da, nil); k.Evaluate(xa) }
			},
			data: func(n int) [][]float64 {
				return [][]float64{append(normalDraws(n, 1), normalDraws(n, 2)...), append(points(n, -4, 4), points(n, -3, 3)...)}
			}},
		object(row{name: "kde.resample", fn: "GaussianKDE.Resample", n: 100_000,
			py: pyCase{"(st.gaussian_kde(np.random.default_rng(1).standard_normal(10_000)), n)", "lambda k, n: k.resample(n, seed=np.random.default_rng(4))"},
			call: func(n int) func() {
				k, _ := NewGaussianKDE(normalDraws(10_000, 1), nil)
				return func() { k.Resample(random.DefaultRNG(4), n) }
			}}),
		object(row{name: "kde.integrate_box_1d", fn: "GaussianKDE.IntegrateBox1D", n: 10_000,
			py: pyCase{"(st.gaussian_kde(" + pyNormal(1) + "),)", "lambda k: k.integrate_box_1d(-1, 0.5)"},
			call: func(n int) func() {
				k, _ := NewGaussianKDE(normalDraws(n, 1), nil)
				return func() { k.IntegrateBox1D(-1, 0.5) }
			}}),
		object(row{name: "mvn.logpdf", fn: "MultivariateNormal.LogPDFPoints", n: k,
			py: pyCase{"(" + pyMVN + ", " + pyMVNX + ")", "lambda d, x: d.logpdf(x)"},
			call: func(n int) func() {
				mv, x := perfMVN(n)
				return func() { mv.LogPDFPoints(x) }
			}}),
		object(row{name: "mvn.rvs", fn: "MultivariateNormal.Rvs", n: k,
			py: pyCase{"(" + pyMVN + ", n)", "lambda d, n: d.rvs(size=n, random_state=np.random.default_rng(5))"},
			call: func(n int) func() {
				mv, _ := perfMVN(1)
				return func() { mv.Rvs(random.DefaultRNG(5), n) }
			}}),
		object(row{name: "mvn.cdf", fn: "MultivariateNormal.CDF", n: 1,
			py: pyCase{"(" + pyMVN + ",)", "lambda d: d.cdf([0.5, 0.2, 1, -0.3, 0.8])"},
			call: func(int) func() {
				mv, _ := perfMVN(1)
				x := []float64{0.5, 0.2, 1, -0.3, 0.8}
				return func() { mv.CDF(x, nil) }
			}}),
		object(row{name: "mvn.fit", fn: "FitMultivariateNormal", n: k,
			py: pyCase{"(" + pyMVNX + ",)", "st.multivariate_normal.fit"},
			call: func(n int) func() {
				_, x := perfMVN(n)
				return func() { FitMultivariateNormal(x, nil) }
			}}),
		row{name: "dirichlet.logpdf", fn: "Dirichlet.LogPDFPoints", n: k,
			py: pyCase{"(st.dirichlet.rvs(" + pyAlpha + ", size=n, random_state=np.random.default_rng(6)).T,)", "lambda x: st.dirichlet.logpdf(x, " + pyAlpha + ")"},
			call: func(n int) func() {
				d, _ := NewDirichlet([]float64{1.5, 2, 3.5, 0.8, 4})
				x, _ := d.Rvs(random.DefaultRNG(6), n)
				return func() { d.LogPDFPoints(x) }
			},
			why: "arguments are draws of the distribution itself (stream checked by the rvs fixtures)"},
		row{name: "dirichlet.rvs", fn: "Dirichlet.Rvs", n: k,
			py: pyCase{"(n,)", "lambda n: st.dirichlet.rvs(" + pyAlpha + ", size=n, random_state=np.random.default_rng(6))"},
			call: func(n int) func() {
				d, _ := NewDirichlet([]float64{1.5, 2, 3.5, 0.8, 4})
				return func() { d.Rvs(random.DefaultRNG(6), n) }
			},
			why: "no arguments"},
		row{name: "multinomial.logpmf", fn: "Multinomial.LogPMFPoints", n: k,
			py: pyCase{"(st.multinomial.rvs(20, " + pyMNP + ", size=n, random_state=np.random.default_rng(6)),)", "lambda x: st.multinomial.logpmf(x, 20, " + pyMNP + ")"},
			call: func(n int) func() {
				d, _ := NewMultinomial(20, []float64{0.1, 0.2, 0.3, 0.15, 0.25})
				x, _ := d.Rvs(random.DefaultRNG(6), n)
				return func() { d.LogPMFPoints(x) }
			},
			why: "arguments are draws of the distribution itself (stream checked by the rvs fixtures)"},
		row{name: "multinomial.rvs", fn: "Multinomial.Rvs", n: k,
			py: pyCase{"(n,)", "lambda n: st.multinomial.rvs(20, " + pyMNP + ", size=n, random_state=np.random.default_rng(6))"},
			call: func(n int) func() {
				d, _ := NewMultinomial(20, []float64{0.1, 0.2, 0.3, 0.15, 0.25})
				return func() { d.Rvs(random.DefaultRNG(6), n) }
			},
			why: "no arguments"},
	)
	return rs
}

func rows() []row {
	rs := append(distRows(), namedRows()...)
	for i := range rs {
		rs[i].n = min(rs[i].n, 100_000)
	}
	return rs
}

// ---- refbench glue.

func startWorker(t testing.TB) *refbench.Worker {
	t.Helper()
	uv, err := exec.LookPath("uv")
	if err != nil {
		home, _ := os.UserHomeDir()
		uv = filepath.Join(home, ".local", "bin", "uv")
	}
	refs, err := filepath.Abs("refs.txt")
	if err != nil {
		t.Fatal(err)
	}
	w, err := refbench.StartPython(context.Background(), uv, refs)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { w.Close() })
	return w
}

var onlyFlag = flag.String("perf.only", "", "profiling only: a regexp; the registry keeps the rows whose name matches (never in CI)")

func cases(w *refbench.Worker) []refbench.Case {
	var out []refbench.Case
	for _, r := range rows() {
		if *onlyFlag != "" {
			if ok, _ := regexp.MatchString(*onlyFlag, r.name); !ok {
				continue
			}
		}
		n := r.size()
		if r.ref == "" { // the named rows: the SciPy (or NumPy) expression is the reference
			r.ref = "scipy.stats / numpy: " + r.py.call
		}
		var ref refbench.Timer
		if w != nil {
			ref = w.Timer(r.py.setup, r.py.call, n)
		}
		var body func() // built on first use: the data of 350 rows must not all be resident at once
		build := r.call
		if r.serial != nil {
			build = r.serial
		}
		out = append(out, refbench.Case{
			Name:      r.name,
			Func:      r.fn,
			Reference: r.ref,
			Parity:    refbench.ParityNone, // the ulp fixtures are the oracle (docs/accuracy.md)
			Go: func(iters int) {
				if body == nil {
					body = build(n)
				}
				for ; iters > 0; iters-- {
					body()
				}
			},
			Ref: ref,
		})
	}
	return out
}

func running() bool {
	f := flag.Lookup("perf.run")
	return f != nil && f.Value.String() == "true"
}

func TestPerf(t *testing.T) {
	var w *refbench.Worker
	if running() {
		w = startWorker(t)
	}
	o := refbench.Options{Repo: "go-statistics/stats"}
	if w != nil {
		v, err := w.RefVersions()
		if err != nil {
			t.Fatal(err)
		}
		o.RefVersions = v
	}
	rbtest.Perf(t, refbench.CIConfig{Root: "..", Cases: cases(w)}, o)
}

func TestRowsAreWellFormed(t *testing.T) {
	seen := map[string]bool{}
	for _, r := range rows() {
		if seen[r.name] {
			t.Fatalf("duplicate row %s", r.name)
		}
		seen[r.name] = true
		if r.py.setup == "" || r.py.call == "" || r.fn == "" || (r.call == nil) == (r.serial == nil) {
			t.Errorf("%s: incomplete row", r.name)
		}
		if r.data == nil && r.why == "" {
			t.Errorf("%s: neither data to hash nor a reason", r.name)
		}
	}
	for name := range nCap {
		if !seen[name] {
			t.Errorf("nCap names %q, which is no row", name)
		}
	}
}

// TestRowsRun runs every row once at a small size, so that a broken row fails
// the ordinary test run.
func TestRowsRun(t *testing.T) {
	for _, r := range rows() {
		n := min(r.size(), 1000)
		if r.name == "chi2_contingency" {
			n = 1000
		}
		build := r.call
		if r.serial != nil {
			build = r.serial
		}
		build(n)()
	}
}

func digest(data [][]float64) string {
	h := sha256.New()
	for _, d := range data {
		for _, x := range d {
			var b [8]byte
			binary.LittleEndian.PutUint64(b[:], math.Float64bits(x))
			h.Write(b[:])
		}
	}
	return hex.EncodeToString(h.Sum(nil))
}

// TestDataParity checks, for every row whose SciPy arguments are arrays, that
// the Go data and NumPy's are the same bytes (SHA-256): a ratio between two
// computations on different data means nothing. The rows with an object
// argument or no argument say why in their definition.
func TestDataParity(t *testing.T) {
	if !*parityFlag {
		t.Skip("set -perf.parity (starts the SciPy worker through uv)")
	}
	w := startWorker(t)
	var bad []string
	checked := 0
	for _, r := range rows() {
		if r.data == nil {
			continue
		}
		n := r.size()
		got, err := w.DataHash(r.py.setup, n)
		if err != nil {
			t.Errorf("%s: %v", r.name, err)
			continue
		}
		checked++
		if want := digest(r.data(n)); got != want {
			bad = append(bad, r.name)
		}
	}
	sort.Strings(bad)
	t.Logf("%d rows hashed", checked)
	if len(bad) > 0 {
		t.Errorf("%d rows differ in data: %v", len(bad), bad)
	}
}
