package stats

import (
	"bytes"
	"compress/gzip"
	_ "embed"
	"encoding/binary"
	"encoding/json"
	"errors"
	"hash/fnv"
	"math"
	"math/rand"
	"os"
	"os/exec"
	"runtime"
	"sort"
	"testing"
)

//go:embed testdata/scipy/multitest.json.gz
var multitestGz []byte

type mtCase struct {
	Fn      string      `json:"fn"`
	P       []*float64  `json:"p"`
	Alpha   float64     `json:"alpha"`
	Method  string      `json:"method"`
	Maxiter int         `json:"maxiter"`
	Ties    bool        `json:"ties"`
	Error   bool        `json:"error"`
	Reject  []bool      `json:"reject"`
	PCorr   []float64   `json:"pcorr"`
	Sidak   *float64    `json:"sidak"`
	Bonf    *float64    `json:"bonf"`
	N       int         `json:"n"`
	Value   float64     `json:"value"`
	_       interface{} `json:"-"`
}

type mtDoc struct {
	Versions map[string]string `json:"versions"`
	Cases    []mtCase          `json:"cases"`
}

func loadMT(t *testing.T) mtDoc {
	t.Helper()
	zr, err := gzip.NewReader(bytes.NewReader(multitestGz))
	if err != nil {
		t.Fatal(err)
	}
	var d mtDoc
	if err := json.NewDecoder(zr).Decode(&d); err != nil {
		t.Fatal(err)
	}
	return d
}

func (c mtCase) ps() []float64 {
	out := make([]float64, len(c.P))
	for i, v := range c.P {
		if v == nil {
			out[i] = math.NaN()
		} else {
			out[i] = *v
		}
	}
	return out
}

// mtUlp is the distance in representable doubles between two non-negative
// numbers (0 for equal values, a large number if either is NaN).
func mtUlp(a, b float64) uint64 {
	if a != a || b != b {
		return 1 << 62
	}
	x, y := math.Float64bits(a), math.Float64bits(b)
	if x > y {
		x, y = y, x
	}
	return y - x
}

func mtSame(a, b float64) bool { return math.Float64bits(a) == math.Float64bits(b) || (a == b) }

// TestMultiTestFixtures replays every case recorded from statsmodels 0.15.0
// and SciPy 1.18.1 (linux/amd64) and requires bit-for-bit equality of the
// adjusted p-values and equality of the rejections.
func TestMultiTestFixtures(t *testing.T) {
	d := loadMT(t)
	if d.Versions["statsmodels"] != "0.15.0" || d.Versions["scipy"] != "1.18.1" {
		t.Fatalf("fixture versions %v", d.Versions)
	}
	counts := map[string]int{}
	bad := 0
	fail := func(c mtCase, format string, args ...interface{}) {
		bad++
		if bad <= 15 {
			t.Errorf("%s %q alpha=%v maxiter=%d n=%d: "+format, append([]interface{}{c.Fn, c.Method, c.Alpha, c.Maxiter, len(c.P)}, args...)...)
		}
	}
	for _, c := range d.Cases {
		switch c.Fn {
		case "multipletests":
			r, err := MultipleTests(c.ps(), c.Alpha, c.Method, c.Maxiter)
			if c.Error {
				if err == nil {
					fail(c, "reference raised, Go did not")
				}
				continue
			}
			if err != nil {
				fail(c, "error %v", err)
				continue
			}
			ok, ulp1 := true, false
			for i := range c.PCorr {
				// sidak and holm-sidak go through glibc's log1p/expm1, whose
				// x86-64 FMA variants can differ by one ulp from the
				// fused-multiply-add-free msun port (documented).
				sidakFamily := c.Method == "sidak" || c.Method == "s" || c.Method == "holm-sidak" || c.Method == "hs"
				d := mtUlp(r.PAdj[i], c.PCorr[i])
				if d > 1 || (d == 1 && !sidakFamily) {
					fail(c, "padj[%d] = %v want %v (p=%v)", i, r.PAdj[i], c.PCorr[i], *c.P[i])
					ok = false
					break
				}
				if d == 1 {
					ulp1 = true
				}
				if !c.Ties && r.Reject[i] != c.Reject[i] {
					fail(c, "reject[%d] = %v want %v", i, r.Reject[i], c.Reject[i])
					ok = false
					break
				}
			}
			if len(c.P) > 0 && (!mtSame(r.AlphaSidak, *c.Sidak) || !mtSame(r.AlphaBonf, *c.Bonf)) {
				fail(c, "alphas %v %v want %v %v", r.AlphaSidak, r.AlphaBonf, *c.Sidak, *c.Bonf)
				ok = false
			}
			if len(c.P) == 0 && (!math.IsNaN(r.AlphaSidak) || !math.IsNaN(r.AlphaBonf)) {
				fail(c, "alphas for n=0 are not NaN")
				ok = false
			}
			if ok && ulp1 {
				counts["  of which one sidak-family value 1 ulp off"]++
			}
			if ok {
				counts["multipletests/"+c.Method]++
			}
		case "fdc":
			got, err := FalseDiscoveryControl(c.ps(), c.Method)
			if err != nil {
				fail(c, "error %v", err)
				continue
			}
			ok := true
			for i := range c.PCorr {
				if !mtSame(got[i], c.PCorr[i]) {
					fail(c, "fdc[%d] = %v want %v", i, got[i], c.PCorr[i])
					ok = false
					break
				}
			}
			if ok {
				counts["fdc/"+c.Method]++
			}
		case "fdc_bad":
			_, err := FalseDiscoveryControl(c.ps(), "bh")
			if (err != nil) != c.Error || (err != nil && !errors.Is(err, ErrData)) {
				fail(c, "bad input error = %v, reference error %v", err, c.Error)
			}
		case "npsum":
			if got := harmonic(c.N); !mtSame(got, c.Value) {
				fail(c, "harmonic(%d) = %v want %v", c.N, got, c.Value)
			} else {
				counts["npsum"]++
			}
		default:
			t.Fatalf("unknown case %q", c.Fn)
		}
	}
	keys := make([]string, 0, len(counts))
	total := 0
	for k, v := range counts {
		keys = append(keys, k)
		if k[0] != 32 {
			total += v
		}
	}
	sort.Strings(keys)
	for _, k := range keys {
		t.Logf("%-28s %5d equal", k, counts[k])
	}
	t.Logf("total %d cases equal of %d, %d mismatches", total, len(d.Cases), bad)
}

// TestMultiTestRegen regenerates the fixture inside a linux/amd64 container
// (needs docker; STATS_REGEN_MULTITEST=1).
func TestMultiTestRegen(t *testing.T) {
	if os.Getenv("STATS_REGEN_MULTITEST") == "" {
		t.Skip("set STATS_REGEN_MULTITEST=1 to regenerate the fixture with docker")
	}
	wd, _ := os.Getwd()
	cmd := exec.Command("docker", "run", "--rm", "--platform", "linux/amd64", "-v", wd+":/w", "-w", "/w",
		"ghcr.io/astral-sh/uv:0.9.5-python3.12-bookworm-slim", "uv", "run",
		"--with", "statsmodels==0.15.0", "--with", "scipy==1.18.1", "--with", "numpy==2.5.4",
		"python", "testdata/gen/multitest.py")
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("%v\n%s", err, out)
	}
	t.Logf("%s", out)
}

func TestMultipleTestsValidation(t *testing.T) {
	if _, err := MultipleTests([]float64{0.5}, 0.05, "nope", 1); !errors.Is(err, ErrParam) {
		t.Fatal(err)
	}
	for _, a := range []float64{-0.1, 1.1, math.NaN()} {
		if _, err := MultipleTests([]float64{0.5}, a, "holm", 1); !errors.Is(err, ErrParam) {
			t.Fatal(a, err)
		}
	}
	for _, p := range []float64{math.NaN(), -0.1, 1.5, math.Inf(1)} {
		if _, err := MultipleTests([]float64{0.2, p}, 0.05, "holm", 1); !errors.Is(err, ErrData) {
			t.Fatal(p, err)
		}
		if _, err := FalseDiscoveryControl([]float64{0.2, p}, "bh"); !errors.Is(err, ErrData) {
			t.Fatal(p, err)
		}
	}
	if _, err := FalseDiscoveryControl([]float64{0.5}, "holm"); !errors.Is(err, ErrParam) {
		t.Fatal(err)
	}
	r, err := MultipleTests(nil, 0.05, "hommel", 1)
	if err != nil || len(r.Reject) != 0 || len(r.PAdj) != 0 {
		t.Fatal(r, err)
	}
	if o, err := FalseDiscoveryControl(nil, "by"); err != nil || len(o) != 0 {
		t.Fatal(o, err)
	}
	if o, err := FalseDiscoveryControl([]float64{0.3}, "by"); err != nil || o[0] != 0.3 {
		t.Fatal(o, err)
	}
}

// TestMultipleTestsDocs checks the SciPy docstring example (Benjamini and
// Hochberg 1995, 15 p-values) and a Holm reference value.
func TestMultipleTestsDocs(t *testing.T) {
	ps := []float64{0.0001, 0.0004, 0.0019, 0.0095, 0.0201, 0.0278, 0.0298, 0.0344,
		0.0459, 0.3240, 0.4262, 0.5719, 0.6528, 0.7590, 1.000}
	got, err := FalseDiscoveryControl(ps, "bh")
	if err != nil {
		t.Fatal(err)
	}
	want := []float64{0.0015, 0.003, 0.0095, 0.035625, 0.0603, 0.06385714, 0.06385714,
		0.0645, 0.0765, 0.486, 0.58118182, 0.714875, 0.75323077, 0.81321429, 1}
	for i := range want {
		if math.Abs(got[i]-want[i]) > 5e-9 {
			t.Fatalf("bh[%d] = %v want %v", i, got[i], want[i])
		}
	}
	if g, _ := FalseDiscoveryControl([]float64{0.5, 0.6, 0.1, 0.001}, "BH"); g[0] != 0.6 || g[1] != 0.6 || g[2] != 0.2 || g[3] != 0.004 {
		t.Fatal(g)
	}
	r, _ := MultipleTests([]float64{0.01, 0.04, 0.03}, 0.05, "holm", 1)
	if r.PAdj[0] != 0.03 || r.PAdj[1] != 0.06 || r.PAdj[2] != 0.06 || !r.Reject[0] || r.Reject[1] {
		t.Fatal(r)
	}
}

// mtPortsGolden is the FNV-1a hash of the bits of log1pFD and expm1FD over the
// inputs of TestMultiTestHelpers (measured on arm64 and on a native s390x).
const mtPortsGolden = 0x4feeada2411be2db

func TestMultiTestHelpers(t *testing.T) {
	if powCR(0, 0.5) != 0 || powCR(0.3, 1) != 0.3 || powCR(1, 0.5) != 1 {
		t.Fatal("powCR special")
	}
	if g, w := powCR(0.95, 0.1), math.Pow(0.95, 0.1); math.Abs(g-w) > 2e-16 {
		t.Fatal(g, w)
	}
	if !math.IsInf(log1pFD(-1), -1) {
		t.Fatal("log1p(-1)")
	}
	// the fused-multiply-add-free ports agree with math.Log1p/Expm1 to one
	// ulp everywhere, and exactly where Go does not fuse (amd64).
	rng := rand.New(rand.NewSource(7))
	xs := []float64{-1e-30, -1e-8, -1e-17, -1e-5, -0.1, -0.2, -0.3, -0.5, -0.75, -0.999, math.Copysign(0, -1), -1e-300,
		-0.3, -0.4, -0.35, -0.7, -1 + 0x1p-52, -40, -2, -1.2, -0.6, -0.5, -0.3466, -0.35, -1.03, -1.05, -80, math.Inf(-1)}
	for i := 0; i < 4000; i++ {
		// math.Exp is assembly on s390x, so the inputs use only exact Ldexp
		// scalings: they must be the same bits on every architecture.
		xs = append(xs, -rng.Float64(), -math.Ldexp(rng.Float64()+0.5, rng.Intn(14)-8), -math.Ldexp(rng.Float64()+0.5, -rng.Intn(87)))
	}
	// On s390x math.Log1p and math.Expm1 are assembly routines (not the msun
	// code the ports follow) that differ from msun by 2 ulp, measured on a
	// native z17 and under qemu: they are no oracle there. The golden hash
	// below pins the ports' own output on every architecture instead.
	stdlibIsMsun := runtime.GOARCH != "s390x"
	h := fnv.New64a()
	var bits [8]byte
	for _, x := range xs {
		if x >= -1 {
			g, w := log1pFD(x), math.Log1p(x)
			if d := mtUlp(g, w); stdlibIsMsun && (d > 1 || (d != 0 && runtime.GOARCH == "amd64")) {
				t.Fatalf("log1pFD(%g) = %v want %v", x, g, w)
			}
			binary.LittleEndian.PutUint64(bits[:], math.Float64bits(g))
			h.Write(bits[:])
		}
		g, w := expm1FD(x), math.Expm1(x)
		if d := mtUlp(g, w); stdlibIsMsun && (d > 1 || (d != 0 && runtime.GOARCH == "amd64")) {
			t.Fatalf("expm1FD(%g) = %v want %v", x, g, w)
		}
		binary.LittleEndian.PutUint64(bits[:], math.Float64bits(g))
		h.Write(bits[:])
	}
	if got := h.Sum64(); got != mtPortsGolden {
		t.Fatalf("log1pFD/expm1FD output hash %#x on %s, want %#x", got, runtime.GOARCH, uint64(mtPortsGolden))
	}
	if npSum(nil) != 0 || npSum([]float64{2}) != 2 {
		t.Fatal("npSum small")
	}
	if harmonic(0) != 0 || harmonic(1) != 1 {
		t.Fatal("harmonic")
	}
}

// TestTwoStageBranches reaches the iterating branch with several stages.
func TestTwoStageBranches(t *testing.T) {
	p := []float64{0.001, 0.002, 0.004, 0.01, 0.03, 0.04, 0.2, 0.5, 0.7, 0.9}
	for _, m := range []string{"fdr_tsbh", "fdr_tsbky"} {
		for _, it := range []int{-1, 0, 1, 2, 5} {
			r, err := MultipleTests(p, 0.05, m, it)
			if err != nil || len(r.PAdj) != len(p) {
				t.Fatal(m, it, err)
			}
		}
	}
}

func checkMTInvariants(t *testing.T, p []float64, alpha float64) {
	t.Helper()
	for _, m := range []string{"b", "s", "hs", "h", "sh", "ho", "bh", "by", "tsbh", "tsbky"} {
		if m == "ho" && len(p) > 60 {
			continue
		}
		name := map[string]string{"bh": "fdr_bh", "by": "fdr_by", "tsbh": "fdr_tsbh", "tsbky": "fdr_tsbky"}[m]
		if name == "" {
			name = m
		}
		r, err := MultipleTests(p, alpha, name, 1)
		if err != nil {
			t.Fatalf("%s: %v on %v", m, err, p)
		}
		if len(r.Reject) != len(p) || len(r.PAdj) != len(p) {
			t.Fatalf("%s: lengths", m)
		}
		for i, v := range r.PAdj {
			if !(v >= 0 && v <= 1) {
				t.Fatalf("%s: padj[%d] = %v out of [0,1]", m, i, v)
			}
			if v < p[i]*(1-1e-15) && m != "tsbh" && m != "tsbky" {
				t.Fatalf("%s: padj[%d] = %v below p = %v", m, i, v, p[i])
			}
		}
	}
	for _, m := range []string{"bh", "by"} {
		o, err := FalseDiscoveryControl(p, m)
		if err != nil || len(o) != len(p) {
			t.Fatal(m, err)
		}
		for i, v := range o {
			if !(v >= p[i] && v <= 1) && len(p) > 1 {
				t.Fatalf("fdc %s: [%d] = %v for p = %v", m, i, v, p[i])
			}
		}
	}
}

func TestMultiTestInvariants(t *testing.T) {
	checkMTInvariants(t, []float64{0, 1, 0.5, 0.5, 1e-300}, 0.05)
	checkMTInvariants(t, []float64{1, 1, 1}, 1)
	checkMTInvariants(t, []float64{0, 0, 0}, 0)
	big := make([]float64, 5000)
	for i := range big {
		big[i] = float64(i%97) / 97
	}
	for _, m := range []string{"holm", "fdr_by", "fdr_tsbky", "bonferroni"} {
		if _, err := MultipleTests(big, 0.05, m, 1); err != nil {
			t.Fatal(m, err)
		}
	}
}

func FuzzMultipleTests(f *testing.F) {
	f.Add(0.01, 0.04, 0.03, 0.5, 0.05)
	f.Add(math.NaN(), 0.5, 0.1, 0.2, 0.05)
	f.Add(-1.0, 2.0, 0.0, 1.0, 0.3)
	f.Add(0.0, 0.0, 1.0, 1.0, 1.0)
	f.Fuzz(func(t *testing.T, a, b, c, d, alpha float64) {
		p := []float64{a, b, c, d}
		valid := checkPValues(p) == nil
		_, err := MultipleTests(p, alpha, "holm-sidak", 1)
		okAlpha := alpha >= 0 && alpha <= 1
		switch {
		case valid && okAlpha && err != nil:
			t.Fatal(err)
		case (!valid || !okAlpha) && err == nil:
			t.Fatal("accepted invalid input")
		}
		if valid && okAlpha {
			checkMTInvariants(t, p, alpha)
		}
		if _, err := FalseDiscoveryControl(p, "by"); (err == nil) != valid {
			t.Fatal(err, valid)
		}
	})
}
