package stats

import (
	"bufio"
	"flag"
	"fmt"
	"math"
	"os"
	"os/exec"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/go-ndarray/ndarray"
	"github.com/go-ndarray/ndarray/random"
)

// The speed comparison with SciPy (docs/perf.md):
//
//	go test -run TestPerfReport -perf.report=docs/perf-table.md -timeout 60m
//
// starts bench/scipy_server.py through uv (the native interpreter: the
// comparison is with what a user of this machine runs) and, case by case and
// round by round, times the SciPy call and the Go call on the same data back
// to back, so that the load of the shared host hits both alike. The table
// reports medians.
var (
	perfReport = flag.String("perf.report", "", "write the SciPy speed comparison (markdown) to this file")
	perfRounds = flag.Int("perf.rounds", 7, "interleaved rounds per case")
	perfOnly   = flag.String("perf.only", "", "only cases whose name has this prefix (a comma-separated list)")
	perfFrom   = flag.String("perf.from", "", "only cases whose name sorts at or after this one")
)

const gold = 0.6180339887498949

// points is lo + (hi-lo)·frac(i·gold), the same IEEE operations as
// bench/scipy_server.py.
func points(n int, lo, hi float64) []float64 {
	v := make([]float64, n)
	for i := range v {
		f := math.Mod(float64(i)*gold, 1)
		v[i] = lo + float64((hi-lo)*f)
	}
	return v
}

func normalDraws(n int, seed uint64) []float64 {
	a, _ := random.DefaultRNG(seed).StandardNormal(n)
	v, _ := ndarray.Data[float64](a)
	return v
}

// perfCase is one row: n, and a setup returning the timed calls. serial is
// a single-goroutine loop (nil when the operation is not elementwise), par
// the call a user makes (Map for elementwise functions).
type perfCase struct {
	n     int
	setup func(n int) (serial, par func())
	// py, when set, is the command sent to bench/scipy_server.py instead of
	// the case name (generic distribution cases: "@dist norm pdf ...").
	py string
}

var perfCases = map[string]perfCase{}

func elementwise(n int, lo, hi float64, f func(float64) float64) perfCase {
	return perfCase{n: n, setup: func(n int) (func(), func()) {
		x := points(n, lo, hi)
		xa, _ := ndarray.FromData(x, n)
		out := make([]float64, n)
		return func() {
				for i, v := range x {
					out[i] = f(v)
				}
			}, func() {
				Map(xa, f)
			}
	}}
}

func init() {
	const m = 1_000_000
	perfCases["norm.pdf"] = elementwise(m, -10, 10, Normal{Scale: 1}.PDF)
	perfCases["norm.cdf"] = elementwise(m, -10, 10, Normal{Scale: 1}.CDF)
	perfCases["norm.ppf"] = elementwise(m, 1e-12, 1-1e-12, Normal{Scale: 1}.PPF)
	perfCases["beta.pdf"] = elementwise(m, 1e-6, 1-1e-6, Beta{A: 2.5, B: 7, Scale: 1}.PDF)
	perfCases["beta.cdf"] = elementwise(m, 1e-6, 1-1e-6, Beta{A: 2.5, B: 7, Scale: 1}.CDF)
	perfCases["beta.ppf"] = elementwise(100_000, 1e-9, 1-1e-9, Beta{A: 2.5, B: 7, Scale: 1}.PPF)
	perfCases["gamma.pdf"] = elementwise(m, 1e-3, 30, Gamma{A: 4.5, Scale: 1}.PDF)
	perfCases["gamma.cdf"] = elementwise(m, 1e-3, 30, Gamma{A: 4.5, Scale: 1}.CDF)
	perfCases["gamma.ppf"] = elementwise(100_000, 1e-9, 1-1e-9, Gamma{A: 4.5, Scale: 1}.PPF)
	perfCases["t.cdf"] = elementwise(m, -20, 20, StudentT{Df: 7.5, Scale: 1}.CDF)
	perfCases["t.ppf"] = elementwise(100_000, 1e-9, 1-1e-9, StudentT{Df: 7.5, Scale: 1}.PPF)
	perfCases["chi2.sf"] = elementwise(m, 1e-3, 100, ChiSquared{Df: 12, Scale: 1}.SF)
	perfCases["norm.rvs"] = perfCase{n: m, setup: func(n int) (func(), func()) {
		return nil, func() { Normal{Scale: 1}.Rvs(random.DefaultRNG(1), n) }
	}}
	perfCases["gamma.rvs"] = perfCase{n: m, setup: func(n int) (func(), func()) {
		return nil, func() { Gamma{A: 2.5, Scale: 1}.Rvs(random.DefaultRNG(1), n) }
	}}
}

func medianOf(v []float64) float64 {
	s := append([]float64(nil), v...)
	sort.Float64s(s)
	return s[len(s)/2]
}

func timeIt(f func()) float64 {
	t0 := time.Now()
	f()
	return time.Since(t0).Seconds()
}

func TestPerfReport(t *testing.T) {
	if *perfReport == "" {
		t.Skip("set -perf.report=PATH to run the SciPy speed comparison")
	}
	uv := os.ExpandEnv("$HOME/.local/bin/uv")
	cmd := exec.Command(uv, "run", "--with", "scipy==1.18.1", "--with", "numpy==2.5.4", "python", "bench/scipy_server.py")
	// One thread on the SciPy side too: the comparison is core for core.
	cmd.Env = append(os.Environ(), "OMP_NUM_THREADS=1", "OPENBLAS_NUM_THREADS=1", "MKL_NUM_THREADS=1", "VECLIB_MAXIMUM_THREADS=1")
	stdin, _ := cmd.StdinPipe()
	stdout, _ := cmd.StdoutPipe()
	cmd.Stderr = os.Stderr
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	rd := bufio.NewScanner(stdout)
	scipy := func(name string, n int) float64 {
		fmt.Fprintf(stdin, "%s %d\n", name, n)
		if !rd.Scan() {
			t.Fatalf("scipy server: %v", rd.Err())
		}
		v, err := strconv.ParseFloat(strings.TrimSpace(rd.Text()), 64)
		if err != nil {
			t.Fatal(err)
		}
		return v
	}
	up, _ := exec.Command("uptime").Output()
	var names []string
	for k := range perfCases {
		if k < *perfFrom {
			continue
		}
		for _, p := range strings.Split(*perfOnly, ",") {
			if strings.HasPrefix(k, p) {
				names = append(names, k)
				break
			}
		}
	}
	sort.Strings(names)
	var b strings.Builder
	fmt.Fprintf(&b, "<!-- generated by: go test -run TestPerfReport -perf.report=docs/perf-table.md -->\n\n")
	fmt.Fprintf(&b, "Median of %d interleaved rounds (3 for calls over 2 s, 1 over 20 s); %s/%s, GOMAXPROCS=%d, %d CPUs; host load at start: %s\n\n",
		*perfRounds, runtime.GOOS, runtime.GOARCH, runtime.GOMAXPROCS(0), runtime.NumCPU(), strings.TrimSpace(string(up)))
	b.WriteString("| case | n | SciPy (ms) | Go, one core (ms) | Go, all cores (ms) | speed-up, one core | speed-up, all cores | slower because |\n")
	b.WriteString("|---|--:|--:|--:|--:|--:|--:|---|\n")
	for _, name := range names {
		c := perfCases[name]
		serial, par := c.setup(c.n)
		if serial == nil {
			// Not elementwise: the single-core time is the same call with
			// GOMAXPROCS=1 (the functions that split work over goroutines
			// then stay on one).
			serial = func() {
				defer runtime.GOMAXPROCS(runtime.GOMAXPROCS(1))
				par()
			}
		}
		py := name
		if c.py != "" {
			py = c.py
		}
		scipy(py, c.n) // warm-up both sides
		warm := timeIt(par)
		rounds := *perfRounds
		switch {
		case warm > 20: // the slowest fits: one round (minutes per call)
			rounds = 1
		case warm > 2:
			rounds = min(rounds, 3)
		}
		var ts, tg, tp []float64
		for r := 0; r < rounds; r++ {
			ts = append(ts, scipy(py, c.n))
			tg = append(tg, timeIt(serial))
			tp = append(tp, timeIt(par))
		}
		ms, mg, mp := medianOf(ts)*1e3, medianOf(tg)*1e3, medianOf(tp)*1e3
		fmt.Fprintf(&b, "| %s | %d | %.4g | %.4g | %.4g | %.2fx | %.2fx |%s\n", name, c.n, ms, mg, mp, ms/mg, ms/mp, perfNote(name, ms/mg))
		t.Logf("%s: scipy %.4g ms, go one core %.4g ms, go %.4g ms", name, ms, mg, mp)
		// the table so far, so that an interrupted run keeps its rows
		if err := os.WriteFile(*perfReport, []byte(b.String()), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	fmt.Fprintln(stdin, "quit")
	cmd.Wait()
	if err := os.WriteFile(*perfReport, []byte(b.String()), 0o644); err != nil {
		t.Fatal(err)
	}
}
