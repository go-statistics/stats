package stats

import (
	"errors"
	"fmt"
	"math"
	"sort"
	"testing"

	"github.com/go-ndarray/ndarray/random"
)

type bootstrapFixture struct {
	Samples map[string]mvFloats `json:"samples"`
	Cases   []struct {
		Data            []string `json:"data"`
		Stat            string   `json:"stat"`
		Method          string   `json:"method"`
		Alternative     string   `json:"alternative"`
		NResamples      int      `json:"n_resamples"`
		Batch           int      `json:"batch"`
		ConfidenceLevel float64  `json:"confidence_level"`
		Paired          bool     `json:"paired"`
		Seed            uint64   `json:"seed"`
		Low             mvFloat  `json:"low"`
		High            mvFloat  `json:"high"`
		SE              mvFloat  `json:"se"`
		Head            mvFloats `json:"head"`
		Sum             mvFloat  `json:"sum"`
		Len             int      `json:"len"`
		Dist            mvFloats `json:"dist"`
	} `json:"cases"`
}

func mvMedian(x []float64) float64 {
	s := append([]float64(nil), x...)
	sort.Float64s(s)
	n := len(s)
	if n%2 == 1 {
		return s[n/2]
	}
	return (s[n/2-1] + s[n/2]) / 2
}

var bootstrapStats = map[string]func(...[]float64) float64{
	"mean":   func(s ...[]float64) float64 { return mean(s[0]) },
	"diff":   func(s ...[]float64) float64 { return mean(s[0]) - mean(s[1]) },
	"median": func(s ...[]float64) float64 { return mvMedian(s[0]) },
}

var bootstrapMethods = map[string]BootstrapMethod{"BCa": BootstrapBCa, "percentile": BootstrapPercentile, "basic": BootstrapBasic}
var bootstrapAlts = map[string]Alternative{"two-sided": TwoSided, "less": Less, "greater": Greater}

// TestBootstrapParity checks that the resamples, the bootstrap
// distribution, the interval and the standard error are SciPy's.
func TestBootstrapParity(t *testing.T) {
	var fx bootstrapFixture
	loadMV(t, "bootstrap", &fx)
	var dist, ci, ciBCa, se mvMax
	nDist, nDiff := 0, 0
	for i, c := range fx.Cases {
		w := fmt.Sprintf("case %d (%v %s %s %s paired=%v)", i, c.Data, c.Stat, c.Method, c.Alternative, c.Paired)
		samples := make([][]float64, len(c.Data))
		for j, n := range c.Data {
			samples[j] = fx.Samples[n].f()
		}
		o := &BootstrapOptions{NResamples: c.NResamples, ConfidenceLevel: c.ConfidenceLevel,
			Method: bootstrapMethods[c.Method], Alternative: bootstrapAlts[c.Alternative],
			Paired: c.Paired, Batch: c.Batch}
		if c.NResamples == 9999 {
			o.NResamples = 0 // the default
		}
		if c.ConfidenceLevel == 0.95 {
			o.ConfidenceLevel = 0
		}
		r, err := Bootstrap(samples, bootstrapStats[c.Stat], random.DefaultRNG(c.Seed), o)
		if err != nil {
			t.Fatal(err)
		}
		if len(r.BootstrapDistribution) != c.Len {
			t.Fatalf("%s: %d resamples, SciPy %d", w, len(r.BootstrapDistribution), c.Len)
		}
		want := c.Dist.f()
		if want == nil {
			want = c.Head.f()
		}
		for j := range want {
			nDist++
			if u := mvULP(r.BootstrapDistribution[j], want[j]); u != 0 {
				nDiff++
				dist.add(u, w)
			}
		}
		// Every value, through its exact sum (math.fsum in the fixture).
		if s := sumDD(r.BootstrapDistribution).Float(); s != float64(c.Sum) {
			t.Errorf("%s: sum of the distribution %v, SciPy %v", w, s, float64(c.Sum))
		}
		m := &ci
		if c.Method == "BCa" {
			m = &ciBCa
		}
		m.add(mvULP(r.ConfidenceInterval.Low, float64(c.Low)), w)
		m.add(mvULP(r.ConfidenceInterval.High, float64(c.High)), w)
		se.add(mvULP(r.StandardError, float64(c.SE)), w)
	}
	t.Logf("bootstrap distribution: %d values compared, %d differ (max %g ULP, %s)", nDist, nDiff, dist.v, dist.where)
	t.Logf("interval: percentile/basic max %g ULP from SciPy (%s); BCa max %g ULP (%s); standard error: max %g ULP (%s)",
		ci.v, ci.where, ciBCa.v, ciBCa.where, se.v, se.where)
	// The BCa levels go through ndtri and ndtr, where SciPy's are off by an
	// ULP or two (special.Ndtri(0.025000000000000022) is correctly
	// rounded, SciPy's is not): a few ULP in the interval.
	if nDiff > 0 || ci.v > 0 || se.v > 0 || ciBCa.v > 16 {
		t.Errorf("not SciPy's: %d values differ, interval %g ULP (BootstrapBCa %g), standard error %g ULP", nDiff, ci.v, ciBCa.v, se.v)
	}
}

func TestBootstrapErrors(t *testing.T) {
	x := []float64{1, 2, 3, 4}
	st := bootstrapStats["mean"]
	g := random.DefaultRNG(0)
	for _, c := range []struct {
		name    string
		samples [][]float64
		o       BootstrapOptions
		err     error
	}{
		{"no samples", nil, BootstrapOptions{}, ErrData},
		{"short", [][]float64{{1}}, BootstrapOptions{}, ErrData},
		{"paired lengths", [][]float64{x, {1, 2, 3}}, BootstrapOptions{Paired: true}, ErrData},
		{"resamples", [][]float64{x}, BootstrapOptions{NResamples: -1}, ErrParam},
		{"batch", [][]float64{x}, BootstrapOptions{Batch: -1}, ErrParam},
		{"level", [][]float64{x}, BootstrapOptions{ConfidenceLevel: 1.5}, ErrParam},
		{"method", [][]float64{x}, BootstrapOptions{Method: 9}, ErrParam},
		{"alternative", [][]float64{x}, BootstrapOptions{Alternative: 9}, ErrParam},
	} {
		if _, err := Bootstrap(c.samples, st, g, &c.o); !errors.Is(err, c.err) {
			t.Errorf("%s: %v, want %v", c.name, err, c.err)
		}
	}
	// nil options and generator; a NaN statistic gives a NaN interval.
	r, err := Bootstrap([][]float64{x}, st, nil, nil)
	if err != nil || len(r.BootstrapDistribution) != 9999 {
		t.Error(err, len(r.BootstrapDistribution))
	}
	r, _ = Bootstrap([][]float64{{1, math.NaN(), 3}}, st, g, &BootstrapOptions{NResamples: 50, Method: BootstrapPercentile})
	if !math.IsNaN(r.ConfidenceInterval.Low) || !math.IsNaN(r.ConfidenceInterval.High) {
		t.Error(r.ConfidenceInterval)
	}
	for m, s := range map[BootstrapMethod]string{BootstrapBCa: "BCa", BootstrapPercentile: "percentile", BootstrapBasic: "basic", 7: "invalid"} {
		if m.String() != s {
			t.Errorf("%d: %q", m, m.String())
		}
	}
	if q := quantileLinear([]float64{1, 2}, 1.5, false); !math.IsNaN(q) {
		t.Error(q)
	}
}
