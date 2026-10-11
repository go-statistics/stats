package stats

import (
	"compress/gzip"
	"embed"
	"encoding/json"
	"math"
	"strconv"
	"testing"

	"github.com/go-ndarray/ndarray"
)

// The SciPy parity fixtures of the multivariate distributions, the kernel
// density estimate and the bootstrap (testdata/gen/scipy_multivariate.py).

//go:embed testdata/scipy/mv_*.json.gz
var mvFS embed.FS

// mvFloat is a float64 stored as its exact hexadecimal form.
type mvFloat float64

func (f *mvFloat) UnmarshalJSON(b []byte) error {
	var s string
	if err := json.Unmarshal(b, &s); err != nil {
		return err
	}
	v, err := strconv.ParseFloat(s, 64)
	*f = mvFloat(v)
	return err
}

type mvFloats []mvFloat

func (h mvFloats) f() []float64 {
	out := make([]float64, len(h))
	for i, v := range h {
		out[i] = float64(v)
	}
	return out
}

func loadMV(t testing.TB, name string, v any) {
	t.Helper()
	fh, err := mvFS.Open("testdata/scipy/mv_" + name + ".json.gz")
	if err != nil {
		t.Fatal(err)
	}
	defer fh.Close()
	gz, err := gzip.NewReader(fh)
	if err != nil {
		t.Fatal(err)
	}
	if err := json.NewDecoder(gz).Decode(v); err != nil {
		t.Fatal(err)
	}
}

// mvRel is |got-want|/|want| (0 when both are equal, including both
// infinite or NaN; +Inf when exactly one is NaN or infinite).
func mvRel(got, want float64) float64 {
	if got == want || math.IsNaN(got) && math.IsNaN(want) {
		return 0
	}
	if math.IsNaN(got) || math.IsNaN(want) || math.IsInf(got, 0) || math.IsInf(want, 0) {
		return math.Inf(1)
	}
	if want == 0 {
		return math.Abs(got)
	}
	return math.Abs(got-want) / math.Abs(want)
}

// mvULP is the distance in units in the last place between two floats.
func mvULP(a, b float64) float64 {
	if a == b || math.IsNaN(a) && math.IsNaN(b) {
		return 0
	}
	if math.IsNaN(a) || math.IsNaN(b) || math.IsInf(a, 0) || math.IsInf(b, 0) {
		return math.Inf(1)
	}
	ia, ib := int64(math.Float64bits(a)), int64(math.Float64bits(b))
	if ia < 0 {
		ia = math.MinInt64 - ia
	}
	if ib < 0 {
		ib = math.MinInt64 - ib
	}
	d := ia - ib
	if d < 0 {
		d = -d
	}
	return float64(d)
}

// mvMax accumulates a maximum with where it happened.
type mvMax struct {
	v     float64
	where string
}

func (m *mvMax) add(v float64, where string) {
	if v > m.v || math.IsNaN(v) {
		m.v, m.where = v, where
	}
}

func mvArr(v []float64, shape ...int) *ndarray.Array { return mustArray(v, shape...) }

// mvDraws counts draws compared and differing, with the largest ULP
// distance.
type mvDraws struct {
	n, diff int
	maxULP  float64
}

func (s *mvDraws) add(got, want []float64) {
	for i := range want {
		s.n++
		if u := mvULP(got[i], want[i]); u != 0 {
			s.diff++
			s.maxULP = math.Max(s.maxULP, u)
		}
	}
}
