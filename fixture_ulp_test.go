package stats

import (
	"math"
	"testing"
)

func TestULPMeasure(t *testing.T) {
	if got := ulp(1); got != 0x1p-52 {
		t.Errorf("ulp(1) = %g", got)
	}
	if got := ulp(1.5); got != 0x1p-52 {
		t.Errorf("ulp(1.5) = %g", got)
	}
	if got := ulp(0x1p-1030); got != 0x1p-1074 {
		t.Errorf("ulp(subnormal) = %g", got)
	}
	if got := ulp(math.Inf(-1)); !math.IsInf(got, 1) {
		t.Errorf("ulp(-inf) = %g", got)
	}
	cases := []struct{ got, hi, lo, want float64 }{
		{1, 1, 0, 0},
		{math.Nextafter(1, 2), 1, 0, 1},
		{1, 1, 0x1p-54, 0.25},
		{math.NaN(), math.NaN(), 0, 0},
		{1, math.NaN(), 0, math.Inf(1)},
		{math.NaN(), 1, 0, math.Inf(1)},
		{math.Inf(1), math.Inf(1), 0, 0},
		{math.MaxFloat64, math.Inf(1), 0, math.Inf(1)},
		{math.Inf(1), math.MaxFloat64, 0, math.Inf(1)},
	}
	for _, c := range cases {
		if got := ulpErr(c.got, c.hi, c.lo); got != c.want {
			t.Errorf("ulpErr(%g, %g, %g) = %g, want %g", c.got, c.hi, c.lo, got, c.want)
		}
	}
	if s := summarise(nil, nil); s.n != 0 {
		t.Error("summarise(nil)")
	}
	for _, x := range []float64{math.Inf(1), 0, 0.5, 12, 2e7} {
		if fmtULP(x) == "" {
			t.Error("fmtULP")
		}
	}
}
