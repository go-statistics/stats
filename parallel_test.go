package stats

import (
	"math"
	"runtime"
	"testing"

	"github.com/go-ndarray/ndarray"
)

func TestMap(t *testing.T) {
	for _, n := range []int{5, parallelMin + 3} {
		v := make([]float64, n)
		for i := range v {
			v[i] = float64(i)/float64(n)*10 - 5
		}
		x, _ := ndarray.FromData(v, n)
		d := Normal{Scale: 1}
		y := Map(x, d.CDF)
		got, _ := ndarray.Data[float64](y)
		for i := range v {
			if got[i] != d.CDF(v[i]) {
				t.Fatalf("n=%d: Map[%d] = %v", n, i, got[i])
			}
		}
	}
	// a single worker stays on the calling goroutine
	old := runtime.GOMAXPROCS(1)
	defer runtime.GOMAXPROCS(old)
	src := make([]float64, parallelMin+1)
	MapSlice(src, src, math.Exp)
	if src[0] != 1 {
		t.Fatal("MapSlice in place")
	}
}
