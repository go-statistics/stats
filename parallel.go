package stats

import (
	"runtime"
	"sync"

	"github.com/go-ndarray/ndarray"
)

// parallelMin is the length below which Map and MapSlice stay on the
// calling goroutine: the cost of starting workers is a few microseconds.
const parallelMin = 1 << 14

// Map applies a scalar function (typically a distribution method value,
// d.CDF) to every element of x and returns a float64 array of x's shape:
// the vectorised call scipy.stats.<dist>.cdf(x, ...) makes. Large arrays
// are split into contiguous blocks evaluated on GOMAXPROCS goroutines; each
// element is computed by exactly one call of f, so the result does not
// depend on the number of goroutines. f must be safe for concurrent use
// (every method of this package's distributions is).
func Map(x *ndarray.Array, f func(float64) float64) *ndarray.Array {
	in := arrayFloats(x)
	r, _ := ndarray.New(x.Shape()...)
	out, _ := ndarray.Data[float64](r) // a new array's own storage
	MapSlice(out, in, f)
	return r
}

// MapSlice sets dst[i] = f(src[i]) for every i (dst and src may be the same
// slice; dst must be at least as long as src), in parallel like Map.
func MapSlice(dst, src []float64, f func(float64) float64) {
	n := len(src)
	w := runtime.GOMAXPROCS(0)
	if n < parallelMin || w == 1 {
		for i, v := range src {
			dst[i] = f(v)
		}
		return
	}
	chunk := (n + w - 1) / w
	var wg sync.WaitGroup
	for lo := 0; lo < n; lo += chunk {
		hi := min(lo+chunk, n)
		wg.Add(1)
		go func(lo, hi int) {
			defer wg.Done()
			for i := lo; i < hi; i++ {
				dst[i] = f(src[i])
			}
		}(lo, hi)
	}
	wg.Wait()
}
