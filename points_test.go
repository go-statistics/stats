package stats

import "math"

// points is lo + (hi-lo)·frac(i·gold): n deterministic points of [lo, hi], the
// same IEEE operations as bench/perf_test.go and NumPy's mod (the explicit
// float64 conversion forbids a fused multiply-add).
func points(n int, lo, hi float64) []float64 {
	v := make([]float64, n)
	for i := range v {
		f := math.Mod(float64(i)*0.6180339887498949, 1)
		v[i] = lo + float64((hi-lo)*f)
	}
	return v
}
