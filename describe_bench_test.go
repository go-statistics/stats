package stats

import (
	"testing"

	"github.com/go-ndarray/ndarray"
	"github.com/go-ndarray/ndarray/random"
)

// benchData is 10^6 standard normal values from NumPy's generator.
func benchData(b *testing.B) []float64 {
	a, err := random.DefaultRNG(1).StandardNormal(1_000_000)
	if err != nil {
		b.Fatal(err)
	}
	v, _ := ndarray.Data[float64](a)
	return v
}

func BenchmarkQuantile1e6(b *testing.B) {
	x := benchData(b)
	for b.Loop() {
		_, _ = Quantile(x, 0.37, nil)
	}
}

func BenchmarkQuantiles1e6(b *testing.B) {
	x := benchData(b)
	qs := []float64{0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99}
	for b.Loop() {
		_, _ = Quantiles(x, qs, nil)
	}
}

func BenchmarkPercentile1e6(b *testing.B) {
	x := benchData(b)
	for b.Loop() {
		_, _ = Percentile(x, 95, &QuantileOptions{Method: QuantileMedianUnbiased})
	}
}

func BenchmarkMedian1e6(b *testing.B) {
	x := benchData(b)
	for b.Loop() {
		_, _ = Median(x, nil)
	}
}

func BenchmarkDescribe1e6(b *testing.B) {
	x := benchData(b)
	for b.Loop() {
		_, _ = Describe(x, nil)
	}
}
