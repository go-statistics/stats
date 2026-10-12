package stats

import (
	"testing"

	"github.com/go-ndarray/ndarray/random"
)

func benchNormals(seed uint64, n int) []float64 {
	g := random.DefaultRNG(seed)
	a, _ := g.StandardNormal(n)
	return arrayFloats(a)
}

func BenchmarkTTestInd1e6(b *testing.B) {
	x, y := benchNormals(1, 1_000_000), benchNormals(2, 1_000_000)
	b.ReportAllocs()
	for b.Loop() {
		if _, err := TTestInd(x, y, &TTestIndOptions{Welch: true}); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkSpearmanr1e6(b *testing.B) {
	x, y := benchNormals(3, 1_000_000), benchNormals(4, 1_000_000)
	b.ReportAllocs()
	for b.Loop() {
		if _, err := Spearmanr(x, y, nil); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkKendalltau1e5(b *testing.B) {
	x, y := benchNormals(5, 100_000), benchNormals(6, 100_000)
	b.ReportAllocs()
	for b.Loop() {
		if _, err := Kendalltau(x, y, nil); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkMannWhitneyU1e5(b *testing.B) {
	x, y := benchNormals(7, 100_000), benchNormals(8, 100_000)
	b.ReportAllocs()
	for b.Loop() {
		if _, err := MannWhitneyU(x, y, nil); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkKSTest2Samp1e5(b *testing.B) {
	x, y := benchNormals(9, 100_000), benchNormals(10, 100_000)
	b.ReportAllocs()
	for b.Loop() {
		if _, err := KSTest2Samp(x, y, nil); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkKSTest1Samp1e5(b *testing.B) {
	x := benchNormals(11, 100_000)
	cdf := Normal{Scale: 1}.CDF
	b.ReportAllocs()
	for b.Loop() {
		if _, err := KSTest1Samp(x, cdf, nil); err != nil {
			b.Fatal(err)
		}
	}
}
