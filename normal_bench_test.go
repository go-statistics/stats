package stats

import (
	"testing"

	"github.com/go-numerics/special"
)

func BenchmarkSpecialNdtri(b *testing.B) {
	x := points(1024, 1e-12, 1-1e-12)
	s := 0.0
	for i := 0; i < b.N; i++ {
		s += special.Ndtri(x[i&1023])
	}
	_ = s
}

func BenchmarkNdtr(b *testing.B) {
	x := points(1024, -10, 10)
	s := 0.0
	for i := 0; i < b.N; i++ {
		s += special.Ndtr(x[i&1023])
	}
	_ = s
}

func BenchmarkNormPDF(b *testing.B) {
	x := points(1024, -10, 10)
	s := 0.0
	for i := 0; i < b.N; i++ {
		s += normalStd{}.pdf(x[i&1023])
	}
	_ = s
}
