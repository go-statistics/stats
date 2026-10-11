package stats

import "testing"

// BenchmarkClosedForm times PDF, CDF and PPF of each closed-form family at
// a point in the bulk (go test -run - -bench ClosedForm -benchmem); every
// one of them is allocation-free.
func BenchmarkClosedForm(b *testing.B) {
	dists := []struct {
		name string
		d    Continuous
		x    float64
	}{
		{"Exponential", Exponential{Scale: 1}, 0.7},
		{"Uniform", Uniform{Scale: 1}, 0.3},
		{"Reciprocal", Reciprocal{A: 0.01, B: 100, Scale: 1}, 0.7},
		{"LogNormal", LogNormal{S: 0.8, Scale: 1}, 0.7},
		{"Cauchy", Cauchy{Scale: 1}, 0.7},
		{"Laplace", Laplace{Scale: 1}, 0.7},
		{"Logistic", Logistic{Scale: 1}, 0.7},
		{"Weibull", Weibull{C: 1.5, Scale: 1}, 0.7},
		{"Pareto", Pareto{B: 2.5, Scale: 1}, 1.7},
		{"Gumbel", Gumbel{Scale: 1}, 0.7},
		{"Triangular", Triangular{C: 0.3, Scale: 1}, 0.7},
		{"HalfNormal", HalfNormal{Scale: 1}, 0.7},
		{"Rayleigh", Rayleigh{Scale: 1}, 0.7},
		{"Lomax", Lomax{C: 2.5, Scale: 1}, 0.7},
		{"VonMises", VonMises{Kappa: 2, Scale: 1}, 0.7},
	}
	for _, c := range dists {
		d, x := c.d, c.x
		b.Run(c.name+"/pdf", func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				_ = d.PDF(x)
			}
		})
		b.Run(c.name+"/cdf", func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				_ = d.CDF(x)
			}
		})
		b.Run(c.name+"/ppf", func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				_ = d.PPF(0.3)
			}
		})
	}
}
