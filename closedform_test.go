package stats

import (
	"reflect"
	"testing"
)

func init() {
	// Draws through SciPy's formulas evaluated with correctly rounded log
	// and exp; the reference build's exp misrounds about 0.2% of its
	// results, hence a few draws per thousand 1 ulp apart (2 after the
	// scale and shift of an instance with Scale < 1).
	rvsULP["Reciprocal"] = 1
	rvsULP["LogNormal"] = 2
	// go-ndarray's Laplace sampler takes Go's log where NumPy takes the C
	// library's (1 draw in 9000 differs, by 1 ulp).
	rvsULP["Laplace"] = 1
	rvsULP["Gumbel"] = 1 // -log(-log U), the C library's log misrounding
	// pow(-log1p(-U), 1/c) with SciPy's own log1p reproduced exactly, the
	// power correctly rounded where the C library's misrounds a few in a
	// thousand (2 ulp after the shift of an instance with Loc).
	rvsULP["Weibull"] = 2
	rvsULP["Pareto"] = 1 // pow(1-U, -1/b), the C library's pow misrounding
	// expm1(-log1p(-U)/c) with SciPy's own log1p and expm1 reproduced
	// exactly; beyond |x| = 1/2 its expm1 is exp(x) - 1, where the C
	// library's exp misrounds a few in a thousand and the subtraction
	// magnifies that last bit up to 4 ulp of the result.
	rvsULP["Lomax"] = 4

	// SciPy's lomax ISF is q^(-1/c) - 1, which cancels for q near 1 (the
	// ULP fixture lomax_isf measures it at 8.9e15 ulp in its upper region,
	// 6204 in the core): at c = 50, q = 0.999999 it is off by 5e-10
	// relative.
	parityPointTol["Lomax"] = 1e-8
	// SciPy's Cauchy draws are Boost's -1/tan(π·U) with the C library's
	// tan, which on the reference build is its sin/cos ratio and misrounds
	// a third of its results by up to 3 ulp; ours rounds the same formula
	// correctly. A quantile near 0 then shifted by Loc magnifies the
	// difference in units of the result's ulp (measured: 256).
	rvsULP["Cauchy"] = 256
}

// exactFits holds the fits of a fixture whose fixed parameters satisfy
// closed to SciPy's parameters bit for bit (the fixture's fit_kind is
// "mle" because its other fits solve equations numerically).
func exactFits(t *testing.T, scipy string, closed func(fixed map[string]float64) bool) {
	t.Helper()
	n := 0
	for _, pf := range loadParity(t) {
		if pf.SciPy != scipy {
			continue
		}
		e := distTypes[pf.Go]
		for _, c := range pf.Fit {
			if !closed(c.Fixed) {
				continue
			}
			x := make([]float64, len(c.Data))
			for i, v := range c.Data {
				x[i] = float64(v)
			}
			got, err := e.fit(x, &FitOptions{Fixed: c.Fixed})
			if err != nil {
				t.Fatalf("%s fit %v: %v", scipy, c.Fixed, err)
			}
			gv := reflect.ValueOf(got)
			for k, want := range c.Params {
				if g := gv.FieldByName(k).Float(); g != float64(want) {
					t.Errorf("%s fit (fixed %v): %s = %v, SciPy %v", scipy, c.Fixed, k, g, float64(want))
				}
			}
			n++
		}
	}
	if n == 0 {
		t.Errorf("%s: no closed-form fit in the fixture", scipy)
	}
}

func has(m map[string]float64, k string) bool { _, ok := m[k]; return ok }

func TestExactFits(t *testing.T) {
	exactFits(t, "lognorm", func(f map[string]float64) bool { return has(f, "Loc") })
	exactFits(t, "rayleigh", func(f map[string]float64) bool { return has(f, "Loc") })
	exactFits(t, "pareto", func(f map[string]float64) bool { return has(f, "Loc") || has(f, "Scale") })
}
