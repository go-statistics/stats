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
}
