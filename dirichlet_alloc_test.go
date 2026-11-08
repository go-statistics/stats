package stats

import (
	"math"
	"testing"

	"github.com/go-ndarray/ndarray"
	"github.com/go-ndarray/ndarray/random"
)

// The batch methods keep one buffer for the completed point instead of copying
// every row: the allocations of LogPDFPoints and PDFPoints do not grow with the
// number of points (before: one per point, 50003 for 100000 points).
func TestDirichletPointsDoNotAllocatePerPoint(t *testing.T) {
	d, err := NewDirichlet([]float64{1.5, 2, 3.5, 0.8, 4})
	if err != nil {
		t.Fatal(err)
	}
	draw := func(n int) *ndarray.Array {
		x, err := d.Rvs(random.DefaultRNG(6), n)
		if err != nil {
			t.Fatal(err)
		}
		return x
	}
	small, large := draw(100), draw(2000)
	for name, f := range map[string]func(*ndarray.Array) ([]float64, error){"LogPDFPoints": d.LogPDFPoints, "PDFPoints": d.PDFPoints} {
		a := testing.AllocsPerRun(20, func() { f(small) })
		b := testing.AllocsPerRun(20, func() { f(large) })
		if a != b {
			t.Errorf("%s allocates %v times for 100 points and %v for 2000", name, a, b)
		}
		if a != 4 {
			t.Errorf("%s allocates %v times, want exactly 4 (the result, the point buffer, the array's data and its header)", name, a)
		}
	}
}

// The buffer changes nothing: a batch gives, bit for bit, the points one at a
// time give, for K and for K-1 coordinates.
func TestDirichletPointsMatchOnePointAtATime(t *testing.T) {
	d, _ := NewDirichlet([]float64{1.5, 2, 3.5, 0.8, 4})
	x, err := d.Rvs(random.DefaultRNG(6), 300)
	if err != nil {
		t.Fatal(err)
	}
	v := arrayFloats(x)
	for _, w := range []int{5, 4} {
		rows := make([]float64, 0, 300*w)
		for i := range 300 {
			rows = append(rows, v[i*5:i*5+w]...)
		}
		a, _ := ndarray.FromData(rows, 300, w)
		lp, err := d.LogPDFPoints(a)
		if err != nil {
			t.Fatal(err)
		}
		pp, err := d.PDFPoints(a)
		if err != nil {
			t.Fatal(err)
		}
		for i := range 300 {
			l, _ := d.LogPDF(rows[i*w : (i+1)*w])
			p, _ := d.PDF(rows[i*w : (i+1)*w])
			if math.Float64bits(l) != math.Float64bits(lp[i]) || math.Float64bits(p) != math.Float64bits(pp[i]) {
				t.Fatalf("K=%d row %d: batch (%v, %v), single (%v, %v)", w, i, lp[i], pp[i], l, p)
			}
		}
	}
}
