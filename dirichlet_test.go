package stats

import (
	"errors"
	"fmt"
	"math"
	"testing"

	"github.com/go-ndarray/ndarray"
	"github.com/go-ndarray/ndarray/random"
)

type dirichletFixture struct {
	Cases []struct {
		Alpha  mvFloats `json:"alpha"`
		Points []struct {
			X           mvFloats `json:"x"`
			LogPDF      mvFloat  `json:"logpdf"`
			PDF         mvFloat  `json:"pdf"`
			Mp          mvFloat  `json:"mp"`
			MpPDF       mvFloat  `json:"mp_pdf"`
			LogPDFShort *mvFloat `json:"logpdf_short"`
		} `json:"points"`
		Mean      mvFloats   `json:"mean"`
		Var       mvFloats   `json:"var"`
		Cov       mvFloats   `json:"cov"`
		Entropy   mvFloat    `json:"entropy"`
		MpEntropy mvFloat    `json:"mp_entropy"`
		Bad       []mvFloats `json:"bad"`
		BadRaises []bool     `json:"bad_raises"`
		Rvs       []struct {
			Seed uint64   `json:"seed"`
			Size int      `json:"size"`
			Hex  mvFloats `json:"hex"`
		} `json:"rvs"`
	} `json:"cases"`
}

func TestDirichletParity(t *testing.T) {
	var fx dirichletFixture
	loadMV(t, "dirichlet", &fx)
	var ours, sp, oursP, spP, ent, spEnt, mom mvMax
	var draws mvDraws
	for ci, c := range fx.Cases {
		d, err := NewDirichlet(c.Alpha.f())
		if err != nil {
			t.Fatal(err)
		}
		var rows []float64
		for pi, p := range c.Points {
			w := fmt.Sprintf("alpha %v point %d", c.Alpha.f(), pi)
			lp, err := d.LogPDF(p.X.f())
			if err != nil {
				t.Fatal(err)
			}
			pd, _ := d.PDF(p.X.f())
			// Relative to the size of the terms that cancel (log B and
			// the log terms), the scale of the log-density's rounding.
			scale := math.Max(1, math.Abs(float64(p.Mp)))
			ours.add(math.Abs(lp-float64(p.Mp))/scale, w)
			sp.add(math.Abs(float64(p.LogPDF)-float64(p.Mp))/scale, w)
			oursP.add(mvRel(pd, float64(p.MpPDF)), w)
			spP.add(mvRel(float64(p.PDF), float64(p.MpPDF)), w)
			if p.LogPDFShort != nil {
				x := p.X.f()
				ls, err := d.LogPDF(x[:len(x)-1])
				if err != nil {
					t.Fatal(err)
				}
				if math.Abs(ls-float64(*p.LogPDFShort)) > 1e-9*scale {
					t.Errorf("%s: K-1 coordinates give %g, SciPy %g", w, ls, float64(*p.LogPDFShort))
				}
			}
			rows = append(rows, p.X.f()...)
		}
		if len(c.Points) > 0 {
			k := len(c.Alpha)
			got, err := d.LogPDFPoints(mvArr(rows, len(rows)/k, k))
			if err != nil {
				t.Fatal(err)
			}
			pg, _ := d.PDFPoints(mvArr(rows, len(rows)/k, k))
			for i := range got {
				want, _ := d.LogPDF(rows[i*k : (i+1)*k])
				wp, _ := d.PDF(rows[i*k : (i+1)*k])
				if got[i] != want || pg[i] != wp {
					t.Errorf("LogPDFPoints row %d: %g, LogPDF %g", i, got[i], want)
				}
			}
		}
		ent.add(mvRel(d.Entropy(), float64(c.MpEntropy)), fmt.Sprint("case ", ci))
		spEnt.add(mvRel(float64(c.Entropy), float64(c.MpEntropy)), fmt.Sprint("case ", ci))
		cv, _ := ndarray.Data[float64](d.Cov())
		for _, pair := range [][2][]float64{{d.Mean(), c.Mean.f()}, {d.Var(), c.Var.f()}, {cv, c.Cov.f()}} {
			for i := range pair[1] {
				mom.add(mvULP(pair[0][i], pair[1][i]), fmt.Sprint("moments case ", ci))
			}
		}
		for bi, b := range c.Bad {
			_, err := d.LogPDF(b.f())
			if (err != nil) != c.BadRaises[bi] {
				t.Errorf("case %d bad point %v: error %v, SciPy raises %v", ci, b.f(), err, c.BadRaises[bi])
			}
		}
		for _, r := range c.Rvs {
			a, err := d.Rvs(random.DefaultRNG(r.Seed), r.Size)
			if err != nil {
				t.Fatal(err)
			}
			v, _ := ndarray.Data[float64](a)
			draws.add(v, r.Hex.f())
		}
	}
	t.Logf("logpdf abs err/max(1,|logpdf|) vs mpmath: ours %.3g (%s), SciPy %.3g (%s)", ours.v, ours.where, sp.v, sp.where)
	t.Logf("pdf rel err vs mpmath: ours %.3g (%s), SciPy %.3g (%s)", oursP.v, oursP.where, spP.v, spP.where)
	t.Logf("entropy rel err vs mpmath: ours %.3g (%s), SciPy %.3g (%s)", ent.v, ent.where, spEnt.v, spEnt.where)
	t.Logf("mean/var/cov: max %g ULP from SciPy (%s)", mom.v, mom.where)
	t.Logf("rvs: %d draws, %d differ, max %g ULP", draws.n, draws.diff, draws.maxULP)
	if ours.v > math.Max(sp.v, 4e-16) {
		t.Errorf("logpdf: error %g above SciPy's %g", ours.v, sp.v)
	}
	if oursP.v > math.Max(spP.v, 1e-15) || ent.v > math.Max(spEnt.v, 1e-15) || mom.v > 2 {
		t.Errorf("pdf %g, entropy %g, moments %g ULP", oursP.v, ent.v, mom.v)
	}
	if draws.diff != 0 {
		t.Errorf("rvs: %d of %d draws differ from NumPy's", draws.diff, draws.n)
	}
}

func TestDirichletErrors(t *testing.T) {
	for _, a := range [][]float64{nil, {1, 0}, {1, -1}, {math.Inf(1)}, {math.NaN()}} {
		if _, err := NewDirichlet(a); !errors.Is(err, ErrParam) {
			t.Errorf("alpha %v: %v", a, err)
		}
	}
	d, _ := NewDirichlet([]float64{0.5, 2, 1})
	if got := d.Alpha(); got[1] != 2 {
		t.Error(got)
	}
	for _, x := range [][]float64{{0.5}, {0.2, 0.3, 0.4, 0.1}, {0, 0.5, 0.5}, {0.5, math.NaN(), 0.2}, {1.2, -0.2, 0}, {-0.1, 0.6, 0.5}} {
		if _, err := d.LogPDF(x); !errors.Is(err, ErrData) {
			t.Errorf("x %v: %v", x, err)
		}
		if _, err := d.PDF(x); !errors.Is(err, ErrData) {
			t.Errorf("x %v: %v", x, err)
		}
	}
	// x = 0 where alpha > 1: density 0.
	if p, err := d.PDF([]float64{0.5, 0, 0.5}); p != 0 || err != nil {
		t.Errorf("pdf on the boundary: %g %v", p, err)
	}
	// x = 0 where alpha = 1 is allowed (xlogy(0, 0) = 0).
	if p, err := d.LogPDF([]float64{0.5, 0.5, 0}); err != nil || math.IsInf(p, 0) {
		t.Errorf("logpdf with alpha = 1 at 0: %g %v", p, err)
	}
	if _, err := d.LogPDFPoints(nil); !errors.Is(err, ErrData) {
		t.Error(err)
	}
	if _, err := d.LogPDFPoints(ndarray.Scalar(1.0)); !errors.Is(err, ErrData) {
		t.Error(err)
	}
	if _, err := d.LogPDFPoints(mvArr([]float64{0.2, 0.3, 0.5, 2, 0, 0}, 2, 3)); !errors.Is(err, ErrData) {
		t.Error(err)
	}
	if _, err := d.Rvs(random.DefaultRNG(0), -1); !errors.Is(err, ErrParam) {
		t.Error(err)
	}
}
