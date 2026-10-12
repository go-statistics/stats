package stats

import (
	"errors"
	"fmt"
	"math"
	"testing"

	"github.com/go-ndarray/ndarray"
	"github.com/go-ndarray/ndarray/random"
)

type multinomialFixture struct {
	Cases []struct {
		N      int      `json:"n"`
		P      mvFloats `json:"p"`
		Points []struct {
			X      mvFloats `json:"x"`
			LogPMF mvFloat  `json:"logpmf"`
			PMF    mvFloat  `json:"pmf"`
			Mp     *mvFloat `json:"mp"`
			MpPMF  *mvFloat `json:"mp_pmf"`
		} `json:"points"`
		Mean      mvFloats `json:"mean"`
		Cov       mvFloats `json:"cov"`
		Entropy   mvFloat  `json:"entropy"`
		MpEntropy mvFloat  `json:"mp_entropy"`
		Rvs       []struct {
			Seed uint64  `json:"seed"`
			Size int     `json:"size"`
			Int  []int64 `json:"int"`
		} `json:"rvs"`
	} `json:"cases"`
	Invalid []struct {
		N   int      `json:"n"`
		P   mvFloats `json:"p"`
		NaN bool     `json:"nan"`
	} `json:"invalid"`
}

func TestMultinomialParity(t *testing.T) {
	var fx multinomialFixture
	loadMV(t, "multinomial", &fx)
	var ours, sp, vsSciPy, spP, ent, spEnt, mom mvMax
	draws, differ := 0, 0
	for ci, c := range fx.Cases {
		m, err := NewMultinomial(c.N, c.P.f())
		if err != nil {
			t.Fatal(err)
		}
		k := len(c.P)
		var rows []float64
		for pi, p := range c.Points {
			w := fmt.Sprintf("n=%d p=%v x=%v", c.N, c.P.f(), p.X.f())
			lp, err := m.LogPMF(p.X.f())
			if err != nil {
				t.Fatal(err)
			}
			pm, _ := m.PMF(p.X.f())
			if p.Mp != nil {
				scale := math.Max(1, math.Abs(float64(*p.Mp)))
				ours.add(math.Abs(lp-float64(*p.Mp))/scale, w)
				sp.add(math.Abs(float64(p.LogPMF)-float64(*p.Mp))/scale, w)
				vsSciPy.add(mvRel(pm, float64(*p.MpPMF)), w)
				spP.add(mvRel(float64(p.PMF), float64(*p.MpPMF)), w)
			} else if lp != float64(p.LogPMF) || pm != float64(p.PMF) {
				t.Errorf("%s (point %d): %g %g, SciPy %g %g", w, pi, lp, pm, float64(p.LogPMF), float64(p.PMF))
			}
			rows = append(rows, p.X.f()...)
		}
		got, err := m.LogPMFPoints(mvArr(rows, len(rows)/k, k))
		if err != nil {
			t.Fatal(err)
		}
		pg, _ := m.PMFPoints(mvArr(rows, len(rows)/k, k))
		for i := range got {
			want, _ := m.LogPMF(rows[i*k : (i+1)*k])
			wp, _ := m.PMF(rows[i*k : (i+1)*k])
			if got[i] != want || pg[i] != wp {
				t.Errorf("points row %d", i)
			}
		}
		ent.add(mvRel(m.Entropy(), float64(c.MpEntropy)), fmt.Sprint("entropy case ", ci))
		spEnt.add(mvRel(float64(c.Entropy), float64(c.MpEntropy)), fmt.Sprint("entropy case ", ci))
		cv, _ := ndarray.Data[float64](m.Cov())
		for _, pair := range [][2][]float64{{m.Mean(), c.Mean.f()}, {cv, c.Cov.f()}} {
			for i := range pair[1] {
				mom.add(mvULP(pair[0][i], pair[1][i]), fmt.Sprint("moments case ", ci))
			}
		}
		for _, r := range c.Rvs {
			a, err := m.Rvs(random.DefaultRNG(r.Seed), r.Size)
			if err != nil {
				t.Fatal(err)
			}
			v, _ := ndarray.Data[int64](a)
			for i := range r.Int {
				draws++
				if v[i] != r.Int[i] {
					differ++
				}
			}
		}
	}
	t.Logf("logpmf abs err/max(1,|logpmf|) vs mpmath: ours %.3g (%s), SciPy %.3g (%s)", ours.v, ours.where, sp.v, sp.where)
	t.Logf("pmf rel err vs mpmath: ours %.3g (%s), SciPy %.3g (%s)", vsSciPy.v, vsSciPy.where, spP.v, spP.where)
	t.Logf("entropy rel err vs mpmath: ours %.3g (%s), SciPy %.3g (%s)", ent.v, ent.where, spEnt.v, spEnt.where)
	t.Logf("mean/cov: max %g ULP from SciPy (%s)", mom.v, mom.where)
	t.Logf("rvs: %d draws, %d differ", draws, differ)
	if ours.v > math.Max(sp.v, 4e-16) {
		t.Errorf("logpmf error %g above SciPy's %g", ours.v, sp.v)
	}
	if vsSciPy.v > math.Max(spP.v, 4e-15) || ent.v > math.Max(spEnt.v, 1e-15) || mom.v > 0 || differ > 0 {
		t.Errorf("pmf %g, entropy %g, moments %g ULP, %d draws differ", vsSciPy.v, ent.v, mom.v, differ)
	}
	for _, c := range fx.Invalid {
		if _, err := NewMultinomial(c.N, c.P.f()); !errors.Is(err, ErrParam) || !c.NaN {
			t.Errorf("n=%d p=%v: %v (SciPy NaN %v)", c.N, c.P.f(), err, c.NaN)
		}
	}
}

func TestMultinomialEdges(t *testing.T) {
	for _, p := range [][]float64{nil, {math.NaN(), 1}} {
		if _, err := NewMultinomial(3, p); !errors.Is(err, ErrParam) {
			t.Errorf("p %v: %v", p, err)
		}
	}
	m, _ := NewMultinomial(4, []float64{0.25, 0.75})
	if m.N() != 4 || m.P()[1] != 0.75 {
		t.Error(m.N(), m.P())
	}
	if _, err := m.LogPMF([]float64{4}); !errors.Is(err, ErrData) {
		t.Error(err)
	}
	if _, err := m.PMF([]float64{4}); !errors.Is(err, ErrData) {
		t.Error(err)
	}
	if v, err := m.LogPMF([]float64{math.NaN(), 4}); !math.IsNaN(v) || err != nil {
		t.Error(v, err)
	}
	if v, _ := m.PMF([]float64{math.NaN(), 4}); !math.IsNaN(v) {
		t.Error(v)
	}
	if _, err := m.LogPMFPoints(mvArr([]float64{1, 2, 3}, 1, 3)); !errors.Is(err, ErrData) {
		t.Error(err)
	}
	if _, err := m.Rvs(random.DefaultRNG(0), -1); !errors.Is(err, ErrParam) {
		t.Error(err)
	}
	// The binomial probabilities against mpmath: Loader's algorithm is
	// accurate to about |log pmf|·ε relative (the final exponential).
	for _, c := range []struct {
		x, n int
		p    float64
		want float64
	}{{0, 10, 0.3, 0.028247524900000005}, {10, 10, 0.3, 5.9048999999999975e-06}, {5, 10, 0.5, 0.24609375},
		{400, 1000, 0.4, 0.025744818801482515}, {1, 2000, 0.3, 1.3462733191013244e-307}, {3, 10, 0.3, 0.266827932},
		{3, 10, 0, 0}, {0, 10, 0, 1}, {7, 10, 1, 0}, {10, 10, 1, 1}} {
		got := binomPMF(c.x, c.n, c.p)
		tol := 8 * 0x1p-52 * math.Max(1, -math.Log(c.want))
		if mvRel(got, c.want) > tol {
			t.Errorf("binom(%d; %d, %g) = %g, want %g", c.x, c.n, c.p, got, c.want)
		}
	}
	// stirlerr against mpmath.
	for _, c := range [][2]float64{{16, 0.0052076559196096404}, {40, 0.0020832899383024218}, {100, 0.0008333305556349147}, {600, 0.00013888887602881678}, {5000, 1.6666666644444446e-05}} {
		n, want := c[0], c[1]
		if got := stirlerr(n); math.Abs(got-want) > 2e-16 {
			t.Errorf("stirlerr(%g) = %g, want %g", n, got, want)
		}
	}
}
