package stats

import (
	"fmt"
	"math"
	"testing"

	"github.com/go-ndarray/ndarray"
	"github.com/go-ndarray/ndarray/random"
)

type mvnFixture struct {
	Cases []struct {
		Mean          mvFloats `json:"mean"`
		Cov           mvFloats `json:"cov"`
		D             int      `json:"d"`
		AllowSingular bool     `json:"allow_singular"`
		Points        mvFloats `json:"points"`
		LogPDF        mvFloats `json:"logpdf"`
		PDF           mvFloats `json:"pdf"`
		Entropy       mvFloat  `json:"entropy"`
		MpLogPDF      mvFloats `json:"mp_logpdf"`
		MpEntropy     *mvFloat `json:"mp_entropy"`
		MpPDF         mvFloats `json:"mp_pdf"`
		CDFPoints     mvFloats `json:"cdf_points"`
		CDF           mvFloats `json:"cdf"`
		Lower         mvFloats `json:"lower"`
		CDFLower      mvFloats `json:"cdf_lower"`
		Rvs           []struct {
			Seed uint64   `json:"seed"`
			Size int      `json:"size"`
			Hex  mvFloats `json:"hex"`
		} `json:"rvs"`
	} `json:"cases"`
	BVN []struct {
		R, H, K, Scipy, Mp mvFloat
	} `json:"bvn"`
	Rect []struct {
		Cov   mvFloats `json:"cov"`
		Lower mvFloats `json:"lower"`
		Upper mvFloats `json:"upper"`
		Scipy mvFloat  `json:"scipy"`
	} `json:"rect"`
	Lattice []struct {
		Dim, M, N int
		Q         mvFloats
	} `json:"lattice"`
	Fits []struct {
		X       mvFloats `json:"x"`
		N, D    int
		FixMean mvFloats `json:"fix_mean"`
		FixCov  mvFloats `json:"fix_cov"`
		Mean    mvFloats `json:"mean"`
		Cov     mvFloats `json:"cov"`
	} `json:"fits"`
	Invalid []struct {
		Cov    mvFloats `json:"cov"`
		D      int      `json:"d"`
		Raises bool     `json:"raises"`
	} `json:"invalid"`
}

func newMVN(t testing.TB, mean, cov []float64, d int, sing bool) *MultivariateNormal {
	t.Helper()
	m, err := NewMultivariateNormal(mean, mvArr(cov, d, d), &MultivariateNormalOptions{AllowSingular: sing})
	if err != nil {
		t.Fatal(err)
	}
	return m
}

// TestMultivariateNormalParity checks the densities, entropy, cdf and
// draws against SciPy, and the densities and entropy against mpmath.
func TestMultivariateNormalParity(t *testing.T) {
	var fx mvnFixture
	loadMV(t, "mvn", &fx)
	var ourLog, spLog, ourPDF, spPDF, vsSciPy, ent, spEnt, cdf, cdfL mvMax
	var draws, draws1 mvDraws
	var qmc mvMax
	flips, nq, same := 0, 0, 0
	for ci, c := range fx.Cases {
		m := newMVN(t, c.Mean.f(), c.Cov.f(), c.D, c.AllowSingular)
		pts := c.Points.f()
		lp, err := m.LogPDFPoints(mvArr(pts, len(pts)/c.D, c.D))
		if err != nil {
			t.Fatal(err)
		}
		pd, _ := m.PDFPoints(mvArr(pts, len(pts)/c.D, c.D))
		for i := range lp {
			w := fmt.Sprintf("case %d (d=%d sing=%v) point %d", ci, c.D, c.AllowSingular, i)
			if c.MpLogPDF != nil {
				ex := float64(c.MpLogPDF[i])
				ourLog.add(mvRel(lp[i], ex), w)
				spLog.add(mvRel(float64(c.LogPDF[i]), ex), w)
				exP := float64(c.MpPDF[i])
				if exP > 1e-300 {
					ourPDF.add(mvRel(pd[i], exP), w)
					spPDF.add(mvRel(float64(c.PDF[i]), exP), w)
				}
			} else {
				vsSciPy.add(mvRel(lp[i], float64(c.LogPDF[i])), w)
				vsSciPy.add(mvRel(pd[i], float64(c.PDF[i])), w)
			}
		}
		if c.MpEntropy != nil {
			ent.add(mvRel(m.Entropy(), float64(*c.MpEntropy)), fmt.Sprint("case ", ci))
			spEnt.add(mvRel(float64(c.Entropy), float64(*c.MpEntropy)), fmt.Sprint("case ", ci))
		} else {
			vsSciPy.add(mvRel(m.Entropy(), float64(c.Entropy)), fmt.Sprint("entropy case ", ci))
		}
		cp := c.CDFPoints.f()
		if c.D >= 3 {
			// The randomized lattice rule is costly (about 0.1 s a point,
			// ten times that under -race): two points per case.
			cp = cp[:2*c.D]
		}
		o := &MultivariateNormalCDFOptions{RNG: random.DefaultRNG(7)}
		got, err := m.CDFPoints(mvArr(cp, len(cp)/c.D, c.D), o)
		if err != nil {
			t.Fatal(err)
		}
		for i := range got {
			w := fmt.Sprintf("case %d d=%d point %d", ci, c.D, i)
			if c.D <= 2 {
				cdf.add(math.Abs(got[i]-float64(c.CDF[i])), w)
			} else {
				qmc.add(math.Abs(got[i]-float64(c.CDF[i])), w)
				nq++
				if mvRel(got[i], float64(c.CDF[i])) < 1e-12 {
					same++
				}
			}
		}
		o = &MultivariateNormalCDFOptions{RNG: random.DefaultRNG(8), Lower: c.Lower.f()}
		got, _ = m.CDFPoints(mvArr(cp, len(cp)/c.D, c.D), o)
		for i := range got {
			w := fmt.Sprintf("case %d d=%d point %d", ci, c.D, i)
			if c.D <= 2 {
				cdfL.add(math.Abs(got[i]-float64(c.CDFLower[i])), w)
			} else {
				qmc.add(math.Abs(got[i]-float64(c.CDFLower[i])), w)
				nq++
				if mvRel(got[i], float64(c.CDFLower[i])) < 1e-12 {
					same++
				}
			}
		}
		for _, r := range c.Rvs {
			a, err := m.Rvs(random.DefaultRNG(r.Seed), r.Size)
			if err != nil {
				t.Fatal(err)
			}
			v, _ := ndarray.Data[float64](a)
			if c.D == 1 {
				draws1.add(v, r.Hex.f())
			} else {
				draws.add(v, r.Hex.f())
			}
			for i := range v {
				if mvULP(v[i], float64(r.Hex[i])) > 1e6 {
					flips++
					break
				}
			}
		}
	}
	t.Logf("logpdf rel err vs mpmath: ours %.3g (%s), SciPy %.3g (%s)", ourLog.v, ourLog.where, spLog.v, spLog.where)
	t.Logf("pdf rel err vs mpmath: ours %.3g (%s), SciPy %.3g (%s)", ourPDF.v, ourPDF.where, spPDF.v, spPDF.where)
	t.Logf("entropy rel err vs mpmath: ours %.3g, SciPy %.3g", ent.v, spEnt.v)
	t.Logf("singular cases, rel diff vs SciPy: %.3g (%s)", vsSciPy.v, vsSciPy.where)
	t.Logf("cdf abs diff vs SciPy: %.3g (%s); with lower limit %.3g (%s)", cdf.v, cdf.where, cdfL.v, cdfL.where)
	t.Logf("cdf d>=3 (randomized lattice): max abs diff vs SciPy %.3g (%s); %d of %d values equal SciPy's to 1e-12", qmc.v, qmc.where, same, nq)
	t.Logf("rvs d=1: %d draws, %d differ", draws1.n, draws1.diff)
	t.Logf("rvs d>=2: %d draws, %d differ, max %g ULP; %d draw sets with a sign flip", draws.n, draws.diff, draws.maxULP, flips)
	if ourLog.v > 4e-16 {
		t.Errorf("logpdf: rel err %g vs mpmath at %s", ourLog.v, ourLog.where)
	}
	if ourPDF.v > 4e-16 {
		t.Errorf("pdf: rel err %g vs mpmath at %s", ourPDF.v, ourPDF.where)
	}
	if ent.v > 4e-16 {
		t.Errorf("entropy: rel err %g vs mpmath", ent.v)
	}
	if vsSciPy.v > 1e-12 {
		t.Errorf("singular: rel diff %g vs SciPy at %s", vsSciPy.v, vsSciPy.where)
	}
	if draws1.diff != 0 {
		t.Errorf("rvs d=1: %d of %d draws differ from NumPy's", draws1.diff, draws1.n)
	}
	if qmc.v > 2e-5 {
		t.Errorf("cdf d>=3: abs diff %g vs SciPy at %s", qmc.v, qmc.where)
	}
	if cdf.v > 1e-12 || cdfL.v > 1e-12 {
		t.Errorf("cdf: abs diff %g / %g vs SciPy", cdf.v, cdfL.v)
	}
}

// TestBivariateNormalTails measures the bivariate normal cdf against
// mpmath, where SciPy's inclusion–exclusion loses the lower tail.
func TestBivariateNormalTails(t *testing.T) {
	var fx mvnFixture
	loadMV(t, "mvn", &fx)
	var ours, sp mvMax
	for _, b := range fx.BVN {
		r := float64(b.R)
		m := newMVN(t, []float64{0, 0}, []float64{1, r, r, 1}, 2, false)
		got, err := m.CDF([]float64{float64(b.H), float64(b.K)}, nil)
		if err != nil {
			t.Fatal(err)
		}
		w := fmt.Sprintf("r=%g h=%g k=%g (exact %g)", r, float64(b.H), float64(b.K), float64(b.Mp))
		ours.add(mvRel(got, float64(b.Mp)), w)
		sp.add(mvRel(float64(b.Scipy), float64(b.Mp)), w)
	}
	t.Logf("bivariate cdf rel err vs mpmath: ours %.3g (%s); SciPy %.3g (%s)", ours.v, ours.where, sp.v, sp.where)
	if ours.v > 2e-15 {
		t.Errorf("bivariate cdf: rel err %g at %s", ours.v, ours.where)
	}
	for _, r := range fx.Rect {
		m := newMVN(t, nil, r.Cov.f(), 2, false)
		got, err := m.CDF(r.Upper.f(), &MultivariateNormalCDFOptions{Lower: r.Lower.f()})
		if err != nil {
			t.Fatal(err)
		}
		if math.Abs(got-float64(r.Scipy)) > 1e-14 {
			t.Errorf("rectangle %v..%v: got %g, SciPy %g", r.Lower.f(), r.Upper.f(), got, float64(r.Scipy))
		}
	}
}

// TestCBCLattice compares the lattice generating vectors with SciPy's
// _cbc_lattice. They agree except where the CBC criterion has exact ties
// (which index wins then depends on the FFT's rounding).
func TestCBCLattice(t *testing.T) {
	var fx mvnFixture
	loadMV(t, "mvn", &fx)
	same, n := 0, 0
	for _, l := range fx.Lattice {
		if l.M > 20011 {
			continue // slow; the large sizes agree too (checked when written)
		}
		n++
		q, nq := cbcLattice(l.Dim, l.M)
		if nq != l.N || q[0] != float64(l.Q[0]) {
			t.Errorf("dim %d m %d: n %d (SciPy %d), q[0] %g", l.Dim, l.M, nq, l.N, q[0])
		}
		eq := true
		for i := range q {
			eq = eq && q[i] == float64(l.Q[i])
		}
		if eq {
			same++
		}
	}
	t.Logf("lattices identical to SciPy's: %d of %d", same, n)
	if same < n*2/3 {
		t.Errorf("only %d of %d lattices agree", same, n)
	}
	// The FFT against the direct transform, for both kinds of length.
	for _, n := range []int{1, 2, 8, 7, 12, 210} {
		x := make([]complex128, n)
		for i := range x {
			x[i] = complex(math.Sin(float64(i)*1.3), math.Cos(float64(i*i)))
		}
		got := fft(x, false)
		back := fft(got, true)
		for k := range n {
			var s complex128
			for j := range n {
				s += x[j] * cmplxRect(-2*math.Pi*float64(j*k%n)/float64(n))
			}
			if d := s - got[k]; math.Hypot(real(d), imag(d)) > 1e-12*float64(n) {
				t.Errorf("fft n=%d k=%d: %v, direct %v", n, k, got[k], s)
			}
			if d := back[k] - x[k]; math.Hypot(real(d), imag(d)) > 1e-13*float64(n) {
				t.Errorf("inverse fft n=%d k=%d", n, k)
			}
		}
	}
}

func cmplxRect(a float64) complex128 { return complex(math.Cos(a), math.Sin(a)) }

func TestMultivariateNormalFit(t *testing.T) {
	var fx mvnFixture
	loadMV(t, "mvn", &fx)
	var dm, dc mvMax
	for i, f := range fx.Fits {
		o := &MultivariateNormalFitOptions{FixMean: f.FixMean.f()}
		if f.FixMean == nil {
			o.FixMean = nil
		}
		if f.FixCov != nil {
			o.FixCov = mvArr(f.FixCov.f(), f.D, f.D)
		}
		mean, cov, err := FitMultivariateNormal(mvArr(f.X.f(), f.N, f.D), o)
		if err != nil {
			t.Fatal(err)
		}
		cv, _ := ndarray.Data[float64](cov)
		for j := range mean {
			dm.add(mvULP(mean[j], float64(f.Mean[j])), fmt.Sprint("fit ", i))
		}
		for j := range cv {
			dc.add(mvRel(cv[j], float64(f.Cov[j])), fmt.Sprint("fit ", i))
		}
	}
	t.Logf("fit: mean max %g ULP from SciPy (%s); cov max rel diff %.3g (%s)", dm.v, dm.where, dc.v, dc.where)
	if dm.v > 0 || dc.v > 1e-14 {
		t.Errorf("fit: mean %g ULP, cov rel %g", dm.v, dc.v)
	}
	for _, c := range fx.Invalid {
		_, err := NewMultivariateNormal(make([]float64, c.D), mvArr(c.Cov.f(), c.D, c.D), nil)
		if (err != nil) != c.Raises {
			t.Errorf("cov %v: error %v, SciPy raises %v", c.Cov.f(), err, c.Raises)
		}
	}
}
