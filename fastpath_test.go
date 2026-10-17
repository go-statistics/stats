package stats

import (
	"flag"
	"math"
	"math/rand/v2"
	"slices"
	"testing"

	"github.com/go-statistics/stats/internal/dd"
)

// The fast paths added for speed must return, bit for bit, what the
// accurate paths they bypass return. -fastpath.n raises the number of
// random arguments (the default keeps go test quick; the evidence in
// docs/perf.md was produced with 10^9).
var fastpathN = flag.Int("fastpath.n", 200_000, "random arguments per fast-path identity check")

// fastpathArgs yields positive finite float64s: random bit patterns over the
// whole range, then the neighbourhoods of 1 and of powers of two, where a
// fast path is most fragile.
func fastpathArgs(n int, f func(x float64)) {
	r := rand.New(rand.NewPCG(1, 2))
	for range n {
		x := math.Float64frombits(r.Uint64() &^ (1 << 63))
		if x > 0 && x <= math.MaxFloat64 {
			f(x)
		}
	}
	for k := -2000; k <= 2000; k++ {
		f(math.Nextafter(1, 2) + float64(k)*0x1p-52)
		f(1 + float64(k)*0x1p-40)
	}
	for e := -1074; e <= 1023; e++ {
		p := math.Ldexp(1, e)
		f(p)
		f(math.Nextafter(p, 0))
		f(math.Nextafter(p, math.Inf(1)))
	}
	for range n {
		f(r.Float64()*4 + 0x1p-30) // the range of probabilities and ratios
	}
}

func TestCrLogFastPath(t *testing.T) {
	bad := 0
	fastpathArgs(*fastpathN, func(x float64) {
		if got, want := crLog(x), dd.LogAccurate(x).Float(); math.Float64bits(got) != math.Float64bits(want) {
			if bad++; bad < 10 {
				t.Errorf("crLog(%v) = %v, LogAccurate %v", x, got, want)
			}
		}
	})
}

func TestRoundDecided(t *testing.T) {
	// below a power of two half an ulp is 2^-54 of it: a value just above
	// that midpoint must not be decided by the 2^-53 margin of the binade
	r := dd.DD{Hi: 1, Lo: -0x1p-54 * (1 - 0x1p-20)}
	if _, ok := roundDecided(r, 0x1p-66); ok {
		t.Errorf("decided %v near the midpoint below 1", r)
	}
	if y, ok := roundDecided(dd.DD{Hi: 1, Lo: -0x1p-60}, 0x1p-66); !ok || y != 1 {
		t.Errorf("1 - 2^-60: %v %v", y, ok)
	}
	if _, ok := roundDecided(dd.DD{}, 0x1p-66); ok {
		t.Error("zero decided")
	}
}

// TestDotDevIdentity checks dotDD (and the deviation form) against the
// double-double sum on random, collinear and ill-conditioned inner
// products, and that the fast path decides nearly always.
func TestDotDevIdentity(t *testing.T) {
	r := rand.New(rand.NewPCG(3, 4))
	sets := *fastpathN / 100
	fast, total := 0, 0
	var undecided [5]int
	kind := 4
	check := func(a, b []float64, ma, mb float64) {
		want := dotDDExact(subMean(a, ma), subMean(b, mb))
		got, ok := dotDev(a, b, ma, mb)
		total++
		if !ok {
			undecided[kind]++
		}
		if ok {
			fast++
			if math.Float64bits(got) != math.Float64bits(want) {
				t.Errorf("n=%d: dotDev %v (decided), double-double %v", len(a), got, want)
			}
		}
		if g := devDot(a, b, ma, mb); math.Float64bits(g) != math.Float64bits(want) && !(g != g && want != want) {
			t.Errorf("devDot %v, want %v", g, want)
		}
	}
	for s := range sets {
		n := 1 + r.IntN(2000)
		if s%50 == 0 {
			n = 100_000
		}
		a, b := make([]float64, n), make([]float64, n)
		kind = s % 4
		switch kind {
		case 0: // independent normals
			for i := range a {
				a[i], b[i] = r.NormFloat64(), r.NormFloat64()
			}
		case 1: // nearly collinear, offset by 1e8
			for i := range a {
				a[i] = 1e8 + r.NormFloat64()
				b[i] = 3*a[i] + 1e-9*r.NormFloat64()
			}
		case 2: // wide exponents
			for i := range a {
				a[i] = math.Ldexp(r.Float64()-0.5, r.IntN(200)-100)
				b[i] = math.Ldexp(r.Float64()-0.5, r.IntN(200)-100)
			}
		case 3: // terms that cancel to far below their size
			h := n / 2
			for i := range h {
				a[i] = math.Ldexp(r.Float64()+0.5, r.IntN(60)-30)
				b[i] = math.Ldexp(r.Float64()+0.5, r.IntN(60)-30)
				a[h+i] = a[i]
				b[h+i] = -b[i] * (1 + 0x1p-40*r.NormFloat64())
			}
		}
		check(a, b, 0, 0)
		ma, mb := mean(a), mean(b)
		check(a, b, ma, mb)
		check(a, a, ma, ma)
	}
	for _, v := range [][2][]float64{
		{{1, 2}, {math.NaN(), 1}},
		{{1e300, 1e300}, {1e300, -1e300}},
		{{1e-300, 2e-300}, {1e-300, 1e-300}},
		{{0, 0}, {1, 2}},
		{{1}, {2}},
		{{}, {}},
	} {
		kind = 4
		check(v[0], v[1], 0, 0)
	}
	t.Logf("decided by the fast path: %d of %d; undecided by kind %v", fast, total, undecided)
	// statistical data (kinds 0-2) is decided but for a few in 10^3: data
	// with few significant bits (kind 1: 1e8 + noise) can sum exactly to a
	// midpoint between two floats, a tie only the exact path may round
	if u := undecided[0] + undecided[1] + undecided[2]; u > total/200 {
		t.Errorf("fast path left %d data sets undecided", u)
	}
}

func TestExpLogPowFastPath(t *testing.T) {
	r := rand.New(rand.NewPCG(5, 6))
	bad := 0
	ys := []float64{-1 / 2.5, 1 / 1.7, 2.5, -1.5, 0.3, 1e-3, -7.25, 100.5, 1000.1}
	fastpathArgs(*fastpathN/10, func(x float64) {
		for _, y := range ys {
			if s := math.Abs(math.Log(x) * y); s > 700 {
				continue
			}
			want := dd.Exp(dd.Mul(dd.LogAccurate(x), dd.F(y)))
			if got := expLogPow(x, y); math.Float64bits(got) != math.Float64bits(want) {
				if bad++; bad < 10 {
					t.Errorf("expLogPow(%v, %v) = %v, accurate path %v", x, y, got, want)
				}
			}
		}
		y := (r.Float64() - 0.5) * 40
		if want, got := dd.Exp(dd.Mul(dd.LogAccurate(x), dd.F(y))), expLogPow(x, y); math.Float64bits(got) != math.Float64bits(want) {
			if bad++; bad < 10 {
				t.Errorf("expLogPow(%v, %v) = %v, accurate path %v", x, y, got, want)
			}
		}
	})
}

func TestSumFastIdentity(t *testing.T) {
	r := rand.New(rand.NewPCG(7, 8))
	undecided := 0
	sets := *fastpathN / 100
	for s := range sets {
		n := 1 + r.IntN(3000)
		a := make([]float64, n)
		for i := range a {
			switch s % 3 {
			case 0:
				a[i] = r.ExpFloat64() * 10
			case 1:
				a[i] = math.Ldexp(r.NormFloat64(), r.IntN(80)-40)
			case 2:
				a[i] = math.Floor(r.Float64()*100) / 8
			}
		}
		want := sumDDExact(a)
		got, ok := sumFast(a)
		if !ok {
			undecided++
		} else if math.Float64bits(got) != math.Float64bits(want) {
			t.Errorf("n=%d: sumFast %v, double-double %v", n, got, want)
		}
		if g := sumDD(a); math.Float64bits(g) != math.Float64bits(want) {
			t.Errorf("sumDD %v, want %v", g, want)
		}
	}
	for _, a := range [][]float64{{1, math.Inf(1)}, {math.NaN()}, {}, {0, 0}, {1e300, 1e300, -1e300}} {
		if g, w := sumDD(a), sumDDExact(a); math.Float64bits(g) != math.Float64bits(w) && !(g != g && w != w) {
			t.Errorf("sumDD(%v) = %v, want %v", a, g, w)
		}
	}
	t.Logf("undecided: %d of %d", undecided, sets)
	if undecided > sets/50 {
		t.Errorf("sumFast left %d of %d undecided", undecided, sets)
	}
}

func TestRadixSortIdentity(t *testing.T) {
	r := rand.New(rand.NewPCG(9, 10))
	gen := []func() float64{
		func() float64 { return r.NormFloat64() },
		func() float64 { return math.Round(r.NormFloat64() * 3) },  // ties, zeros of one sign
		func() float64 { return -math.Round(r.Float64() * 3) },     // -0 only
		func() float64 { return math.Float64frombits(r.Uint64()) }, // every pattern, NaNs included
		func() float64 {
			return []float64{math.Inf(1), math.Inf(-1), math.MaxFloat64, -math.SmallestNonzeroFloat64, 1}[r.IntN(5)]
		},
		func() float64 { return []float64{math.Copysign(0, -1), 0, 1}[r.IntN(3)] }, // both zeros: fallback
	}
	for s := range 300 {
		n := radixMin + r.IntN(5000)
		if s%7 == 0 {
			n = r.IntN(radixMin)
		}
		g := gen[s%len(gen)]
		a := make([]float64, n)
		for i := range a {
			a[i] = g()
		}
		want := slices.Clone(a)
		slices.Sort(want)
		got := slices.Clone(a)
		sortFloats(got)
		for i := range want {
			if math.Float64bits(got[i]) != math.Float64bits(want[i]) && !(got[i] != got[i] && want[i] != want[i]) {
				t.Fatalf("set %d (n=%d): sortFloats[%d] = %v, slices.Sort %v", s, n, i, got[i], want[i])
			}
		}
		type pr struct {
			v float64
			i int
		}
		p := make([]pr, n)
		for i, v := range a {
			p[i] = pr{v, i}
		}
		wp := slices.Clone(p)
		slices.SortStableFunc(wp, func(x, y pr) int {
			switch {
			case x.v < y.v:
				return -1
			case x.v > y.v:
				return 1
			}
			return 0
		})
		if radixSortPairs(p, func(e pr) float64 { return e.v }) {
			for i := range wp {
				if p[i].i != wp[i].i {
					t.Fatalf("set %d: radixSortPairs differs from the stable sort at %d", s, i)
				}
			}
		}
	}
}

func TestDirichletLogPDFFast(t *testing.T) {
	r := rand.New(rand.NewPCG(11, 12))
	undecided, total := 0, 0
	for s := range *fastpathN / 20 {
		k := 2 + s%6
		alpha := make([]float64, k)
		for i := range alpha {
			alpha[i] = []float64{0.3, 1, 1.5, 7.25, 1e3, 1 + 1e-12}[r.IntN(6)]
		}
		d, err := NewDirichlet(alpha)
		if err != nil {
			t.Fatal(err)
		}
		x := make([]float64, k)
		tot := 0.0
		for i := range x {
			x[i] = math.Ldexp(r.Float64()+0.01, -r.IntN(40))
			tot += x[i]
		}
		for i := range x {
			x[i] /= tot
		}
		want := d.logPDF(x)
		got, ok := d.logPDFFast(x)
		total++
		if !ok {
			undecided++
			continue
		}
		if w := want.Hi + want.Lo; math.Float64bits(got) != math.Float64bits(w) {
			t.Errorf("alpha %v x %v: fast %v, accurate %v", alpha, x, got, w)
		}
	}
	if undecided > total/100 {
		t.Errorf("undecided %d of %d", undecided, total)
	}
}

// TestRadixPathsIdentity runs the statistics that sort with the radix
// sort and with the comparison sort (radixMin raised) and compares bits.
func TestRadixPathsIdentity(t *testing.T) {
	r := rand.New(rand.NewPCG(13, 14))
	n := 3000
	a, b, w := make([]float64, n), make([]float64, n), make([]float64, n)
	for i := range a {
		a[i] = math.Round(r.NormFloat64()*20) / 4
		b[i] = r.NormFloat64()
		w[i] = r.Float64()
	}
	run := func() []float64 {
		var out []float64
		add := func(v ...float64) { out = append(out, v...) }
		for _, m := range []RankMethod{RankAverage, RankMin, RankMax, RankDense, RankOrdinal} {
			rk, _ := RankData(a, m)
			add(rk...)
		}
		md, _ := Mode(a, nil)
		add(md.Mode, float64(md.Count))
		ks, _ := KSTest2Samp(a, b, nil)
		add(ks.Statistic, ks.PValue)
		cf, _ := CumFreq(a, &FreqOptions{NumBins: 17, Weights: w})
		add(cf.Freq...)
		cf, _ = CumFreq(b, &FreqOptions{NumBins: 17})
		add(cf.Freq...)
		kt, _ := Kendalltau(a, b, nil)
		add(kt.Statistic, kt.PValue)
		mw, _ := MannWhitneyU(a, b, nil)
		add(mw.Statistic, mw.PValue)
		return out
	}
	fast := run()
	defer func(m int) { radixMin = m }(radixMin)
	radixMin = 1 << 62
	slow := run()
	for i := range fast {
		if math.Float64bits(fast[i]) != math.Float64bits(slow[i]) {
			t.Fatalf("value %d: radix %v, comparison sort %v", i, fast[i], slow[i])
		}
	}
}

func TestDirichletLogPDFFastZero(t *testing.T) {
	d, _ := NewDirichlet([]float64{2, 1.5, 3})
	if y, ok := d.logPDFFast([]float64{0, 0.5, 0.5}); !ok || !math.IsInf(y, -1) {
		t.Errorf("logPDFFast at x_0 = 0 with a_0 > 1: %v %v", y, ok)
	}
}
