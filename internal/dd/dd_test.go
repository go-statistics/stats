package dd

import (
	"math"
	"math/big"
	"testing"
)

func bf(a DD) *big.Float {
	x := new(big.Float).SetPrec(300).SetFloat64(a.Hi)
	return x.Add(x, new(big.Float).SetPrec(300).SetFloat64(a.Lo))
}

func relErr(got DD, want *big.Float) float64 {
	d := new(big.Float).SetPrec(300).Sub(bf(got), want)
	r, _ := new(big.Float).Quo(d, want).Float64()
	return math.Abs(r)
}

func TestMulDivExact(t *testing.T) {
	p := F(1)
	exact := new(big.Float).SetPrec(2000).SetInt64(1)
	for k := 1; k <= 170; k++ {
		p = MulF(p, float64(k))
		exact.Mul(exact, new(big.Float).SetInt64(int64(k)))
		if e := relErr(p, exact); e > 1e-29 {
			t.Fatalf("k=%d rel err %g", k, e)
		}
	}
}

// bigAtanh2 returns 2 atanh(s) = log((1+s)/(1-s)) to ~300 bits.
func bigAtanh2(s *big.Float) *big.Float {
	const prec = 320
	s2 := new(big.Float).SetPrec(prec).Mul(s, s)
	sum := new(big.Float).SetPrec(prec)
	p := new(big.Float).SetPrec(prec).Set(s)
	for i := 1; i < 700; i += 2 {
		t := new(big.Float).SetPrec(prec).Quo(p, new(big.Float).SetPrec(prec).SetInt64(int64(i)))
		sum.Add(sum, t)
		p.Mul(p, s2)
	}
	return sum.Mul(sum, new(big.Float).SetPrec(prec).SetFloat64(2))
}

// bigLog computes log(x) to ~300 bits with math/big (atanh series; ln 2 =
// 2 atanh(1/3)).
func bigLog(x float64) *big.Float {
	const prec = 320
	m, k := math.Frexp(x)
	bm := new(big.Float).SetPrec(prec).SetFloat64(m)
	one := new(big.Float).SetPrec(prec).SetFloat64(1)
	num := new(big.Float).SetPrec(prec).Sub(bm, one)
	den := new(big.Float).SetPrec(prec).Add(bm, one)
	sum := bigAtanh2(new(big.Float).SetPrec(prec).Quo(num, den))
	third := new(big.Float).SetPrec(prec).Quo(one, new(big.Float).SetPrec(prec).SetInt64(3))
	ln2 := bigAtanh2(third)
	return sum.Add(sum, new(big.Float).SetPrec(prec).Mul(ln2, new(big.Float).SetPrec(prec).SetInt64(int64(k))))
}

func TestLog(t *testing.T) {
	worst := 0.0
	for i := 0; i < 3000; i++ {
		x := math.Exp(-700 + 1400*float64(i)/3000 + 1e-3*float64(i%11))
		if i%3 == 0 {
			x = 1 + float64(i-1500)*1e-6
		}
		want := bigLog(x)
		if want.Sign() == 0 {
			continue
		}
		if e := relErr(LogAccurate(x), want); e > 0x1p-100 {
			t.Errorf("LogAccurate(%v) relative error %g", x, e)
		}
		if e := relErr(Log(x), want); e > worst {
			worst = e
			if e > 0x1p-72 {
				t.Logf("x=%v rel %g got %v want %s", x, e, Log(x), want.Text('g', 40))
			}
		}
	}
	t.Logf("dd.Log worst relative error %g (2^%.1f)", worst, math.Log2(worst))
	if worst > 0x1p-72 {
		t.Errorf("dd.Log relative error %g > 2^-72", worst)
	}
}

// bigExp computes exp(a) to ~300 bits: reduce by 2^-30, Taylor, square.
func bigExp(a *big.Float) *big.Float {
	const prec = 400
	r := new(big.Float).SetPrec(prec).SetMantExp(a, -30)
	sum := new(big.Float).SetPrec(prec).SetFloat64(1)
	term := new(big.Float).SetPrec(prec).SetFloat64(1)
	for i := 1; i < 30; i++ {
		term.Mul(term, r)
		term.Quo(term, new(big.Float).SetPrec(prec).SetInt64(int64(i)))
		sum.Add(sum, term)
	}
	for i := 0; i < 30; i++ {
		sum.Mul(sum, sum)
	}
	return sum
}

func TestExp(t *testing.T) {
	worst := 0.0
	for i := 0; i < 4000; i++ {
		hi := -740 + 1449*float64(i)/4000 + 1e-4*float64(i%13)
		a := DD{hi, hi * 0x1p-60 * float64(i%5-2)}
		a = AddF(a, 0) // normalise
		want, _ := bigExp(bf(a)).Float64()
		got := Exp(a)
		if want == 0 || math.IsInf(want, 0) {
			if got != want {
				t.Errorf("Exp(%v) = %v, want %v", a, got, want)
			}
			continue
		}
		u := math.Abs(want)
		_, e := math.Frexp(u)
		ulp := math.Ldexp(1, e-53)
		if u < 0x1p-1022 {
			ulp = 0x1p-1074
		}
		// error against the exact value, not the rounded one
		ex := new(big.Float).SetPrec(400).Sub(new(big.Float).SetFloat64(got), bigExp(bf(a)))
		ee, _ := ex.Float64()
		if err := math.Abs(ee) / ulp; err > worst && u >= 0x1p-1022 {
			worst = err
		} else if err > 1 {
			t.Errorf("subnormal Exp(%v) error %.3f ulp", a, err)
		}
	}
	t.Logf("dd.Exp worst error %.3f ulp", worst)
	if worst > 1 {
		t.Errorf("dd.Exp error %.3f ulp > 1", worst)
	}
	if !math.IsNaN(Exp(DD{math.NaN(), 0})) || Exp(DD{-800, 0}) != 0 || !math.IsInf(Exp(DD{800, 0}), 1) {
		t.Error("Exp specials")
	}
}

func TestExpMant(t *testing.T) {
	worst := 0.0
	for i := 0; i < 2000; i++ {
		hi := -700 + 1400*float64(i)/2000 + 1e-4*float64(i%13)
		a := DD{hi, hi * 0x1p-58 * float64(i%5-2)}
		a = AddF(a, 0)
		m, k := ExpMant(a)
		got := new(big.Float).SetPrec(400).SetMantExp(bf(m), k)
		want := bigExp(bf(a))
		d := new(big.Float).SetPrec(400).Sub(got, want)
		r, _ := new(big.Float).Quo(d, want).Float64()
		worst = math.Max(worst, math.Abs(r))
	}
	t.Logf("ExpMant worst relative error 2^%.1f", math.Log2(worst))
	if worst > 0x1p-75 {
		t.Errorf("ExpMant relative error %g", worst)
	}
	for _, x := range []float64{math.NaN(), 2e7, -2e7} {
		m, _ := ExpMant(DD{x, 0})
		if !(m.Hi != m.Hi || math.IsInf(m.Hi, 1) || m.Hi == 0) {
			t.Errorf("ExpMant(%v) = %v", x, m)
		}
	}
}

func TestSpecials(t *testing.T) {
	if s := Sqrt(DD{0, 0}); s.Hi != 0 {
		t.Errorf("Sqrt(0) = %v", s)
	}
	if s := Sqrt(DD{-1, 0}); s.Hi == s.Hi {
		t.Errorf("Sqrt(-1) = %v", s)
	}
	if s := Sqrt(DD{2, 0}); math.Abs(s.Hi*s.Hi-2) > 1e-15 {
		t.Errorf("Sqrt(2) = %v", s)
	}
	if l := Log(0); !math.IsInf(l.Hi, -1) {
		t.Errorf("Log(0) = %v", l)
	}
	if l := Log(math.Inf(1)); !math.IsInf(l.Hi, 1) {
		t.Errorf("Log(Inf) = %v", l)
	}
	if l := LogAccurate(-1); l.Hi == l.Hi {
		t.Errorf("LogAccurate(-1) = %v", l)
	}
	if l := Log(-1); l.Hi == l.Hi {
		t.Errorf("Log(-1) = %v", l)
	}
	if q := DivF(DD{1, 0}, 3); math.Abs(q.Hi-1.0/3) > 1e-16 {
		t.Errorf("DivF = %v", q)
	}
}
