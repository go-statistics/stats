// Package dd is a minimal double-double ("float128-lite") arithmetic used
// where a special function must carry more than 53 bits through an
// intermediate step: the argument of an exponential, a long product, a
// cancellation. A value is Hi + Lo with |Lo| <= ulp(Hi)/2.
//
// The error-free transformations are the classical ones: TwoSum (Knuth,
// TAOCP vol. 2, 4.2.2), Fast2Sum (Dekker 1971) and TwoProd through a fused
// multiply-add (Ogita, Rump and Oishi, "Accurate sum and dot product", SIAM
// J. Sci. Comput. 26(6), 2005). Algorithms for the compound operations follow
// Joldes, Muller and Popescu, "Tight and rigorous error bounds for basic
// building blocks of double-word arithmetic", ACM TOMS 44(2), 2017
// (DWPlusDW: Algorithm 6, DWTimesDW: Algorithm 12, DWDivDW: Algorithm 18).
//
// This package is a copy of go-numerics/special's internal/dd (BSD-3-Clause),
// which an internal package cannot share across modules.
package dd

import "math"

// DD is an unevaluated sum Hi + Lo.
type DD struct{ Hi, Lo float64 }

// TwoSum returns s, e with s = fl(a+b) and s + e = a + b exactly.
func TwoSum(a, b float64) (s, e float64) {
	s = a + b
	bb := s - a
	e = (a - (s - bb)) + (b - bb)
	return
}

// FastTwoSum requires |a| >= |b| (or a == 0).
func FastTwoSum(a, b float64) (s, e float64) {
	s = a + b
	e = b - (s - a)
	return
}

// TwoProd returns p, e with p = fl(a*b) and p + e = a*b exactly (barring
// underflow).
//
// The explicit float64 conversion is load-bearing: the Go specification
// allows a*b to be fused with a later addition (an FMA on arm64, ppc64le,
// s390x, riscv64, loong64), which would break every error-free
// transformation built on p. A conversion forces the rounding.
func TwoProd(a, b float64) (p, e float64) {
	p = float64(a * b)
	e = math.FMA(a, b, -p)
	return
}

// F returns x as a DD.
func F(x float64) DD { return DD{x, 0} }

// Float rounds to the nearest float64.
func (a DD) Float() float64 { return a.Hi + a.Lo }

// Add returns a + b (DWPlusDW, relative error <= 3u^2).
//
// It is written out (rather than through TwoSum/FastTwoSum) to stay within
// the inliner's budget: it is the most frequent operation in the package.
func Add(a, b DD) DD {
	s := a.Hi + b.Hi
	bb := s - a.Hi
	t := (a.Hi - (s - bb)) + (b.Hi - bb)
	u := a.Lo + b.Lo
	bb = u - a.Lo
	v := (a.Lo - (u - bb)) + (b.Lo - bb)
	t += u
	h := s + t
	t -= h - s
	t += v
	s = h + t
	return DD{s, t - (s - h)}
}

// AddF returns a + b for a float64 b (DWPlusFP, relative error <= 2u^2).
func AddF(a DD, b float64) DD {
	s, t := TwoSum(a.Hi, b)
	t += a.Lo
	s, t = FastTwoSum(s, t)
	return DD{s, t}
}

// Neg returns -a.
func Neg(a DD) DD { return DD{-a.Hi, -a.Lo} }

// Sub returns a - b.
func Sub(a, b DD) DD { return Add(a, Neg(b)) }

// Mul returns a * b (DWTimesDW with FMA, relative error <= 5u^2).
func Mul(a, b DD) DD {
	p, e := TwoProd(a.Hi, b.Hi)
	t := a.Hi * b.Lo
	t = math.FMA(a.Lo, b.Hi, t)
	e += t
	p, e = FastTwoSum(p, e)
	return DD{p, e}
}

// MulF returns a * b for a float64 b (DWTimesFP, relative error <= 2u^2).
func MulF(a DD, b float64) DD {
	p, e := TwoProd(a.Hi, b)
	e = math.FMA(a.Lo, b, e)
	p, e = FastTwoSum(p, e)
	return DD{p, e}
}

// MulFF returns the exact product a*b of two float64.
func MulFF(a, b float64) DD {
	p, e := TwoProd(a, b)
	return DD{p, e}
}

// Div returns a / b (DWDivDW, relative error <= 15u^2 + 56u^3).
func Div(a, b DD) DD {
	q := a.Hi / b.Hi
	// r = a - q*b
	r := Sub(a, MulF(b, q))
	q2 := r.Hi / b.Hi
	r = Sub(r, MulF(b, q2))
	q3 := r.Hi / b.Hi
	s, e := FastTwoSum(q, q2)
	return AddF(DD{s, e}, q3)
}

// DivF returns a / b for float64 b.
func DivF(a DD, b float64) DD { return Div(a, DD{b, 0}) }

// Sqrt returns sqrt(a) (one Newton step from the float64 root).
func Sqrt(a DD) DD {
	if a.Hi <= 0 {
		return DD{math.Sqrt(a.Hi), 0}
	}
	s := math.Sqrt(a.Hi)
	// s + (a - s^2)/(2s)
	r := Sub(a, MulFF(s, s))
	return AddF(DD{s, 0}, r.Hi/(2*s))
}

// ln2 split as a double-double: Ln2Hi + Ln2Lo = ln 2 to 106 bits.
const (
	Ln2Hi = 6.93147180559945286227e-01
	Ln2Lo = 2.31904681384629955842e-17
)

// Exp returns exp(a.Hi + a.Lo) rounded to float64: the double-double
// mantissa from ExpMant is accurate to about 2^-76, so the result is
// correctly rounded except in cases closer to a rounding boundary than that.
func Exp(a DD) float64 {
	m, k := ExpMant(a)
	return math.Ldexp(m.Hi+m.Lo, k)
}

// ExpMant returns m, k with exp(a) = m * 2^k, m in about [0.99, 2.02] as a
// double-double accurate to about 2^-76 relative, so that a caller can
// multiply further factors in before the single final rounding. k is not
// limited to the float64 exponent range: a scaled result may be combined
// with others before math.Ldexp decides overflow or underflow. For
// |a.Hi| > 1e7 it returns m = +Inf or m = 0 with k = 0 (NaN for NaN).
//
// Method (Tang, "Table-driven implementation of the exponential function in
// IEEE floating-point arithmetic", ACM TOMS 15(2), 1989, carried to
// double-double): a = (64k + j) ln2/64 + r, |r| <= ln2/128, the reduction
// exact (ln2/64 split in three parts, the first with 32 significant bits so
// that n times it is exact for |n| < 2^21, the second through TwoProd);
// exp(r) = 1 + r + r²/2 + r³ P(r) with 1 + r + r²/2 in double-double and
// P (to r^9/9!, truncation below 2^-86) in float64; 2^(j/64) from a 300-bit
// mpmath table (zz_exptable.go).
func ExpMant(a DD) (m DD, k int) {
	x := a.Hi
	switch {
	case x != x:
		return DD{x, 0}, 0
	case x > 1e7:
		return DD{math.Inf(1), 0}, 0
	case x < -1e7:
		return DD{}, 0
	}
	n := math.Floor(x*(1/ln2o64Hi) + 0.5)
	var s, t float64
	if math.Abs(n) < 1<<21 {
		rh := x - n*ln2o64Hi // n*ln2o64Hi exact; the difference exact (Sterbenz)
		p1, e1 := TwoProd(n, ln2o64Mid)
		s, t = TwoSum(rh, -p1)
		t += a.Lo - e1 - n*ln2o64Lo
		s, t = FastTwoSum(s, t)
	} else {
		r := Sub(a, MulFF(n, ln2o64Hi))
		r = Sub(r, Add(MulFF(n, ln2o64Mid), MulFF(n, ln2o64Lo)))
		s, t = r.Hi, r.Lo
	}
	ni := int(n)
	j := ni & 63
	k = (ni - j) / 64
	// r = s + t; r² = s² + 2st (t² < 2^-200)
	q, qe := TwoProd(s, s)
	p := s * q * (1.0/6 + s*(1.0/24+s*(1.0/120+s*(1.0/720+s*(1.0/5040+s*(1.0/40320+s*(1.0/362880)))))))
	u, v := TwoSum(1, s)
	v += t + 0.5*qe + s*t + p
	e := AddF(DD{u, v}, 0.5*q)
	return Mul(e, expTable[j]), k
}

// Ldexp returns a * 2^k for a double-double (both parts scaled).
func Ldexp(a DD, k int) DD {
	return DD{math.Ldexp(a.Hi, k), math.Ldexp(a.Lo, k)}
}

// Log returns log(x) for finite x > 0 as a double-double accurate to about
// 2^-72 relative (measured 2^-72.9; the float64 tail limits it).
//
// Method (table-driven, after Tang, "Table-driven implementation of the
// logarithm function in IEEE floating-point arithmetic", ACM TOMS 16(4),
// 1990): x = 2^k m with m in [1/sqrt2, sqrt2); c = j/128 is the grid point
// nearest m, and log m = log c + 2 atanh(s), s = (m-c)/(m+c). m-c is exact
// (Sterbenz), s is formed in double-double, |s| < 1/360, and
// 2 atanh(s) = 2s + s³ (2/3 + 2s²/5 + 2s⁴/7) with the second part (below
// 2^-16 of the first) in float64. log c comes from a 300-bit mpmath table
// (zz_logtable.go); k ln2 is added exactly.
//
// Log(0) = -Inf, Log(+Inf) = +Inf, NaN for x < 0 or NaN (as math.Log).
func Log(x float64) DD {
	if !(x > 0) || x > math.MaxFloat64 {
		return DD{math.Log(x), 0}
	}
	k := 0
	if x < 0x1p-1022 {
		x *= 0x1p54 // normalise a subnormal
		k = -54
	}
	b := math.Float64bits(x)
	k += int(b>>52) - 1023
	m := math.Float64frombits(b&(1<<52-1) | 1023<<52) // in [1, 2)
	if m > math.Sqrt2 {
		m *= 0.5
		k++
	}
	j := math.Floor(m*128 + 0.5)
	c := j * (1.0 / 128)
	f := m - c // exact
	h, l := TwoSum(m, c)
	sh := f / h
	sl := (math.FMA(-sh, h, f) - sh*l) / h
	z := sh * sh
	tail := sh * z * (2.0/3 + z*(2.0/5+z*(2.0/7)))
	lm := AddF(DD{2 * sh, 2 * sl}, tail)
	lm = Add(lm, logTable[int(j)-logTableLo])
	if k == 0 {
		return lm
	}
	kk := float64(k)
	// k*Ln2Hi is exact: Ln2Hi has 32 significant bits and |k| < 1100.
	kh, kl := FastTwoSum(kk*ln2Hi32, kk*ln2Lo32)
	return Add(DD{kh, kl}, lm)
}

// LogAccurate is Log carried further, to about 2^-104 relative: the atanh
// series keeps its terms through s⁴/5 in double-double and the quotient s
// comes from a full double-double division. It costs about five times Log
// and is for the few places where the result is a difference of logarithms
// that cancel to below 2^-30 of their size (Digamma near its zeros).
func LogAccurate(x float64) DD {
	if !(x > 0) || x > math.MaxFloat64 {
		return DD{math.Log(x), 0}
	}
	m, k := math.Frexp(x) // x = m * 2^k, m in [0.5, 1); Frexp normalises subnormals
	if m < math.Sqrt2/2 {
		m *= 2
		k--
	}
	j := math.Floor(m*128 + 0.5)
	c := j / 128
	num := m - c // exact
	h, l := TwoSum(m, c)
	s := Div(DD{num, 0}, DD{h, l})
	s2 := Mul(s, s)
	z := s2.Hi
	// series = 1 + s2/3 + s2^2/5 + s2^3 (1/7 + s2/9 + s2^2/11 + s2^3/13)
	t := Add(Div(s2, DD{3, 0}), DivF(Mul(s2, s2), 5))
	t = AddF(t, z*z*z*(1.0/7+z*(1.0/9+z*(1.0/11+z/13))))
	t = AddF(t, 1)
	lm := MulF(Mul(s, t), 2)
	lm = Add(lm, logTable[int(j)-logTableLo])
	kk := float64(k)
	return Add(Add(MulFF(kk, Ln2Hi), MulFF(kk, Ln2Lo)), lm)
}

// ln2 split so that ln2Hi32 has 32 significant bits (k*ln2Hi32 exact for
// |k| < 2^21) and ln2Hi32 + ln2Lo32 = ln 2 to 2^-85.
const (
	ln2Hi32 = 0.6931471803691238
	ln2Lo32 = 1.9082149292705877e-10
)
