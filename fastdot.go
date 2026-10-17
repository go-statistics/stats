package stats

import (
	"math"

	"github.com/go-statistics/stats/internal/dd"
)

// The inner products of the correlation functions are double-double sums
// of exact products (dotDDExact), rounded once. That costs about 25
// floating-point operations per term in one dependency chain. dotDev
// computes the same rounded value with a compensated inner product
// (Ogita, Rump and Oishi's Dot2: TwoProd and TwoSum per term, the errors
// summed in float64) on two independent accumulators, and a running bound
// on its own error. When the bound shows that every value within it rounds
// to the same float64 — and that the double-double sum, whose error is
// bounded by 3u²·n·Σ|p| (AccurateDWPlusDW, Joldes, Muller and Popescu
// 2017), rounds there too — that float64 is returned; otherwise (about
// once in 10^4 data sets, for sums that fall exactly on a midpoint between
// two floats, and for data near underflow or overflow) the
// double-double sum is computed. The result is the one dotDDExact returns,
// bit for bit, and about 3x cheaper.

const u53 = 0x1p-53

// dotDev returns Σ (a[i]-ma)·(b[i]-mb), each difference rounded as
// deviations rounds it, and whether that is dotDDExact's value (false: the
// caller must compute it so).
func dotDev(a, b []float64, ma, mb float64) (float64, bool) {
	n := len(a)
	b = b[:n]
	var s0, c0, s1, c1, ea, pa float64
	i := 0
	for ; i+1 < n; i += 2 {
		x0, y0 := a[i]-ma, b[i]-mb
		x1, y1 := a[i+1]-ma, b[i+1]-mb
		p0 := float64(x0 * y0)
		p1 := float64(x1 * y1)
		q0 := math.FMA(x0, y0, -p0)
		q1 := math.FMA(x1, y1, -p1)
		t0 := s0 + p0
		t1 := s1 + p1
		v0 := t0 - s0
		v1 := t1 - s1
		f0 := (s0 - (t0 - v0)) + (p0 - v0)
		f1 := (s1 - (t1 - v1)) + (p1 - v1)
		s0, s1 = t0, t1
		c0 += q0 + f0
		c1 += q1 + f1
		ea += (math.Abs(q0) + math.Abs(f0)) + (math.Abs(q1) + math.Abs(f1))
		pa += math.Abs(p0) + math.Abs(p1)
	}
	if i < n {
		x0, y0 := a[i]-ma, b[i]-mb
		p0 := float64(x0 * y0)
		q0 := math.FMA(x0, y0, -p0)
		t0 := s0 + p0
		v0 := t0 - s0
		f0 := (s0 - (t0 - v0)) + (p0 - v0)
		s0 = t0
		c0 += q0 + f0
		ea += math.Abs(q0) + math.Abs(f0)
		pa += math.Abs(p0)
	}
	// Σ|p| beyond 2^1000 may have overflowed somewhere; below 2^-900 the
	// products may have lost bits to underflow; NaN: let the exact path
	// say what it says.
	// The exact sum is h + l + Σ(q+f). The float64 sums c0, c1 err by at
	// most γ_2n·Σ(|q|+|f|) together; ea, pa (float64 sums of nonnegative
	// terms) are below their exact values by at most a factor 1-γ_n, which
	// the constants absorb (n·u < 2^-20 is required).
	fn := float64(n)
	if !(pa < 0x1p1000 && pa > 0x1p-900) || fn*u53 > 0x1p-20 {
		return 0, false
	}
	h, l := dd.TwoSum(s0, s1)
	l2 := l + c0 + c1
	errFast := 4.4*fn*u53*ea + 2.2*u53*(math.Abs(l)+math.Abs(c0)+math.Abs(c1))
	errDD := 3.3 * fn * u53 * u53 * pa
	y, rem := dd.TwoSum(h, l2)
	return y, decidedAbs(y, rem, errFast+errDD)
}

// decidedAbs reports whether every value within err of y + rem rounds to
// y (y = fl(y + rem)): |rem| plus err stays below half an ulp of y on the
// side of rem.
func decidedAbs(y, rem, err float64) bool {
	b := math.Float64bits(y)
	if e := b & (0x7ff << 52); e == 0 || e == 0x7ff<<52 {
		return false // zero, subnormal, infinite or NaN: no claim
	}
	p := math.Float64frombits(b & (0x7ff << 52))
	half := p * 0x1p-53
	if b&(1<<52-1) == 0 && (rem < 0) != (y < 0) {
		half = p * 0x1p-54
	}
	return math.Abs(rem) < half-err
}

// dotDD is Σ a[i]·b[i] in double-double, rounded once (dotDDExact's value,
// through dotDev when it can vouch for it).
func dotDD(a, b []float64) float64 {
	if y, ok := dotDev(a, b, 0, 0); ok {
		return y
	}
	return dotDDExact(a, b)
}

// dotDDExact is Σ a[i]·b[i] summed in double-double and rounded once.
func dotDDExact(a, b []float64) float64 {
	var s dd.DD
	for i := range a {
		s = dd.Add(s, dd.MulFF(a[i], b[i]))
	}
	return s.Float()
}

// devDot is dotDD(deviations(a) with mean ma, deviations(b) with mean mb)
// without forming the deviations unless the exact path is needed.
func devDot(a, b []float64, ma, mb float64) float64 {
	if y, ok := dotDev(a, b, ma, mb); ok {
		return y
	}
	return dotDDExact(subMean(a, ma), subMean(b, mb))
}

// subMean returns a - m (each difference rounded).
func subMean(a []float64, m float64) []float64 {
	d := make([]float64, len(a))
	for i, v := range a {
		d[i] = v - m
	}
	return d
}

// sumFast returns sumDDExact(a) and true when its compensated sum (Sum2 on
// two accumulators, with a running bound on its error) can vouch for the
// rounding of the exact sum, and for the double-double sum's (AddF,
// relative error 2u² per step) rounding there too; as dotDev.
func sumFast(a []float64) (float64, bool) {
	n := len(a)
	var s0, c0, s1, c1, ea, pa float64
	i := 0
	for ; i+1 < n; i += 2 {
		p0, p1 := a[i], a[i+1]
		t0 := s0 + p0
		t1 := s1 + p1
		v0 := t0 - s0
		v1 := t1 - s1
		f0 := (s0 - (t0 - v0)) + (p0 - v0)
		f1 := (s1 - (t1 - v1)) + (p1 - v1)
		s0, s1 = t0, t1
		c0 += f0
		c1 += f1
		ea += math.Abs(f0) + math.Abs(f1)
		pa += math.Abs(p0) + math.Abs(p1)
	}
	if i < n {
		p0 := a[i]
		t0 := s0 + p0
		v0 := t0 - s0
		f0 := (s0 - (t0 - v0)) + (p0 - v0)
		s0 = t0
		c0 += f0
		ea += math.Abs(f0)
		pa += math.Abs(p0)
	}
	fn := float64(n)
	if !(pa < 0x1p1000 && pa > 0x1p-900) || fn*u53 > 0x1p-20 {
		return 0, false
	}
	h, l := dd.TwoSum(s0, s1)
	l2 := l + c0 + c1
	errFast := 4.4*fn*u53*ea + 2.2*u53*(math.Abs(l)+math.Abs(c0)+math.Abs(c1))
	errDD := 2.2 * fn * u53 * u53 * pa
	y, rem := dd.TwoSum(h, l2)
	return y, decidedAbs(y, rem, errFast+errDD)
}
