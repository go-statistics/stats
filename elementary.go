package stats

import (
	"math"

	"github.com/go-numerics/optimize"
	"github.com/go-statistics/stats/internal/dd"
)

// Elementary functions in double-double for the closed-form distributions:
// each returns its value to about 2^-70 relative, so that the one final
// rounding of a distribution function built on them is (almost always)
// the correct one.

// expD returns exp(a) as a double-double: +Inf above the float64 range,
// with a low part that may lose bits only where the result is subnormal.
func expD(a dd.DD) dd.DD {
	m, k := dd.ExpMant(a)
	if k > 1024 {
		return dd.DD{Hi: math.Inf(1)}
	}
	r := dd.Ldexp(m, k)
	if math.IsInf(r.Hi, 0) {
		r.Lo = 0
	}
	return r
}

// expm1D returns exp(a) - 1 as a double-double, relative accuracy about
// 2^-70 for every a: the Taylor series below 2^-6 (where exp(a) - 1 would
// cancel), the exponential's mantissa minus one above.
func expm1D(a dd.DD) dd.DD {
	x := a.Hi
	if math.Abs(x) < 1.0/64 {
		// a + a²/2 + a³/6 in double-double, the rest (below 2^-22 of a) in
		// float64.
		sq := dd.Mul(a, a)
		cube := dd.Mul(sq, a)
		r := dd.Add(a, dd.Add(dd.MulF(sq, 0.5), dd.DivF(cube, 6)))
		tail := x * x * x * x * (1.0/24 + x*(1.0/120+x*(1.0/720+x*(1.0/5040))))
		return dd.AddF(r, tail)
	}
	e := expD(a)
	if math.IsInf(e.Hi, 1) {
		return e
	}
	return dd.AddF(e, -1)
}

// log1mexpD returns log(1 - exp(-t)) for t > 0 as a double-double:
// log1p(-exp(-t)) above ln 2, log(-expm1(-t)) below, so that neither form
// cancels.
func log1mexpD(t dd.DD) dd.DD {
	if t.Hi > ln2DD.Hi {
		return log1pDD(dd.Neg(expD(dd.Neg(t))))
	}
	return logDD(dd.Neg(expm1D(dd.Neg(t))))
}

// logRatio returns log(x/y) for x, y > 0 as a double-double: log1p of the
// exact difference when x/y is within a factor 2 of 1 (where the
// difference of the logarithms would cancel), else the difference.
func logRatio(x, y float64) dd.DD {
	if x >= 0.5*y && x <= 2*y {
		return log1pDD(dd.DivF(dd.F(x-y), y)) // x - y exact (Sterbenz)
	}
	return dd.Sub(dd.Log(x), dd.Log(y))
}

// sincosSmall returns sin(y) and cos(y) for a double-double 0 <= y <= π/4
// (and slightly beyond): y = j/128 + r with |r| <= 1/256, the table values
// at j/128 combined with the Taylor series of r, whose r³ and r⁴ terms
// onward (below 2^-18 of the leading term) are in float64.
func sincosSmall(y dd.DD) (s, c dd.DD) {
	t := y.Hi*128 + 0.5
	if !(t >= 0 && t < float64(len(sinTable))) {
		// outside the table (only an unreduced or NaN argument): NaN
		// rather than an index out of range.
		return dd.F(math.NaN()), dd.F(math.NaN())
	}
	j := int(t)
	r := dd.AddF(y, -float64(j)/128) // the leading difference is exact
	x := r.Hi
	x2 := x * x
	sr := dd.AddF(r, -x*x2*(1.0/6-x2*(1.0/120-x2*(1.0/5040))))
	cr := dd.AddF(dd.AddF(dd.MulF(dd.Mul(r, r), -0.5), 1), x2*x2*(1.0/24-x2*(1.0/720)))
	st, ct := sinTable[j], cosTable[j]
	s = dd.Add(dd.Mul(st, cr), dd.Mul(ct, sr))
	c = dd.Sub(dd.Mul(ct, cr), dd.Mul(st, sr))
	return s, c
}

// sincosD returns sin(y) and cos(y) for a double-double 0 <= y <= π/2,
// reflecting the upper half onto [0, π/4] (cos(π/2 - y) = sin y).
func sincosD(y dd.DD) (s, c dd.DD) {
	if y.Hi <= quarterPiDD.Hi {
		return sincosSmall(y)
	}
	c, s = sincosSmall(dd.Sub(halfPiDD, y))
	return s, c
}

// atanD returns atan(u) for a double-double 0 <= u <= 1: u = c + ...,
// c = j/128 the nearest grid point, atan u = atan c + atan d with
// d = (u - c)/(1 + u c), |d| <= 1/256, whose series from d³ on is in
// float64.
func atanD(u dd.DD) dd.DD {
	tj := u.Hi*128 + 0.5
	if !(tj >= 0 && tj < float64(len(atanTable))) {
		return dd.F(math.NaN()) // outside [0, 1]: NaN, not an index out of range
	}
	j := int(tj)
	c := float64(j) / 128
	d := dd.Div(dd.AddF(u, -c), dd.AddF(dd.MulF(u, c), 1))
	x := d.Hi
	x2 := x * x
	t := dd.AddF(d, -x*x2*(1.0/3-x2*(1.0/5-x2*(1.0/7))))
	return dd.Add(atanTable[j], t)
}

// sqrtD returns the square root of a double-double (dd.Sqrt, 0 for 0).
func sqrtD(a dd.DD) dd.DD {
	if a.Hi == 0 {
		return dd.DD{}
	}
	return dd.Sqrt(a)
}

// logD returns log(x) for a double-double x > 0 (logDD), -Inf for 0.
func logD(a dd.DD) dd.DD {
	if a.Hi == 0 {
		return dd.DD{Hi: math.Inf(-1)}
	}
	return logDD(a)
}

// logPair returns log p for a probability p whose complement is q, both
// double-doubles: log1p(-q) when p > 1/2, where log p would cancel.
func logPair(p, q dd.DD) float64 {
	if p.Hi > 0.5 {
		return log1pDD(dd.Neg(q)).Float()
	}
	return logD(p).Float()
}

// brentRoot finds a root of f in the bracket [a, b] (f(a), f(b) of
// opposite signs) with Brent's method, to full float64 precision.
func brentRoot(f func(float64) float64, a, b float64) (float64, bool) {
	r, err := optimize.Brentq(f, a, b, &optimize.BracketOptions{Xtol: 1e-300, Rtol: 4 * 0x1p-52, MaxIter: 500})
	if err != nil || !r.Converged {
		return math.NaN(), false
	}
	return r.Root, true
}

// ldexpRound returns (m.Hi + m.Lo)·2^k rounded once, also where the
// result is subnormal (math.Ldexp(m.Hi+m.Lo, k) rounds twice there: to 53
// bits, then to the subnormal grid).
func ldexpRound(m dd.DD, k int) float64 {
	r := math.Ldexp(m.Hi+m.Lo, k)
	if math.Abs(r) >= 0x1p-1022 || r != r {
		return r
	}
	// In units of the smallest subnormal, hs is exact and the low part
	// (at most half an ulp of hs) can only break a tie of hs's rounding.
	hs := math.Ldexp(m.Hi, k+1074)
	ls := math.Ldexp(m.Lo, k+1074)
	n := math.RoundToEven(hs)
	switch d := hs - n; {
	case d == 0.5 && ls > 0:
		n++
	case d == -0.5 && ls < 0:
		n--
	}
	return math.Ldexp(n, -1074)
}

// expR returns exp(a) rounded once (dd.Exp, correct also when subnormal).
func expR(a dd.DD) float64 {
	m, k := dd.ExpMant(a)
	return ldexpRound(m, k)
}

// expScaledR returns c·exp(a) rounded once (expScaled, correct also when
// subnormal).
func expScaledR(a, c dd.DD) float64 {
	m, k := dd.ExpMant(a)
	return ldexpRound(dd.Mul(m, c), k)
}

// log1mexpSmall returns log(1 - exp(-t)) for a double-double t > 0
// rounded: log t - t/2 below t = 2^-26 (the next term, t²/24, is below
// the rounding), which stays exact where t is subnormal; log1mexpD above.
func log1mexpSmall(t dd.DD) float64 {
	if t.Hi < 0x1p-26 {
		return dd.AddF(logD(t), -0.5*t.Hi).Float()
	}
	return log1mexpD(t).Float()
}
