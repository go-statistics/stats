package stats

import "math"

// log1pFD and expm1FD are the FreeBSD msun s_log1p.c and s_expm1.c
// algorithms (Copyright (C) 1993 by Sun Microsystems, Inc.; permission to
// use, copy, modify and distribute is freely granted, provided this notice is
// preserved), as Go's math.Log1p and math.Expm1 port them, restricted to the
// arguments MultipleTests feeds them (log1p on [-1, 0], expm1 on (-inf, 0])
// and with every product that feeds a sum wrapped in float64(...): the Go
// specification lets arm64, ppc64le, s390x, riscv64 and loong64 fuse a*b+c,
// which would change the last bit of the result. The reference (NumPy on
// glibc) computes sidak and holm-sidak with these libm routines, so the
// adjusted p-values agree with it to within one unit in the last place on
// every platform, and exactly on all but a few inputs (glibc's x86-64 FMA
// variants contract the same C source in places Go cannot reproduce).

// log1pFD returns log(1+x) for -1 <= x <= 0 (NaN is not expected).
func log1pFD(x float64) float64 {
	const (
		Small = 1.0 / (1 << 29)
		Tiny  = 1.0 / (1 << 54)
		Ln2Hi = 6.93147180369123816490e-01
		Ln2Lo = 1.90821492927058770002e-10
		Lp1   = 6.666666666666735130e-01
		Lp2   = 3.999999999940941908e-01
		Lp3   = 2.857142874366239149e-01
		Lp4   = 2.222219843214978396e-01
		Lp5   = 1.818357216161805012e-01
		Lp6   = 1.531383769920937332e-01
		Lp7   = 1.479819860511658591e-01
	)
	if x == -1 {
		return math.Inf(-1)
	}
	absx := -x
	if absx < Small {
		if absx < Tiny {
			return x
		}
		return x - float64(float64(x*x)*0.5)
	}
	var f, c float64
	var iu uint64
	k := 0
	if absx < 4.142135623730950488017e-01 && x > -2.928932188134524755992e-01 {
		f = x // k = 0: f = x is exact
		iu = 1
	} else {
		u := 1.0 + x
		iu = math.Float64bits(u)
		k = int((iu >> 52) - 1023)
		c = (x - (u - 1.0)) / u // k <= 0 here
		iu &= 0x000fffffffffffff
		if iu < 0x0006a09e667f3bcd {
			u = math.Float64frombits(iu | 0x3ff0000000000000)
		} else {
			k++
			u = math.Float64frombits(iu | 0x3fe0000000000000)
			iu = (0x0010000000000000 - iu) >> 2
		}
		f = u - 1.0
	}
	hfsq := float64(float64(0.5*f) * f)
	if iu == 0 { // then f == 0 and k != 0
		c += float64(float64(k) * Ln2Lo)
		return float64(float64(k)*Ln2Hi) + c
	}
	s := f / (2.0 + f)
	z := float64(s * s)
	R := float64(z * (Lp1 + float64(z*(Lp2+float64(z*(Lp3+float64(z*(Lp4+float64(z*(Lp5+float64(z*(Lp6+float64(z*Lp7)))))))))))))
	if k == 0 {
		return f - (hfsq - float64(s*(hfsq+R)))
	}
	return float64(float64(k)*Ln2Hi) - ((hfsq - (float64(s*(hfsq+R)) + (float64(float64(k)*Ln2Lo) + c))) - f)
}

// expm1FD returns exp(x)-1 for x <= 0 (-Inf allowed, NaN not expected).
func expm1FD(x float64) float64 {
	const (
		Ln2X56    = 3.88162421113569373274e+01
		Ln2HalfX3 = 1.03972077083991796413e+00
		Ln2Half   = 3.46573590279972654709e-01
		Ln2Hi     = 6.93147180369123816490e-01
		Ln2Lo     = 1.90821492927058770002e-10
		InvLn2    = 1.44269504088896338700e+00
		Tiny      = 1.0 / (1 << 54)
		Q1        = -3.33333333333331316428e-02
		Q2        = 1.58730158725481460165e-03
		Q3        = -7.93650757867487942473e-05
		Q4        = 4.00821782732936239552e-06
		Q5        = -2.01099218183624371326e-07
	)
	absx := -x
	if absx >= Ln2X56 {
		return -1
	}
	var c float64
	k := 0
	if absx > Ln2Half {
		var hi, lo float64
		if absx < Ln2HalfX3 {
			hi = x + Ln2Hi
			lo = -Ln2Lo
			k = -1
		} else {
			k = int(float64(InvLn2*x) - 0.5) // k <= -2
			t := float64(k)
			hi = x - float64(t*Ln2Hi)
			lo = float64(t * Ln2Lo)
		}
		x = hi - lo
		c = (hi - x) - lo
	} else if absx < Tiny {
		return x
	}
	hfx := 0.5 * x
	hxs := float64(x * hfx)
	r1 := 1 + float64(hxs*(Q1+float64(hxs*(Q2+float64(hxs*(Q3+float64(hxs*(Q4+float64(hxs*Q5)))))))))
	t := 3 - float64(r1*hfx)
	e := float64(hxs * ((r1 - t) / (6.0 - float64(x*t))))
	if k == 0 {
		return x - (float64(x*e) - hxs)
	}
	e = float64(x*(e-c)) - c
	e -= hxs
	if k == -1 {
		return float64(0.5*(x-e)) - 0.5
	}
	y := 1 - (e - x)
	y = math.Float64frombits(math.Float64bits(y) + uint64(k)<<52)
	return y - 1
}
