package stats

import (
	"math"

	"github.com/go-statistics/stats/internal/dd"
)

// The probability masses of the binomial family (Binomial, Bernoulli,
// NegativeBinomial, Hypergeometric) and of Poisson, after Catherine Loader,
// "Fast and accurate computation of binomial probabilities" (2000), the
// method of R's dbinom and dpois:
//
//	C(n,k) p^k q^(n-k) = sqrt(n / (2π k (n-k))) ·
//	    exp(δ(n) - δ(k) - δ(n-k) - D(k, np) - D(n-k, nq))
//
// with δ the error of Stirling's formula for log Γ(x+1) (stirlerr) and
// D(x, M) = x log(x/M) + M - x >= 0 the deviance term (bd0). Nothing large
// cancels: each term is small near the mode however large n is. Here the
// exponent is carried in double-double, np and nq are formed exactly, and
// D is evaluated as M·g((x-M)/M), g(u) = (1+u) log(1+u) - u, from the
// accurate u - log(1+u) of log1pmxNeg, so that the mass keeps its relative
// accuracy also at n = 1e15 and in the far tails.

// stirlerrMin is where stirlingErrDD switches from the shifted form to the
// asymptotic series.
const stirlerrMin = 10

// stirlingErrDD returns δ(x) = log Γ(x+1) - (x+1/2) log x + x - log(2π)/2 for
// x > 0, as a double-double.
//
// For x >= 10 the Stirling series to the x^-19 term (truncation below
// 2^-66 of the value). Below, the recurrence log Γ(x+1) = log Γ(x+m+1) -
// Σ log(x+j) moves the argument to X = x+m >= 10:
//
//	δ(x) = δ(X) + (X+1/2) log X - (x+1/2) log x - m - log Π(x+j),
//
// all in double-double, so that the cancellation between the terms (of size
// about 25) costs nothing.
func stirlingErrDD(x float64) dd.DD {
	if x >= stirlerrMin {
		return dd.F(stirlingSeries(x))
	}
	m := math.Ceil(stirlerrMin - x)
	prod := dd.F(1)
	for j := 1.0; j <= m; j++ {
		h, l := dd.TwoSum(x, j)
		prod = dd.Mul(prod, dd.DD{Hi: h, Lo: l})
	}
	xh, xl := dd.TwoSum(x, m) // X = x + m
	big := dd.DD{Hi: xh, Lo: xl}
	r := dd.Mul(dd.AddF(big, 0.5), logDD(big))
	r = dd.Sub(r, dd.Mul(dd.AddF(dd.F(x), 0.5), dd.Log(x)))
	r = dd.AddF(r, -m)
	r = dd.Sub(r, logDD(prod))
	return dd.AddF(r, stirlingSeries(xh))
}

// stirlingSeries is the asymptotic series of δ(x), x >= 10:
// Σ B_2k / (2k (2k-1) x^(2k-1)), k = 1..10.
func stirlingSeries(x float64) float64 {
	r := 1 / x
	r2 := r * r
	return r * (1.0/12 - r2*(1.0/360-r2*(1.0/1260-r2*(1.0/1680-r2*(1.0/1188-r2*(691.0/360360-r2*(1.0/156-r2*(3617.0/122400-r2*(43867.0/244188-r2*(174611.0/125400))))))))))
}

// devianceTerm returns D(x, M) = x log(x/M) + M - x for x > 0 and M > 0 (a
// double-double, so that the caller's np need not be rounded), as
// M·g(u) with u = (x-M)/M and g(u) = (1+u) log(1+u) - u = u² - (1+u)·h(u),
// h(u) = u - log(1+u) from log1pmx (accurate also for small u); for
// x > 2M and x < M/2 directly.
func devianceTerm(x float64, m dd.DD) dd.DD {
	u := dd.Div(dd.AddF(dd.Neg(m), x), m)
	if u.Hi > 1 || u.Hi < -0.5 {
		// x > 2M or x < M/2: the direct form, without cancellation there
		// (and without the overflow of u² for x/M beyond 1e154, or the
		// rounding of 1+u to 0 for x/M below 1e-16)
		t := dd.Mul(dd.F(x), dd.Sub(dd.Log(x), logDD(m)))
		return dd.AddF(dd.Add(t, m), -x)
	}
	h := log1pmx(u)
	g := dd.Sub(dd.Mul(u, u), dd.Mul(dd.AddF(u, 1), h))
	return dd.Mul(m, g)
}

// log1pmx returns h(u) = u - log(1+u) >= 0 for a double-double u > -1 to
// about 2^-67 relative: for |u| < 1/64 the series u²/2 - u³/3 + u⁴/4 - ...
// with its first two terms in double-double (log1pmxNeg keeps only the
// first, which limits it to 2^-53·|u|, 2^-59 at u = 1/64: six ulps of a
// Poisson mass at 28 standard deviations), above as u - log1p(u).
func log1pmx(u dd.DD) dd.DD {
	if math.Abs(u.Hi) >= 1.0/64 {
		return dd.Sub(u, log1pDD(u))
	}
	x := u.Hi
	sq := dd.Mul(u, u)
	h := dd.Sub(dd.MulF(sq, 0.5), dd.DivF(dd.Mul(sq, u), 3))
	x2 := x * x
	tail := x2 * x2 * (1.0/4 + x*(-1.0/5+x*(1.0/6+x*(-1.0/7+x*(1.0/8+x*(-1.0/9+x*(1.0/10+x*(-1.0/11+x*(1.0/12+x*(-1.0/13))))))))))
	return dd.AddF(h, tail)
}

// binomTerm returns E and c with C(n,k) p^k q^(n-k) = c·exp(E), for real
// n = k + nk (nk = n - k given separately, exactly, as is n as a
// double-double) and p + q = 1 given as double-doubles. k = 0 and nk = 0
// are the closed forms q^n and p^n; otherwise Loader's form.
func binomTerm(k, nk float64, n, p, q dd.DD) (e, c dd.DD) {
	switch {
	case k == 0:
		return dd.Mul(n, logDD(q)), dd.F(1)
	case nk == 0:
		return dd.Mul(n, logDD(p)), dd.F(1)
	}
	e = dd.Sub(stirlingErrDD(n.Hi), stirlingErrDD(k))
	e = dd.Sub(e, stirlingErrDD(nk))
	e = dd.Sub(e, devianceTerm(k, dd.Mul(n, p)))
	e = dd.Sub(e, devianceTerm(nk, dd.Mul(n, q)))
	c = dd.Sqrt(dd.Div(dd.Mul(n, inv2Pi), dd.MulFF(k, nk)))
	return e, c
}

// poissonTerm returns E and c with μ^k e^-μ / k! = c·exp(E) (Loader's
// dpois_raw: exp(-δ(k) - D(k, μ)) / sqrt(2πk)), for μ > 0.
func poissonTerm(k, mu float64) (e, c dd.DD) {
	if k == 0 {
		return dd.F(-mu), dd.F(1)
	}
	e = dd.Neg(dd.Add(stirlingErrDD(k), devianceTerm(k, dd.F(mu))))
	c = dd.Sqrt(dd.DivF(inv2Pi, k))
	return e, c
}

// massOf rounds c·exp(e) once; logMassOf returns e + log c rounded once
// (-Inf for a zero mass, e = -Inf).
func massOf(e, c dd.DD) float64 { return expScaled(e, c) }
func logMassOf(e, c dd.DD) float64 {
	if math.IsInf(e.Hi, -1) {
		return e.Hi
	}
	return dd.Add(e, logDD(c)).Float()
}

// expm1DD returns exp(a) - 1 as a double-double: from the exponential's
// double-double mantissa where |a| >= 2^-20 (the subtraction then loses at
// most 20 of its 76 bits), from the Taylor series below.
func expm1DD(a dd.DD) dd.DD {
	if math.Abs(a.Hi) < 0x1p-20 {
		x := a.Hi
		t := dd.Add(a, dd.MulF(dd.Mul(a, a), 0.5))
		return dd.AddF(t, x*x*x*(1.0/6+x*(1.0/24)))
	}
	m, k := dd.ExpMant(a)
	return dd.AddF(dd.Ldexp(m, k), -1)
}

// expDD2 returns exp(a) as a double-double (for a above the underflow
// threshold).
func expDD2(a dd.DD) dd.DD {
	m, k := dd.ExpMant(a)
	return dd.Ldexp(m, k)
}

// log1mexpDD returns log(1 - exp(a)) for a < 0, rounded once: through
// log(-expm1(a)) above -log 2 and log1p(-exp(a)) below, both in
// double-double.
func log1mexpDD(a dd.DD) float64 {
	if a.Hi > -math.Ln2 {
		return logDD(dd.Neg(expm1DD(a))).Float()
	}
	return log1pDD(dd.Neg(expDD2(a))).Float()
}

// ratioSum returns S = Σ_{i>=0} t_i with t_0 = 1 and t_(i+1) = t_i·r(j_i),
// j_0 = j, j_(i+1) = j_i + step, stopping after the term at end or when a
// term falls below 2^-64 of the sum (the sequences summed here are
// log-concave, so the rest is smaller still). It is how a tail of a
// distribution is summed from its first mass, r the ratio of successive
// masses: the sum stays in range when the masses underflow. With max > 0 it
// gives up after max terms and reports false.
func ratioSum(j, step, end float64, max int, r func(j float64) dd.DD) (dd.DD, bool) {
	s, t := dd.F(1), dd.F(1)
	for i := 0; j != end; i++ {
		// out of budget, or j no longer moving (|j| >= 2^53: the walk
		// from k = MaxFloat64 spun through the whole budget, 4·10^6
		// terms), or a term that is not a number: not converged
		if max > 0 && i >= max || j+step == j || t.Hi != t.Hi {
			return s, false
		}
		t = dd.Mul(t, r(j))
		s = dd.Add(s, t)
		j += step
		if t.Hi < s.Hi*0x1p-64 {
			break
		}
	}
	return s, true
}

// normalGuess is the Cornish-Fisher approximation mean + sd·(z +
// skew·(z²-1)/6) of the quantile at the standard normal quantile z, the
// starting point of the discrete quantile searches.
func normalGuess(mean, sd, skew, z float64) float64 {
	return mean + sd*(z+skew*(z*z-1)/6)
}

// logCDFFrom returns log P for a tail probability P whose complement is Q
// (both computed directly), and, when P is below the normal range (where
// its logarithm would lose digits or be -Inf), log of the tail summed from
// its first mass: logm + log S, S from tail(). tail reports false when the
// sum did not converge in its budget, and log P is returned.
func logCDFFrom(p, q float64, tail func() (logm float64, s dd.DD, ok bool)) float64 {
	if p > 0.5 {
		return math.Log1p(-q)
	}
	if p >= 0x1p-1000 {
		return math.Log(p)
	}
	if lm, s, ok := tail(); ok {
		if math.IsInf(lm, -1) { // a degenerate distribution: no mass there
			return lm
		}
		return dd.AddF(logDD(s), lm).Float()
	}
	return math.Log(p)
}

// tailBudget bounds the number of terms of a tail sum computed for a
// logarithm below the normal range.
const tailBudget = 1 << 22

// entropyTerm is -p log p for a mass p = c·exp(e) (0 for a zero mass, and
// for a mass the term function cannot form, NaN: the sums stop there).
func entropyTerm(e, c dd.DD) (p float64, t dd.DD) {
	p = expScaled(e, c)
	if !(p > 0) {
		return 0, dd.DD{}
	}
	return p, dd.MulF(dd.Add(e, logDD(c)), -p)
}

// entropyMaxTerms bounds each direction of entropySum: the callers sum at
// most some 10^7 terms (they take entropyLargeVar or entropyIntegral for
// wider distributions), and the bound guarantees that a sum ends whatever
// its terms do.
const entropyMaxTerms = 1 << 26

// entropySum returns -Σ p_k log p_k over the integers of [lo, hi] (hi may
// be +Inf), from the mode outwards in both directions, each direction
// stopping where the masses have fallen below 1/e and their terms below
// 2^-60 of the sum (the masses summed here decrease away from the mode),
// at a zero mass, or after entropyMaxTerms terms. term(k) returns the mass
// at k as c·exp(e).
func entropySum(mode, lo, hi float64, term func(k float64) (e, c dd.DD)) float64 {
	s := dd.F(0)
	n := 0
	for k := mode; k <= hi && n < entropyMaxTerms; k, n = k+1, n+1 {
		p, t := entropyTerm(term(k))
		s = dd.Add(s, t)
		if p < 1/math.E && (p == 0 || t.Hi <= s.Hi*0x1p-60) {
			break
		}
	}
	n = 0
	for k := mode - 1; k >= lo && n < entropyMaxTerms; k, n = k-1, n+1 {
		p, t := entropyTerm(term(k))
		s = dd.Add(s, t)
		if p < 1/math.E && (p == 0 || t.Hi <= s.Hi*0x1p-60) {
			break
		}
	}
	return s.Float()
}

// entropyIntegralMinSD is the standard deviation from which entropyIntegral
// replaces entropySum for the distributions whose Edgeworth expansion does
// not apply (skewed, very wide): the sum would take more than 10^6 terms.
const entropyIntegralMinSD = 2e4

// entropyIntegral returns -Σ_{k >= lo} p_k log p_k for a mass function
// whose term(k) continues smoothly to real k (Loader's saddle-point form,
// as nbinomStd.term does), with mode m and standard deviation sd >=
// entropyIntegralMinSD: the first 1024 terms summed, the rest by the
// Euler-Maclaurin formula, ∫_K^∞ g + g(K)/2 - g'(K)/12 with g(k) =
// -p(k) log p(k) and K = lo + 1024 (the next correction, g'''(K)/720, is
// below 10^-11 g(K) there), the integral by tanh-sinh quadrature over
// pieces of width sd/2 from the mode outwards in both directions, each
// direction ending where a piece falls below 2^-60 of the sum (or, below
// the mode, at K, where the sum and the corrections take over). Its
// relative error is about 1e-15 (the quadrature's), not the last-bit
// accuracy of entropySum; SciPy, which sums 1000 terms, is far off there.
func entropyIntegral(lo, mode, sd float64, term func(k float64) (e, c dd.DD)) float64 {
	k0 := lo + 1024
	g := func(k float64) float64 { _, t := entropyTerm(term(k)); return t.Float() }
	w := sd / 2
	s := dd.F(0)
	a := max(mode, k0)
	for i := 0; i < 400; i++ {
		v := tanhSinh(g, a, w, 1)
		s = dd.AddF(s, v)
		a += w
		if !(v > s.Hi*0x1p-60) {
			break
		}
	}
	b := max(mode, k0)
	for i := 0; i < 400 && b > k0; i++ {
		l := max(k0, b-w)
		v := tanhSinh(g, l, b-l, 1)
		s = dd.AddF(s, v)
		b = l
		if !(v > s.Hi*0x1p-60) {
			break
		}
	}
	if b <= k0 {
		s = dd.AddF(s, entropySum(lo, lo, k0-1, term)+g(k0)/2-(g(k0+1)-g(k0-1))/24)
	}
	return s.Float()
}

// entropyLargeVar is the entropy of a lattice distribution with variance v
// (> 1e6) and skewness and excess kurtosis g1, g2, from the Edgeworth
// expansion of its negentropy: log(2πe v)/2 - (g1²/12 + g2²/48 +
// 7g1⁴/48 - g1²g2/8). For the distributions where it is used the omitted
// terms are of order v^-3, below 2^-60 relative.
func entropyLargeVar(v, g1, g2 float64) float64 {
	j := g1*g1/12 + g2*g2/48 + 7*g1*g1*g1*g1/48 - g1*g1*g2/8
	return 0.5*math.Log(2*math.Pi*math.E*v) - j
}

// entropyEdgeworthMin is the variance above which the entropy of a
// near-normal distribution comes from entropyLargeVar rather than a sum.
const entropyEdgeworthMin = 1e6

// stirling2 returns the Stirling numbers of the second kind S(n, j),
// j = 0..n, which turn factorial moments into raw moments:
// E[X^n] = Σ_j S(n, j) E[X(X-1)...(X-j+1)].
func stirling2(n int) []float64 {
	row := make([]float64, n+1)
	row[0] = 1
	for i := 1; i <= n; i++ {
		for j := i; j >= 1; j-- {
			row[j] = float64(j)*row[j] + row[j-1]
		}
		row[0] = 0
	}
	return row
}

// rawFromFactorial returns E[X^n] = Σ_j S(n,j)·f(j) from the factorial
// moments f(j) = E[X(X-1)...(X-j+1)], n >= 1.
func rawFromFactorial(n int, f func(j int) float64) float64 {
	s := stirling2(n)
	r := 0.0
	for j := 1; j <= n; j++ {
		r += s[j] * f(j)
	}
	return r
}

// fallingPow returns x(x-1)...(x-j+1).
func fallingPow(x float64, j int) float64 {
	r := 1.0
	for i := 0; i < j; i++ {
		r *= x - float64(i)
	}
	return r
}
