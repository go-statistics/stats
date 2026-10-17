package stats

import (
	"math"

	"github.com/go-statistics/stats/internal/dd"
)

// The hypergeometric distribution behind Fisher's exact test, computed
// with Loader's saddle-point expansion (the method of R's dhyper and
// dbinom_raw) carried out in double-double: products of binomial
// coefficients or differences of log-gamma values lose accuracy in
// proportion to the magnitude of the logarithms, this does not.

// stirlingErrTable is log(n!) - log(sqrt(2πn)·(n/e)^n) for n = 0..15
// (mpmath, 50 digits).
var stirlingErrTable = [16]float64{
	0,
	0.08106146679532725822,
	0.041340695955409294094,
	0.027677925684998339149,
	0.020790672103765093112,
	0.016644691189821192163,
	0.013876128823070747999,
	0.011896709945891770095,
	0.010411265261972096497,
	0.0092554621827127329177,
	0.0083305634333628712565,
	0.007573675487951840795,
	0.0069428401072095298657,
	0.0064089941880042070684,
	0.0059513701127588477356,
	0.005554733551962801371,
}

// stirlingErr is the error of Stirling's formula for log(n!), n a
// non-negative integer.
func stirlingErr(n float64) float64 {
	if n <= 15 {
		return stirlingErrTable[int(n)]
	}
	const (
		s0 = 1.0 / 12
		s1 = 1.0 / 360
		s2 = 1.0 / 1260
		s3 = 1.0 / 1680
		s4 = 1.0 / 1188
		s5 = 691.0 / 360360
	)
	nn := n * n
	switch {
	case n > 500:
		return (s0 - s1/nn) / n
	case n > 80:
		return (s0 - (s1-s2/nn)/nn) / n
	case n > 35:
		return (s0 - (s1-(s2-s3/nn)/nn)/nn) / n
	}
	return (s0 - (s1-(s2-(s3-(s4-s5/nn)/nn)/nn)/nn)/nn) / n
}

// bd0 is the deviance term x·log(x/np) + np - x in double-double, by its
// series where the direct formula cancels (|x - np| < 0.1·(x + np)).
func bd0(x float64, np dd.DD) dd.DD {
	d := dd.Sub(dd.F(x), np)
	s := dd.AddF(np, x)
	if math.Abs(d.Hi) < 0.1*s.Hi {
		v := dd.Div(d, s)
		sum := dd.Mul(d, v)
		ej := dd.MulF(v, 2*x)
		v2 := dd.Mul(v, v)
		for j := 1; j < 1000; j++ {
			ej = dd.Mul(ej, v2)
			t := dd.DivF(ej, float64(2*j+1))
			sum = dd.Add(sum, t)
			if math.Abs(t.Hi) <= 0x1p-110*math.Abs(sum.Hi) {
				break
			}
		}
		return sum
	}
	return dd.Sub(dd.Mul(dd.F(x), dd.Sub(dd.Log(x), logDD(np))), d)
}

// binomExp returns log of the binomial probability C(n,x)·p^x·q^(n-x)
// (R's dbinom_raw, Loader's saddle-point form) in double-double, for
// 0 <= x <= n, 0 < p, q < 1 (q = 1 - p).
func binomExp(x, n float64, p, q dd.DD) dd.DD {
	switch x {
	case 0:
		if p.Hi < 0.1 {
			return dd.Sub(dd.Neg(bd0(n, dd.MulF(q, n))), dd.MulF(p, n))
		}
		return dd.MulF(logDD(q), n)
	case n:
		if q.Hi < 0.1 {
			return dd.Sub(dd.Neg(bd0(n, dd.MulF(p, n))), dd.MulF(q, n))
		}
		return dd.MulF(logDD(p), n)
	}
	lc := dd.F(stirlingErr(n) - stirlingErr(x) - stirlingErr(n-x))
	lc = dd.Sub(lc, dd.Add(bd0(x, dd.MulF(p, n)), bd0(n-x, dd.MulF(q, n))))
	// log(2π·x·(n-x)/n)
	lf := dd.Add(dd.Add(dd.Log(2*math.Pi), dd.Log(x)), dd.Sub(dd.Log(n-x), dd.Log(n)))
	lf = dd.AddF(lf, 2.4492935982947064e-16/(2*math.Pi)) // 2π's rounding error
	return dd.Sub(lc, dd.MulF(lf, 0.5))
}

// hypergeom is the distribution of the number of marked items in a draw
// of nDraw items without replacement from total items of which marked
// are marked: scipy.stats.hypergeom(M=total, n=marked, N=nDraw). The
// probabilities are computed in double-double and rounded once, so that
// Fisher's p-values carry errors of a few ulps whatever the table.
type hypergeom struct {
	total, marked, nDraw float64
}

func (h hypergeom) lo() float64 { return max(0, h.nDraw-(h.total-h.marked)) }
func (h hypergeom) hi() float64 { return min(h.marked, h.nDraw) }

// pmfDD is P(X = k), k an integer (0 < nDraw < total: Fisher's test
// returns before any empty margin reaches here).
func (h hypergeom) pmfDD(k float64) dd.DD {
	if k < h.lo() || k > h.hi() {
		return dd.DD{}
	}
	b := h.total - h.marked
	// Any p works (it cancels in the ratio); q must be 1 - p.
	p := dd.DivF(dd.F(h.nDraw), h.total)
	q := dd.Sub(dd.F(1), p)
	e := dd.Sub(dd.Add(binomExp(k, h.marked, p, q), binomExp(h.nDraw-k, b, p, q)), binomExp(h.nDraw, h.total, p, q))
	m, x := dd.ExpMant(e)
	return dd.Ldexp(m, x)
}

func (h hypergeom) pmf(k float64) float64 { return h.pmfDD(k).Float() }

// ratioDown is pmf(k-1)/pmf(k).
func (h hypergeom) ratioDown(k float64) dd.DD {
	num := dd.MulFF(k, h.total-h.marked-h.nDraw+k)
	return dd.Div(num, dd.MulFF(h.marked-k+1, h.nDraw-k+1))
}

// lowerSum is Σ_{j <= k} pmf(j), summed downward from k.
func (h hypergeom) lowerSum(k float64) float64 {
	lo := h.lo()
	t := h.pmfDD(k)
	s := t
	for j := k; j > lo && t.Hi > 1e-34*s.Hi; j-- {
		t = dd.Mul(t, h.ratioDown(j))
		s = dd.Add(s, t)
	}
	return s.Float()
}

// upperSum is Σ_{j >= k} pmf(j), summed upward from k.
func (h hypergeom) upperSum(k float64) float64 {
	hi := h.hi()
	t := h.pmfDD(k)
	s := t
	for j := k; j < hi && t.Hi > 1e-34*s.Hi; j++ {
		t = dd.Div(t, h.ratioDown(j+1))
		s = dd.Add(s, t)
	}
	return s.Float()
}

// mode is the most probable count.
func (h hypergeom) mode() float64 {
	return math.Floor((h.nDraw + 1) * (h.marked + 1) / (h.total + 2))
}

// cdf is P(X <= k), k an integer: the lower tail summed directly when k
// is below the mode, otherwise one minus the upper tail.
func (h hypergeom) cdf(k float64) float64 {
	switch {
	case k < h.lo():
		return 0
	case k >= h.hi():
		return 1
	case k < h.mode():
		return h.lowerSum(k)
	}
	return 1 - h.upperSum(k+1)
}

// sf is P(X > k), k an integer, summed upward from k+1 (Fisher's test
// asks for it at or above the mode only).
func (h hypergeom) sf(k float64) float64 {
	if k >= h.hi() {
		return 0
	}
	return h.upperSum(k + 1)
}
