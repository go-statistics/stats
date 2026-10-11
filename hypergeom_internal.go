package stats

import "math"

// The hypergeometric distribution behind Fisher's exact test, computed
// with Loader's saddle-point expansion (the method of R's dhyper and
// dbinom_raw): every probability mass carries a relative error of a few
// ulps whatever the size of the table, where products of binomial
// coefficients or differences of log-gamma values lose accuracy in
// proportion to the magnitude of the logarithms.

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

// bd0 is the deviance term x·log(x/np) + np - x, by a series where the
// direct formula cancels.
func bd0(x, np float64) float64 {
	if math.Abs(x-np) < 0.1*(x+np) {
		v := (x - np) / (x + np)
		s := (x - np) * v
		ej := 2 * x * v
		v *= v
		for j := 1; j < 1000; j++ {
			ej *= v
			s1 := s + ej/float64(2*j+1)
			if s1 == s {
				return s1
			}
			s = s1
		}
	}
	return x*math.Log(x/np) + np - x
}

// binomRaw is the binomial probability C(n,x)·p^x·q^(n-x), q = 1-p given
// separately (R's dbinom_raw), for 0 <= x <= n and 0 < p, q < 1.
func binomRaw(x, n, p, q float64) float64 {
	switch x {
	case 0:
		if n == 0 {
			return 1
		}
		if p < 0.1 {
			return math.Exp(-bd0(n, n*q) - n*p)
		}
		return math.Exp(n * math.Log(q))
	case n:
		if q < 0.1 {
			return math.Exp(-bd0(n, n*p) - n*q)
		}
		return math.Exp(n * math.Log(p))
	}
	lc := stirlingErr(n) - stirlingErr(x) - stirlingErr(n-x) - bd0(x, n*p) - bd0(n-x, n*q)
	lf := math.Log(2*math.Pi) + math.Log(x) + math.Log1p(-x/n)
	return math.Exp(lc - 0.5*lf)
}

// hypergeom is the distribution of the number of marked items in a draw
// of nDraw items without replacement from total items of which marked
// are marked: scipy.stats.hypergeom(M=total, n=marked, N=nDraw).
type hypergeom struct {
	total, marked, nDraw float64
}

func (h hypergeom) lo() float64 { return math.Max(0, h.nDraw-(h.total-h.marked)) }
func (h hypergeom) hi() float64 { return math.Min(h.marked, h.nDraw) }

// pmf is P(X = k), k an integer.
func (h hypergeom) pmf(k float64) float64 {
	if k < h.lo() || k > h.hi() {
		return 0
	}
	if h.nDraw == 0 || h.nDraw == h.total {
		return 1
	}
	b := h.total - h.marked
	p := h.nDraw / h.total
	q := (h.total - h.nDraw) / h.total
	return binomRaw(k, h.marked, p, q) * binomRaw(h.nDraw-k, b, p, q) / binomRaw(h.nDraw, h.total, p, q)
}

// ratioDown is pmf(k-1)/pmf(k).
func (h hypergeom) ratioDown(k float64) float64 {
	return k * (h.total - h.marked - h.nDraw + k) / ((h.marked - k + 1) * (h.nDraw - k + 1))
}

// lowerSum is Σ_{j <= k} pmf(j), summed downward from k.
func (h hypergeom) lowerSum(k float64) float64 {
	lo := h.lo()
	t := h.pmf(k)
	s := t
	for j := k; j > lo && t > 1e-17*s; j-- {
		t *= h.ratioDown(j)
		s += t
	}
	return s
}

// upperSum is Σ_{j >= k} pmf(j), summed upward from k.
func (h hypergeom) upperSum(k float64) float64 {
	hi := h.hi()
	t := h.pmf(k)
	s := t
	for j := k; j < hi && t > 1e-17*s; j++ {
		t /= h.ratioDown(j + 1)
		s += t
	}
	return s
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

// sf is P(X > k), k an integer.
func (h hypergeom) sf(k float64) float64 {
	switch {
	case k < h.lo():
		return 1
	case k >= h.hi():
		return 0
	case k+1 > h.mode():
		return h.upperSum(k + 1)
	}
	return 1 - h.lowerSum(k)
}
