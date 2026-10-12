package stats

import (
	"math"
	"math/bits"
	"slices"

	"github.com/go-statistics/stats/internal/dd"
)

// Helpers shared by the descriptive statistics: correctly rounded powers and
// logarithms (where NumPy calls the C library, whose results are within an
// ulp; ours are the correctly rounded ones), NaN handling and selection.

// crLog returns log(x) correctly rounded (barring a near-tie).
//
// Ziv's strategy: dd.Log (relative error below 2^-72) decides the rounding
// unless the value lies within 2^-66 relative of a midpoint between two
// floats (about one argument in 2^13), and only then is dd.LogAccurate
// (2^-104) evaluated. Where the fast path decides, the rounding of every
// value within its error bound is the same, so the result is the one
// LogAccurate gives: the function is unchanged, bit for bit.
func crLog(x float64) float64 {
	if !(x > 0) || math.IsInf(x, 1) {
		return math.Log(x)
	}
	if y, ok := roundDecided(dd.Log(x), 0x1p-66); ok {
		return y
	}
	return dd.LogAccurate(x).Float()
}

// roundDecided returns the float64 nearest r = r.Hi + r.Lo, and whether that
// rounding holds for every value within rel·|r| of r: the remainder r - y
// (exact) must stay that far from half an ulp of y. y must be a normal
// float64 (or zero, where r is exact).
func roundDecided(r dd.DD, rel float64) (float64, bool) {
	y, rem := dd.FastTwoSum(r.Hi, r.Lo)
	// 2^e <= |y| < 2^(e+1): half an ulp is 2^(e-53), or 2^(e-54) below
	// |y| = 2^e; rel·|r| < rel·2^(e+1)
	b := math.Float64bits(y)
	p := math.Float64frombits(b & (0x7ff << 52))
	half := p * 0x1p-53
	if b&(1<<52-1) == 0 && (rem < 0) != (y < 0) {
		half = p * 0x1p-54
	}
	return y, math.Abs(rem) < half-p*2*rel
}

// crLog2 returns log2(x) correctly rounded (barring a near-tie), for
// finite x > 0.
func crLog2(x float64) float64 {
	if f, e := math.Frexp(x); f == 0.5 {
		return float64(e - 1)
	}
	return dd.Div(dd.LogAccurate(x), dd.DD{Hi: dd.Ln2Hi, Lo: dd.Ln2Lo}).Float()
}

// crExp returns exp(x) correctly rounded (barring a near-tie).
func crExp(x float64) float64 { return dd.Exp(dd.F(x)) }

// crPow returns x**y correctly rounded (barring a near-tie), with the
// special cases of math.Pow (which agree with C's pow). A negative x needs
// an integer y.
func crPow(x, y float64) float64 {
	switch {
	case y == 0 || x == 1:
		return 1
	case y == 1:
		return x
	case y == 2:
		return x * x
	case y == 0.5 && x >= 0:
		return math.Sqrt(x)
	case y == -1:
		return 1 / x
	case x == 0 || math.IsInf(x, 0) || math.IsNaN(x) || math.IsNaN(y) || math.IsInf(y, 0):
		return math.Pow(x, y)
	}
	neg := false
	if x < 0 { // y is an integer here (the callers' contract)
		x = -x
		neg = math.Mod(y, 2) != 0
	}
	var r float64
	if y == math.Trunc(y) && math.Abs(y) <= 8 {
		r = powIntDD(x, int(math.Abs(y)))
		if y < 0 {
			r = dd.Div(dd.F(1), powDD(x, int(-y))).Float()
		}
		if !(r < 0x1p1000 && r > 0x1p-960) {
			r = dd.Exp(dd.Mul(dd.LogAccurate(x), dd.F(y)))
		}
	} else {
		r = dd.Exp(dd.Mul(dd.LogAccurate(x), dd.F(y)))
	}
	if neg {
		return -r
	}
	return r
}

// powDD returns x**n (n >= 2) in double-double.
func powDD(x float64, n int) dd.DD {
	r := dd.MulFF(x, x)
	for i := 2; i < n; i++ {
		r = dd.MulF(r, x)
	}
	return r
}

func powIntDD(x float64, n int) float64 { return powDD(x, n).Float() }

// hasNaN reports whether v holds a NaN.
func hasNaN(v []float64) bool {
	for _, t := range v {
		if t != t {
			return true
		}
	}
	return false
}

// dropNaN returns v without its NaNs (v itself when it has none).
func dropNaN(v []float64) []float64 {
	if !hasNaN(v) {
		return v
	}
	out := make([]float64, 0, len(v))
	for _, t := range v {
		if t == t {
			out = append(out, t)
		}
	}
	return out
}

// zeroNaN returns a copy of v with the NaNs replaced by zeros (kept in
// place, as NumPy's nan-functions and SciPy's nan_policy="omit" do, so that
// the pairwise sums group the same elements) and the count of the others.
func zeroNaN(v []float64) ([]float64, int) {
	out := make([]float64, len(v))
	n := 0
	for i, t := range v {
		if t == t {
			out[i] = t
			n++
		}
	}
	return out, n
}

// pyMod is Python's (and NumPy's) float remainder: the sign of the divisor.
func pyMod(a, b float64) float64 {
	m := math.Mod(a, b)
	if m != 0 && (b < 0) != (m < 0) {
		m += b
	}
	return m
}

// nanLast moves the NaNs of a to its end and returns the count of the
// others: NumPy's sort and partition order NaN after every number.
func nanLast(a []float64) int {
	j := 0
	for j < len(a) && a[j] == a[j] {
		j++
	}
	for i := j + 1; i < len(a); i++ {
		if t := a[i]; t == t {
			a[i], a[j] = a[j], t
			j++
		}
	}
	return j
}

// selectKs rearranges a (free of NaN) so that a[k] is the k-th smallest for
// every k of ks (sorted, distinct, within a), with smaller elements before it
// and larger after: the middle index by Floyd and Rivest's selection, then
// the indices on each side within their part.
func selectKs(a []float64, ks []int) {
	multiSelect(a, 0, len(a), ks)
}

func multiSelect(a []float64, lo, hi int, ks []int) {
	for len(ks) > 0 {
		// an index at an end of the range is its minimum or maximum: one
		// scan (the second of the two neighbours a quantile interpolates)
		if ks[0] == lo {
			m := lo
			for i := lo + 1; i < hi; i++ {
				if a[i] < a[m] {
					m = i
				}
			}
			a[lo], a[m] = a[m], a[lo]
			lo, ks = lo+1, ks[1:]
			continue
		}
		if ks[len(ks)-1] == hi-1 {
			m := hi - 1
			for i := lo; i < hi-1; i++ {
				if a[i] > a[m] {
					m = i
				}
			}
			a[hi-1], a[m] = a[m], a[hi-1]
			hi, ks = hi-1, ks[:len(ks)-1]
			continue
		}
		if hi-lo <= 16 || len(ks) > (hi-lo)/8 {
			slices.Sort(a[lo:hi])
			return
		}
		m := len(ks) / 2
		k := ks[m]
		floydRivest(a, lo, hi-1, k, 2*bits.Len(uint(hi-lo))+8)
		left, right := ks[:m], ks[m+1:]
		if k-lo < hi-k {
			multiSelect(a, lo, k, left)
			lo, ks = k+1, right
		} else {
			multiSelect(a, k+1, hi, right)
			hi, ks = k, left
		}
	}
}

// floydRivest places the k-th smallest of a[left..right] at a[k], smaller
// elements before and larger after (Floyd and Rivest, "Algorithm 489: the
// algorithm SELECT", CACM 18(3), 1975): the pivot is chosen by selecting,
// recursively, within a sample bracketing k, so that a partition pass
// leaves a short range around k. After budget passes it sorts the range
// left (a guard against adversarial inputs).
func floydRivest(a []float64, left, right, k, budget int) {
	for right > left {
		if budget == 0 {
			slices.Sort(a[left : right+1])
			return
		}
		budget--
		if right-left > 600 {
			n := float64(right - left + 1)
			i := float64(k - left + 1)
			z := math.Log(n)
			sm := 0.5 * math.Exp(2*z/3)
			sd := 0.5 * math.Sqrt(z*sm*(n-sm)/n)
			if i < n/2 {
				sd = -sd
			}
			nl := max(left, int(float64(k)-i*sm/n+sd))
			nr := min(right, int(float64(k)+(n-i)*sm/n+sd))
			floydRivest(a, nl, nr, k, budget)
		}
		t := a[k]
		i, j := left, right
		a[left], a[k] = a[k], a[left]
		if a[right] > t {
			a[right], a[left] = a[left], a[right]
		}
		for i < j {
			a[i], a[j] = a[j], a[i]
			i++
			j--
			for a[i] < t {
				i++
			}
			for a[j] > t {
				j--
			}
		}
		if a[left] == t {
			a[left], a[j] = a[j], a[left]
		} else {
			j++
			a[j], a[right] = a[right], a[j]
		}
		if j <= k {
			left = j + 1
		}
		if k <= j {
			right = j - 1
		}
	}
}

// orderStats returns a copy of v (NaNs moved last) in which the order
// statistics at the indices ks (sorted, distinct) are in place.
func orderStats(v []float64, ks []int) []float64 {
	a := slices.Clone(v)
	n := nanLast(a)
	end := len(ks)
	for end > 0 && ks[end-1] >= n {
		end--
	}
	selectKs(a[:n], ks[:end])
	return a
}

// uniqueSorted sorts ks and removes duplicates.
func uniqueSorted(ks []int) []int {
	slices.Sort(ks)
	return slices.Compact(ks)
}
