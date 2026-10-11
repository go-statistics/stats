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
func crLog(x float64) float64 {
	if !(x > 0) || math.IsInf(x, 1) {
		return math.Log(x)
	}
	return dd.LogAccurate(x).Float()
}

// crLog2 returns log2(x) correctly rounded (barring a near-tie).
func crLog2(x float64) float64 {
	if !(x > 0) || math.IsInf(x, 1) {
		return math.Log2(x)
	}
	if f, e := math.Frexp(x); f == 0.5 {
		return float64(e - 1)
	}
	return dd.Div(dd.LogAccurate(x), dd.DD{Hi: dd.Ln2Hi, Lo: dd.Ln2Lo}).Float()
}

// crExp returns exp(x) correctly rounded (barring a near-tie).
func crExp(x float64) float64 { return dd.Exp(dd.F(x)) }

// crPow returns x**y correctly rounded (barring a near-tie), with the
// special cases of math.Pow (which agree with C's pow).
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
	if x < 0 {
		if y != math.Trunc(y) {
			return math.NaN()
		}
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
	for i, t := range a {
		if t == t {
			a[i], a[j] = a[j], a[i]
			j++
		}
	}
	return j
}

// selectKs rearranges a (free of NaN) so that a[k] is the k-th smallest for
// every k of ks (sorted, distinct, within a), with smaller elements before it
// and larger after. It is introselect: quickselect with a median-of-three
// pivot and a three-way partition, falling back to a sort when the
// recursion goes too deep or the indices are dense.
func selectKs(a []float64, ks []int) {
	multiSelect(a, 0, len(a), ks, 2*bits.Len(uint(len(a))))
}

func multiSelect(a []float64, lo, hi int, ks []int, depth int) {
	for len(ks) > 0 {
		if hi-lo <= 16 || len(ks) > (hi-lo)/8 || depth == 0 {
			slices.Sort(a[lo:hi])
			return
		}
		depth--
		lt, gt := partition3(a, lo, hi)
		// a[lo:lt] < pivot, a[lt:gt] == pivot, a[gt:hi] > pivot
		i := 0
		for i < len(ks) && ks[i] < lt {
			i++
		}
		j := i
		for j < len(ks) && ks[j] < gt {
			j++
		}
		left, right := ks[:i], ks[j:]
		// recurse on the smaller side, loop on the larger
		if lt-lo < hi-gt {
			multiSelect(a, lo, lt, left, depth)
			lo, ks = gt, right
		} else {
			multiSelect(a, gt, hi, right, depth)
			hi, ks = lt, left
		}
	}
}

// partition3 partitions a[lo:hi] around a median-of-three pivot.
func partition3(a []float64, lo, hi int) (lt, gt int) {
	m := lo + (hi-lo)/2
	x, y, z := a[lo], a[m], a[hi-1]
	p := y
	if (x < y) != (x < z) {
		p = x
	} else if (z < x) != (z < y) {
		p = z
	}
	lt, i, gt := lo, lo, hi
	for i < gt {
		switch v := a[i]; {
		case v < p:
			a[lt], a[i] = a[i], a[lt]
			lt++
			i++
		case v > p:
			gt--
			a[gt], a[i] = a[i], a[gt]
		default:
			i++
		}
	}
	return lt, gt
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
