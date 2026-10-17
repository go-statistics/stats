package stats

import (
	"math"
	"slices"
	"unsafe"
)

// Sorting float64 data by radix instead of by comparison (pdqsort), for
// the statistics that sort: about 5-10x faster from a few thousand values.
//
// The result is the comparison sort's, bit for bit. slices.Sort orders by
// <, so its output is the nondecreasing arrangement of the values, which is
// unique as a sequence of bit patterns when no two values compare equal
// with different bits: when the data hold no NaN and not both -0 and +0.
// The radix sort orders by the bits (an order-preserving map to unsigned
// integers, -0 before +0), so on such data it returns that same sequence;
// on other data the comparison sort is used. The same holds for the stable
// sorts of (value, payload) pairs: an LSD radix sort is stable.

// radixMin is a variable so that the tests can force either path.
var radixMin = 1024

// floatKey maps x to an unsigned integer whose order is that of the
// float64 values (-0 just before +0, NaNs beyond the infinities).
func floatKey(x float64) uint64 {
	b := math.Float64bits(x)
	if b>>63 != 0 {
		return ^b
	}
	return b | 1<<63
}

func keyFloat(k uint64) float64 {
	if k>>63 != 0 {
		return math.Float64frombits(k &^ (1 << 63))
	}
	return math.Float64frombits(^k)
}

const (
	radixBits   = 11
	radixPasses = 6 // 6·11 >= 64
	radixSize   = 1 << radixBits
)

// radixHist reports whether a holds no NaN and not both signed zeros, and
// fills the per-digit histograms of the keys of a.
func radixHist(a []float64, h *[radixPasses][radixSize]int) bool {
	negZero, posZero := false, false
	for _, x := range a {
		if x != x {
			return false
		}
		if x == 0 {
			if math.Signbit(x) {
				negZero = true
			} else {
				posZero = true
			}
		}
		k := floatKey(x)
		for p := range radixPasses {
			h[p][(k>>(p*radixBits))&(radixSize-1)]++
		}
	}
	return !(negZero && posZero)
}

// sortFloats sorts a ascending: slices.Sort(a), bit for bit.
func sortFloats(a []float64) {
	if len(a) < radixMin || !radixSortFloats(a) {
		slices.Sort(a)
	}
}

func radixSortFloats(a []float64) bool {
	var h [radixPasses][radixSize]int
	if !radixHist(a, &h) {
		return false
	}
	n := len(a)
	// the keys replace the values in place (same size), then come back
	keys := unsafe.Slice((*uint64)(unsafe.Pointer(&a[0])), n)
	for i, x := range a {
		keys[i] = floatKey(x)
	}
	tmp := make([]uint64, n)
	src, dst := keys, tmp
	for p := range radixPasses {
		if !scanOffsets(&h[p], n) {
			continue // one digit for all: the pass would not move anything
		}
		off := &h[p]
		shift := p * radixBits
		for _, k := range src {
			d := (k >> shift) & (radixSize - 1)
			dst[off[d]] = k
			off[d]++
		}
		src, dst = dst, src
	}
	if &src[0] != &keys[0] {
		copy(keys, src)
	}
	for i, k := range keys {
		a[i] = keyFloat(k)
	}
	return true
}

// scanOffsets turns a digit histogram into the starting offsets, and
// reports whether the digit varies (false: a single bucket holds all n).
func scanOffsets(c *[radixSize]int, n int) bool {
	s := 0
	for d, v := range c {
		if v == n {
			return false
		}
		c[d] = s
		s += v
	}
	return true
}

// radixSortPairs sorts p stably by key(p[i]) (the float64 each element is
// sorted by), as slices.SortStableFunc with < on those values would, and
// reports false (p untouched) where that would differ: a NaN, or both
// signed zeros, among the values.
func radixSortPairs[T any](p []T, val func(T) float64) bool {
	n := len(p)
	if n < radixMin {
		return false
	}
	var h [radixPasses][radixSize]int
	negZero, posZero := false, false
	for _, e := range p {
		x := val(e)
		if x != x {
			return false
		}
		if x == 0 {
			if math.Signbit(x) {
				negZero = true
			} else {
				posZero = true
			}
		}
		k := floatKey(x)
		for q := range radixPasses {
			h[q][(k>>(q*radixBits))&(radixSize-1)]++
		}
	}
	if negZero && posZero {
		return false
	}
	tmp := make([]T, n)
	src, dst := p, tmp
	for q := range radixPasses {
		if !scanOffsets(&h[q], n) {
			continue
		}
		off := &h[q]
		shift := q * radixBits
		for _, e := range src {
			d := (floatKey(val(e)) >> shift) & (radixSize - 1)
			dst[off[d]] = e
			off[d]++
		}
		src, dst = dst, src
	}
	copy(p, src) // (onto itself after an even number of passes)
	return true
}
