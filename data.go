package stats

import "github.com/go-ndarray/ndarray"

// Data is what the descriptive statistics, the tests and the fits take as
// observations: a []float64, or a go-ndarray array of any shape and dtype,
// read in row-major order and converted to float64. The input is never
// modified.
type Data interface {
	[]float64 | *ndarray.Array
}

// floats returns the observations as a []float64. The result may share
// storage with the input and must be treated as read-only.
func floats[D Data](x D) []float64 {
	if a, ok := any(x).(*ndarray.Array); ok {
		return arrayFloats(a)
	}
	return any(x).([]float64)
}

func arrayFloats(a *ndarray.Array) []float64 {
	if a == nil {
		return nil
	}
	if a.DType() != ndarray.Float64 {
		a = a.AsType(ndarray.Float64)
	}
	v, _ := ndarray.Data[float64](a)
	return v
}

// like returns v in the form of x: a []float64, or an array of x's shape.
func like[D Data](x D, v []float64) D {
	if a, ok := any(x).(*ndarray.Array); ok {
		shape := []int{len(v)}
		if a != nil && a.Size() == len(v) {
			shape = a.Shape()
		}
		r, _ := ndarray.FromData(v, shape...)
		return any(r).(D)
	}
	return any(v).(D)
}

// sum is NumPy's pairwise summation (numpy/_core/src/umath/loops_utils.h),
// reproduced operation for operation so that a mean or a variance computed
// here is the one np.mean and np.var return, to the bit.
func sum(a []float64) float64 {
	n := len(a)
	switch {
	case n < 8:
		r := 0.0
		for _, v := range a {
			r += v
		}
		return r
	case n <= 128:
		r0, r1, r2, r3, r4, r5, r6, r7 := a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]
		m := n - n%8
		for b := a[8:m]; len(b) >= 8; b = b[8:] {
			r0 += b[0]
			r1 += b[1]
			r2 += b[2]
			r3 += b[3]
			r4 += b[4]
			r5 += b[5]
			r6 += b[6]
			r7 += b[7]
		}
		res := ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7))
		for _, v := range a[m:] {
			res += v
		}
		return res
	default:
		n2 := n / 2
		n2 -= n2 % 8
		return sum(a[:n2]) + sum(a[n2:])
	}
}

// mean is np.mean: the pairwise sum divided by the count.
func mean(a []float64) float64 { return sum(a) / float64(len(a)) }

// sumSqDev is sum of the squared deviations (x - m)², each product rounded
// separately (no fused multiply-add), as NumPy forms them before summing:
// the same pairwise summation as sum, without the array of terms.
func sumSqDev(a []float64, m float64) float64 {
	n := len(a)
	switch {
	case n < 8:
		r := 0.0
		for _, v := range a {
			t := v - m
			r += float64(t * t)
		}
		return r
	case n <= 128:
		sq := func(v float64) float64 { t := v - m; return float64(t * t) }
		r0, r1, r2, r3 := sq(a[0]), sq(a[1]), sq(a[2]), sq(a[3])
		r4, r5, r6, r7 := sq(a[4]), sq(a[5]), sq(a[6]), sq(a[7])
		k := n - n%8
		for b := a[8:k]; len(b) >= 8; b = b[8:] {
			r0 += sq(b[0])
			r1 += sq(b[1])
			r2 += sq(b[2])
			r3 += sq(b[3])
			r4 += sq(b[4])
			r5 += sq(b[5])
			r6 += sq(b[6])
			r7 += sq(b[7])
		}
		res := ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7))
		for _, v := range a[k:] {
			res += sq(v)
		}
		return res
	default:
		n2 := n / 2
		n2 -= n2 % 8
		return sumSqDev(a[:n2], m) + sumSqDev(a[n2:], m)
	}
}

// sumTerms is sum of the n terms fill writes, without an array of them:
// the pairwise recursion of sum splits [0, n) exactly as it splits an array
// of length n, and each leaf (at most 128 terms) is filled into a buffer on
// the stack by fill(dst, lo), which sets dst[i] to term lo+i. The result is
// sum's, bit for bit.
func sumTerms(n int, fill func(dst []float64, lo int)) float64 {
	return sumTermsAt(0, n, fill)
}

func sumTermsAt(lo, n int, fill func(dst []float64, lo int)) float64 {
	if n <= 128 {
		var buf [128]float64
		b := buf[:n]
		fill(b, lo)
		return sum(b)
	}
	n2 := n / 2
	n2 -= n2 % 8
	return sumTermsAt(lo, n2, fill) + sumTermsAt(lo+n2, n-n2, fill)
}

// variance is np.var(a, ddof=ddof) to the bit.
func variance(a []float64, ddof float64) float64 {
	m := mean(a)
	return sumSqDev(a, m) / (float64(len(a)) - ddof)
}

// sumProd is sum of the products x[i]·w[i], each rounded (no fused
// multiply-add), without the array of them.
func sumProd(x, w []float64) float64 {
	w = w[:len(x)]
	return sumTerms(len(x), func(d []float64, lo int) {
		xs, ws := x[lo:lo+len(d)], w[lo:lo+len(d)]
		for i := range d {
			d[i] = float64(xs[i] * ws[i])
		}
	})
}
