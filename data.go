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
		var r [8]float64
		copy(r[:], a[:8])
		i := 8
		for ; i < n-n%8; i += 8 {
			r[0] += a[i]
			r[1] += a[i+1]
			r[2] += a[i+2]
			r[3] += a[i+3]
			r[4] += a[i+4]
			r[5] += a[i+5]
			r[6] += a[i+6]
			r[7] += a[i+7]
		}
		res := ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]))
		for ; i < n; i++ {
			res += a[i]
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

// sqDev returns the squared deviations (x - m)², each product rounded
// separately (no fused multiply-add), as NumPy forms them before summing.
func sqDev(a []float64, m float64) []float64 {
	d := make([]float64, len(a))
	for i, v := range a {
		t := v - m
		d[i] = float64(t * t)
	}
	return d
}

// variance is np.var(a, ddof=ddof) to the bit.
func variance(a []float64, ddof float64) float64 {
	m := mean(a)
	return sum(sqDev(a, m)) / (float64(len(a)) - ddof)
}
