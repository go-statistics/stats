package stats

import (
	"fmt"
	"math"
	"slices"
	"sort"
)

// QuantileMethod is the estimator of a sample quantile, NumPy's method=
// argument of numpy.quantile and numpy.percentile (the nine of Hyndman and
// Fan, "Sample quantiles in statistical packages", The American
// Statistician 50(4), 1996, and four discontinuous ones). The zero value is
// QuantileLinear, NumPy's default.
type QuantileMethod int

const (
	// QuantileLinear is method="linear" (H&F 7): virtual index (n-1)p.
	QuantileLinear QuantileMethod = iota
	// QuantileInvertedCDF is method="inverted_cdf" (H&F 1).
	QuantileInvertedCDF
	// QuantileAveragedInvertedCDF is method="averaged_inverted_cdf" (H&F 2).
	QuantileAveragedInvertedCDF
	// QuantileClosestObservation is method="closest_observation" (H&F 3).
	QuantileClosestObservation
	// QuantileInterpolatedInvertedCDF is method="interpolated_inverted_cdf" (H&F 4).
	QuantileInterpolatedInvertedCDF
	// QuantileHazen is method="hazen" (H&F 5).
	QuantileHazen
	// QuantileWeibull is method="weibull" (H&F 6).
	QuantileWeibull
	// QuantileMedianUnbiased is method="median_unbiased" (H&F 8).
	QuantileMedianUnbiased
	// QuantileNormalUnbiased is method="normal_unbiased" (H&F 9).
	QuantileNormalUnbiased
	// QuantileLower is method="lower": the order statistic floor((n-1)p).
	QuantileLower
	// QuantileHigher is method="higher": the order statistic ceil((n-1)p).
	QuantileHigher
	// QuantileMidpoint is method="midpoint": the mean of lower and higher.
	QuantileMidpoint
	// QuantileNearest is method="nearest": the order statistic nearest to
	// (n-1)p, ties to the even index.
	QuantileNearest
)

var quantileMethodNames = [...]string{
	"linear", "inverted_cdf", "averaged_inverted_cdf", "closest_observation",
	"interpolated_inverted_cdf", "hazen", "weibull", "median_unbiased",
	"normal_unbiased", "lower", "higher", "midpoint", "nearest",
}

// String returns NumPy's name for the method.
func (m QuantileMethod) String() string {
	if m < 0 || int(m) >= len(quantileMethodNames) {
		return "invalid"
	}
	return quantileMethodNames[m]
}

// QuantileOptions are the options of Quantile and Percentile. The zero
// value (or nil) is NumPy's default.
type QuantileOptions struct {
	// Method is the estimator (default QuantileLinear).
	Method QuantileMethod
	// Weights, when not nil, are non-negative weights of the observations
	// (NumPy 2's weights=, supported with QuantileInvertedCDF only).
	Weights []float64
	// OmitNaN ignores the NaNs of the data (numpy.nanquantile); by default a
	// NaN gives a NaN result.
	OmitNaN bool
}

// quantilePlan is where a quantile is read in the sorted sample: the two
// neighbouring order statistics and NumPy's interpolation weight.
type quantilePlan struct {
	prev, next int
	gamma      float64
	discrete   bool
}

// planQuantile reproduces numpy's _quantile index arithmetic (virtual index,
// _get_indexes, _get_gamma and the methods' fix_gamma) operation for
// operation.
func planQuantile(n int, q float64, m QuantileMethod) quantilePlan {
	fn := float64(n)
	nm1 := float64(n - 1)
	discreteBoundary := func(index float64, cond func(gamma, index float64) bool) quantilePlan {
		prev := math.Floor(index)
		gamma := index - prev
		r := int(prev) + 1
		if cond(gamma, index) {
			r = int(prev)
		}
		r = max(r, 0)
		return quantilePlan{prev: r, next: r, discrete: true}
	}
	var v float64
	switch m {
	case QuantileInvertedCDF:
		return discreteBoundary(float64(fn*q)-1, func(g, _ float64) bool { return g == 0 })
	case QuantileClosestObservation:
		return discreteBoundary(float64(fn*q)-1-0.5, func(g, i float64) bool { return g == 0 && pyMod(math.Floor(i), 2) == 1 })
	case QuantileLower:
		k := int(math.Floor(nm1 * q))
		return quantilePlan{prev: k, next: k, discrete: true}
	case QuantileHigher:
		k := int(math.Ceil(nm1 * q))
		return quantilePlan{prev: k, next: k, discrete: true}
	case QuantileNearest:
		k := int(math.RoundToEven(nm1 * q))
		return quantilePlan{prev: k, next: k, discrete: true}
	case QuantileAveragedInvertedCDF:
		v = float64(fn*q) - 1
	case QuantileInterpolatedInvertedCDF:
		v = virtualIndex(fn, q, 0, 1)
	case QuantileHazen:
		v = virtualIndex(fn, q, 0.5, 0.5)
	case QuantileWeibull:
		v = virtualIndex(fn, q, 0, 0)
	case QuantileMedianUnbiased:
		v = virtualIndex(fn, q, 1/3.0, 1/3.0)
	case QuantileNormalUnbiased:
		v = virtualIndex(fn, q, 3/8.0, 3/8.0)
	case QuantileMidpoint:
		t := nm1 * q
		v = 0.5 * (math.Floor(t) + math.Ceil(t))
	default: // QuantileLinear
		v = nm1 * q
	}
	prevF := math.Floor(v)
	prev, next := int(prevF), int(prevF)+1
	if v >= fn-1 {
		prevF, prev, next = -1, n-1, n-1
	}
	if v < 0 {
		prevF, prev, next = 0, 0, 0
	}
	gamma := v - prevF
	switch m {
	case QuantileAveragedInvertedCDF:
		if gamma == 0 {
			gamma = 0.5
		} else {
			gamma = 1
		}
	case QuantileMidpoint:
		if pyMod(v, 1) == 0 {
			gamma = 0
		} else {
			gamma = 0.5
		}
	}
	return quantilePlan{prev: prev, next: next, gamma: gamma}
}

// virtualIndex is numpy's _compute_virtual_index:
// n*q + (alpha + q*(1 - alpha - beta)) - 1, each operation rounded.
func virtualIndex(n, q, alpha, beta float64) float64 {
	return float64(n*q) + (alpha + float64(q*(1-alpha-beta))) - 1
}

// lerp is numpy's _lerp: a + (b-a)t, or b - (b-a)(1-t) when t >= 0.5.
func lerp(a, b, t float64) float64 {
	d := b - a
	if t >= 0.5 {
		return b - float64(d*(1-t))
	}
	return a + float64(d*t)
}

// Quantile returns the q-th quantile of the data, q in [0, 1]:
// numpy.quantile(x, q, method=...), to the bit. The data are not modified
// (a copy is partially ordered by selection, not sorted).
func Quantile[D Data](x D, q float64, o *QuantileOptions) (float64, error) {
	r, err := quantiles(floats(x), []float64{q}, o, "Quantile")
	if err != nil {
		return math.NaN(), err
	}
	return r[0], nil
}

// Quantiles returns the quantiles of the data at each q of qs:
// numpy.quantile(x, qs, method=...), to the bit.
func Quantiles[D Data](x D, qs []float64, o *QuantileOptions) ([]float64, error) {
	return quantiles(floats(x), qs, o, "Quantiles")
}

// Percentile returns the p-th percentile of the data, p in [0, 100]:
// numpy.percentile(x, p, method=...), to the bit (p/100 is then the
// quantile, as NumPy computes it).
func Percentile[D Data](x D, p float64, o *QuantileOptions) (float64, error) {
	r, err := quantiles(floats(x), []float64{p / 100}, o, "Percentile")
	if err != nil {
		return math.NaN(), err
	}
	return r[0], nil
}

// Percentiles returns the percentiles of the data at each p of ps:
// numpy.percentile(x, ps, method=...), to the bit.
func Percentiles[D Data](x D, ps []float64, o *QuantileOptions) ([]float64, error) {
	qs := make([]float64, len(ps))
	for i, p := range ps {
		qs[i] = p / 100
	}
	return quantiles(floats(x), qs, o, "Percentiles")
}

func quantiles(v, qs []float64, o *QuantileOptions, name string) ([]float64, error) {
	if o == nil {
		o = &QuantileOptions{}
	}
	if o.Method < 0 || o.Method > QuantileNearest {
		return nil, fmt.Errorf("%w: %s: unknown method %d", ErrParam, name, o.Method)
	}
	for _, q := range qs {
		if !(q >= 0 && q <= 1) {
			return nil, fmt.Errorf("%w: %s: quantile %v outside [0, 1]", ErrParam, name, q)
		}
	}
	w := o.Weights
	if w != nil {
		if o.Method != QuantileInvertedCDF {
			return nil, fmt.Errorf("%w: %s: weights need the inverted_cdf method, not %v", ErrParam, name, o.Method)
		}
		if len(w) != len(v) {
			return nil, fmt.Errorf("%w: %s: %d weights for %d observations", ErrData, name, len(w), len(v))
		}
		for _, t := range w {
			if t < 0 {
				return nil, fmt.Errorf("%w: %s: negative weight %v", ErrData, name, t)
			}
		}
	}
	if o.OmitNaN && hasNaN(v) {
		keep := make([]float64, 0, len(v))
		var kw []float64
		for i, t := range v {
			if t == t {
				keep = append(keep, t)
				if w != nil {
					kw = append(kw, w[i])
				}
			}
		}
		v, w = keep, kw
	}
	if len(v) == 0 {
		return nil, fmt.Errorf("%w: %s of no observations", ErrData, name)
	}
	out := make([]float64, len(qs))
	if hasNaN(v) {
		for i := range out {
			out[i] = math.NaN()
		}
		return out, nil
	}
	if w != nil {
		return weightedInvertedCDF(v, w, qs, out, name)
	}
	n := len(v)
	plans := make([]quantilePlan, len(qs))
	ks := make([]int, 0, 2*len(qs))
	for i, q := range qs {
		p := planQuantile(n, q, o.Method)
		plans[i] = p
		ks = append(ks, p.prev, p.next)
	}
	a := orderStats(v, uniqueSorted(ks))
	for i, p := range plans {
		if p.discrete {
			out[i] = a[p.prev]
		} else {
			out[i] = lerp(a[p.prev], a[p.next], p.gamma)
		}
	}
	return out, nil
}

// weightedInvertedCDF is numpy's weighted inverted_cdf: the smallest x
// whose normalised cumulative weight reaches q.
func weightedInvertedCDF(v, w, qs, out []float64, name string) ([]float64, error) {
	n := len(v)
	idx := make([]int, n)
	for i := range idx {
		idx[i] = i
	}
	sort.SliceStable(idx, func(i, j int) bool { return v[idx[i]] < v[idx[j]] })
	cdf := make([]float64, n)
	s := 0.0
	for i, k := range idx {
		s += w[k]
		cdf[i] = s
	}
	total := cdf[n-1]
	for i := range cdf {
		cdf[i] /= total
	}
	if math.IsNaN(cdf[n-1]) {
		return nil, fmt.Errorf("%w: %s: weights include NaN or Inf or are all zero", ErrData, name)
	}
	if cdf[0] == 0 {
		for i, c := range cdf {
			if c == 0 {
				cdf[i] = -1
			}
		}
	}
	for i, q := range qs {
		j, _ := slices.BinarySearch(cdf, q)
		out[i] = v[idx[min(j, n-1)]]
	}
	return out, nil
}

// MedianOptions are the options of Median.
type MedianOptions struct {
	// OmitNaN ignores the NaNs (numpy.nanmedian); by default a NaN gives NaN.
	OmitNaN bool
}

// Median returns the median of the data: numpy.median(x), to the bit (the
// middle order statistic, or the mean of the two middle ones).
func Median[D Data](x D, o *MedianOptions) (float64, error) {
	v := floats(x)
	if o != nil && o.OmitNaN {
		v = dropNaN(v)
	}
	if len(v) == 0 {
		return math.NaN(), fmt.Errorf("%w: Median of no observations", ErrData)
	}
	return median(v), nil
}

// median is numpy.median of a non-empty sample.
func median(v []float64) float64 {
	if hasNaN(v) {
		return math.NaN()
	}
	n := len(v)
	if n%2 == 1 {
		return orderStats(v, []int{n / 2})[n/2]
	}
	a := orderStats(v, []int{n/2 - 1, n / 2})
	return mean(a[n/2-1 : n/2+1])
}
