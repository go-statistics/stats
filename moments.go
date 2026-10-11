package stats

import (
	"fmt"
	"math"
)

// Means, variances and moments. Where NumPy computes the statistic
// (numpy.mean, numpy.average, numpy.var, numpy.std), the result is NumPy's
// to the bit: the same pairwise sums, the products rounded separately. The
// SciPy statistics reproduce SciPy's sequence of operations; where SciPy
// calls the C library's pow, log or exp, ours are correctly rounded, so the
// two may differ in the last bit.

// MeanOptions are the options of Mean, Average, Var and Std.
type MeanOptions struct {
	// Weights, when not nil, weight the observations (numpy.average's
	// weights=); they must not sum to zero.
	Weights []float64
	// Ddof is the delta degrees of freedom of Var and Std (numpy's ddof=,
	// default 0): the sum of squares is divided by n - Ddof, or by the sum
	// of the weights minus Ddof with Weights (frequency weights,
	// numpy.cov(x, fweights=w, ddof=Ddof)).
	Ddof float64
	// OmitNaN ignores the NaNs (numpy.nanmean, numpy.nanvar, numpy.nanstd);
	// by default a NaN gives a NaN result.
	OmitNaN bool
}

func meanOptions(o *MeanOptions) *MeanOptions {
	if o == nil {
		return &MeanOptions{}
	}
	return o
}

// prepWeighted validates the data and the weights and applies OmitNaN: a NaN
// observation (or weight) becomes a zero with a zero weight, in place.
func prepWeighted(v []float64, o *MeanOptions, name string) (x, w []float64, n int, err error) {
	w = o.Weights
	if w != nil && len(w) != len(v) {
		return nil, nil, 0, fmt.Errorf("%w: %s: %d weights for %d observations", ErrData, name, len(w), len(v))
	}
	x, n = v, len(v)
	if o.OmitNaN && (hasNaN(v) || hasNaN(w)) {
		x = make([]float64, len(v))
		var ww []float64
		if w != nil {
			ww = make([]float64, len(w))
		}
		n = 0
		for i, t := range v {
			if t == t && (w == nil || w[i] == w[i]) {
				x[i] = t
				if w != nil {
					ww[i] = w[i]
				}
				n++
			}
		}
		w = ww
	}
	if n == 0 {
		return nil, nil, 0, fmt.Errorf("%w: %s of no observations", ErrData, name)
	}
	return x, w, n, nil
}

// average is numpy.average: sum(x*w)/sum(w) (or sum(x)/n), and the
// normaliser.
func average(x, w []float64, n int, name string) (avg, scl float64, err error) {
	if w == nil {
		return sum(x) / float64(n), float64(n), nil
	}
	scl = sum(w)
	if scl == 0 {
		return math.NaN(), 0, fmt.Errorf("%w: %s: weights sum to zero", ErrData, name)
	}
	p := make([]float64, len(x))
	for i := range x {
		p[i] = float64(x[i] * w[i])
	}
	return sum(p) / scl, scl, nil
}

// Mean returns the arithmetic mean of the data: numpy.mean(x) (or
// numpy.nanmean, or numpy.average(x, weights=w)), to the bit.
func Mean[D Data](x D, o *MeanOptions) (float64, error) {
	m, _, err := Average(x, o)
	return m, err
}

// Average returns the weighted mean and the sum of the weights (n without
// weights): numpy.average(x, weights=w, returned=True), to the bit.
func Average[D Data](x D, o *MeanOptions) (avg, sumWeights float64, err error) {
	o = meanOptions(o)
	v, w, n, err := prepWeighted(floats(x), o, "Average")
	if err != nil {
		return math.NaN(), 0, err
	}
	return average(v, w, n, "Average")
}

// Var returns the variance of the data: numpy.var(x, ddof=Ddof) (or
// numpy.nanvar), to the bit. With Weights it is the frequency-weighted
// variance sum(w (x - avg)²) / (sum(w) - Ddof).
func Var[D Data](x D, o *MeanOptions) (float64, error) {
	o = meanOptions(o)
	raw := floats(x)
	v, w, n, err := prepWeighted(raw, o, "Var")
	if err != nil {
		return math.NaN(), err
	}
	m, scl, err := average(v, w, n, "Var")
	if err != nil {
		return math.NaN(), err
	}
	omitted := n < len(v)
	d := make([]float64, len(v))
	for i, t := range v {
		e := t - m
		switch {
		case w != nil:
			d[i] = float64(w[i] * float64(e*e))
		case omitted && raw[i] != raw[i]:
			// numpy.nanvar leaves the NaN positions at zero
		default:
			d[i] = float64(e * e)
		}
	}
	dof := scl - o.Ddof
	switch {
	case w == nil && o.OmitNaN && dof <= 0:
		return math.NaN(), nil
	case w == nil:
		dof = max(dof, 0)
	}
	return sum(d) / dof, nil
}

// Std returns the standard deviation: numpy.std(x, ddof=Ddof), to the bit.
func Std[D Data](x D, o *MeanOptions) (float64, error) {
	v, err := Var(x, o)
	return math.Sqrt(v), err
}

// moment is scipy.stats._moment: mean((x - c)**order) with the power of
// order 2 a product (NumPy's fast path) and the others correctly rounded.
func moment(v []float64, order int, c float64) float64 {
	d := make([]float64, len(v))
	for i, t := range v {
		e := t - c
		switch order {
		case 2:
			d[i] = float64(e * e)
		default:
			d[i] = crPow(e, float64(order))
		}
	}
	return mean(d)
}

// MomentOptions are the options of Moment.
type MomentOptions struct {
	// Center, when not nil, is the point about which the moment is taken
	// (SciPy's center=); by default the mean.
	Center *float64
	// OmitNaN ignores the NaNs (nan_policy="omit"); by default a NaN gives
	// NaN.
	OmitNaN bool
}

// Moment returns the sample moment of the given order about the mean (or
// Center): scipy.stats.moment(x, order, center=Center). Order 0 is 1, and
// order 1 about the mean is 0, as in SciPy.
func Moment[D Data](x D, order int, o *MomentOptions) (float64, error) {
	if o == nil {
		o = &MomentOptions{}
	}
	if order < 0 {
		return math.NaN(), fmt.Errorf("%w: Moment: negative order %d", ErrParam, order)
	}
	v := floats(x)
	if o.OmitNaN {
		v = dropNaN(v)
	}
	if len(v) == 0 {
		return math.NaN(), fmt.Errorf("%w: Moment of no observations", ErrData)
	}
	if hasNaN(v) {
		return math.NaN(), nil
	}
	switch {
	case order == 0:
		return 1, nil
	case order == 1 && o.Center == nil:
		return 0, nil
	}
	c := 0.0
	if o.Center != nil {
		c = *o.Center
	} else {
		c = mean(v)
	}
	return moment(v, order, c), nil
}

// ShapeOptions are the options of Skew, Kurtosis and Describe.
type ShapeOptions struct {
	// Unbiased corrects the statistics for bias (SciPy's bias=False); by
	// default they are the biased moment ratios, SciPy's default.
	Unbiased bool
	// Pearson returns Pearson's kurtosis (3 for a normal sample, SciPy's
	// fisher=False) rather than Fisher's excess kurtosis.
	Pearson bool
	// Ddof is Describe's delta degrees of freedom of the variance; nil
	// means 1, SciPy's default.
	Ddof *float64
	// OmitNaN ignores the NaNs (nan_policy="omit"); by default a NaN gives
	// NaN.
	OmitNaN bool
}

func shapeData(x []float64, o *ShapeOptions, name string) ([]float64, *ShapeOptions, error) {
	if o == nil {
		o = &ShapeOptions{}
	}
	if o.OmitNaN {
		x = dropNaN(x)
	}
	if len(x) == 0 {
		return nil, o, fmt.Errorf("%w: %s of no observations", ErrData, name)
	}
	return x, o, nil
}

// skewness is scipy.stats.skew on a non-empty sample.
func skewness(v []float64, unbiased bool) float64 {
	n := float64(len(v))
	m := mean(v)
	m2 := moment(v, 2, m)
	m3 := moment(v, 3, m)
	if m2 <= float64(eps*m)*float64(eps*m) {
		return math.NaN()
	}
	p := crPow(m2, 1.5)
	if unbiased && n > 2 {
		return float64(math.Sqrt(float64((n-1)*n))/(n-2)*m3) / p
	}
	return m3 / p
}

// eps is the float64 machine epsilon, numpy.finfo(float).eps.
const eps = 0x1p-52

// kurt is scipy.stats.kurtosis on a non-empty sample.
func kurt(v []float64, unbiased, pearson bool) float64 {
	n := float64(len(v))
	m := mean(v)
	m2 := moment(v, 2, m)
	m4 := moment(v, 4, m)
	val := math.NaN()
	if !(m2 <= float64(eps*m)*float64(eps*m)) {
		m22 := float64(m2 * m2)
		val = m4 / m22
		if unbiased && n > 3 {
			a := float64((float64(n*n)-1)*m4) / m22
			b := 3 * float64((n-1)*(n-1))
			val = float64(1/(n-2)/(n-3)*(a-b)) + 3
		}
	}
	if pearson {
		return val
	}
	return val - 3
}

// Skew returns the sample skewness m3 / m2^1.5: scipy.stats.skew(x,
// bias=!Unbiased). NaN when the data are constant.
func Skew[D Data](x D, o *ShapeOptions) (float64, error) {
	v, o, err := shapeData(floats(x), o, "Skew")
	if err != nil {
		return math.NaN(), err
	}
	return skewness(v, o.Unbiased), nil
}

// Kurtosis returns the sample (excess) kurtosis m4 / m2² - 3:
// scipy.stats.kurtosis(x, fisher=!Pearson, bias=!Unbiased). NaN when the
// data are constant.
func Kurtosis[D Data](x D, o *ShapeOptions) (float64, error) {
	v, o, err := shapeData(floats(x), o, "Kurtosis")
	if err != nil {
		return math.NaN(), err
	}
	return kurt(v, o.Unbiased, o.Pearson), nil
}

// DescribeResult is scipy.stats.describe's result.
type DescribeResult struct {
	NObs               int
	Min, Max           float64
	Mean, Variance     float64
	Skewness, Kurtosis float64
}

// Describe returns the count, extremes, mean, variance (ddof 1 by default),
// skewness and excess kurtosis of the data: scipy.stats.describe(x,
// ddof=Ddof, bias=!Unbiased). Pearson is ignored (SciPy reports Fisher's
// kurtosis).
func Describe[D Data](x D, o *ShapeOptions) (DescribeResult, error) {
	v, o, err := shapeData(floats(x), o, "Describe")
	if err != nil {
		return DescribeResult{}, err
	}
	ddof := 1.0
	if o.Ddof != nil {
		ddof = *o.Ddof
	}
	n := len(v)
	r := DescribeResult{NObs: n, Min: v[0], Max: v[0]}
	for _, t := range v {
		if t != t {
			nan := math.NaN()
			return DescribeResult{NObs: n, Min: nan, Max: nan, Mean: nan, Variance: nan, Skewness: nan, Kurtosis: nan}, nil
		}
		r.Min = min(r.Min, t)
		r.Max = max(r.Max, t)
	}
	r.Mean = mean(v)
	r.Variance = moment(v, 2, r.Mean)
	if ddof != 0 {
		fn := float64(n)
		r.Variance *= fn / (fn - ddof)
	}
	r.Skewness = skewness(v, o.Unbiased)
	r.Kurtosis = kurt(v, o.Unbiased, false)
	return r, nil
}

// DdofOptions are the options of SEM, Variation, ZScore and ZMap.
type DdofOptions struct {
	// Ddof is the delta degrees of freedom of the standard deviation; nil
	// means SciPy's default (1 for SEM, 0 for the others).
	Ddof *float64
	// OmitNaN ignores the NaNs (nan_policy="omit"); by default a NaN gives
	// NaN.
	OmitNaN bool
}

func ddofOf(o *DdofOptions, def float64) (float64, bool) {
	if o == nil {
		return def, false
	}
	if o.Ddof == nil {
		return def, o.OmitNaN
	}
	return *o.Ddof, o.OmitNaN
}

// SEM returns the standard error of the mean std(x, ddof)/sqrt(n):
// scipy.stats.sem(x, ddof=Ddof), ddof 1 by default.
func SEM[D Data](x D, o *DdofOptions) (float64, error) {
	ddof, omit := ddofOf(o, 1)
	v := floats(x)
	if omit {
		v = dropNaN(v)
	}
	if len(v) < 2 {
		return math.NaN(), fmt.Errorf("%w: SEM of %d observations", ErrData, len(v))
	}
	n := float64(len(v))
	return math.Sqrt(npVar(v, ddof)) / math.Sqrt(n), nil
}

// npVar is numpy.var(v, ddof=ddof): the divisor max(n - ddof, 0).
func npVar(v []float64, ddof float64) float64 {
	m := mean(v)
	return sum(sqDev(v, m)) / max(float64(len(v))-ddof, 0)
}

// Variation returns the coefficient of variation std(x, ddof)/mean(x):
// scipy.stats.variation(x, ddof=Ddof), ddof 0 by default.
func Variation[D Data](x D, o *DdofOptions) (float64, error) {
	ddof, omit := ddofOf(o, 0)
	v := floats(x)
	if omit {
		v = dropNaN(v)
	}
	if len(v) == 0 {
		return math.NaN(), fmt.Errorf("%w: Variation of no observations", ErrData)
	}
	n := float64(len(v))
	m := mean(v)
	s := math.Sqrt(npVar(v, 0))
	if ddof == n {
		if s > 0 {
			return math.Copysign(math.Inf(1), m), nil
		}
		return math.NaN(), nil
	}
	return float64(s*math.Sqrt(n/(n-ddof))) / m, nil
}

// ZScore returns (x - mean)/std for each observation, in the form of x:
// scipy.stats.zscore(x, ddof=Ddof), ddof 0 by default. All NaN when the
// data are constant.
func ZScore[D Data](x D, o *DdofOptions) (D, error) {
	v := floats(x)
	z, err := zmap(v, v, o, true)
	return like(x, z), err
}

// ZMap returns (scores - mean(compare))/std(compare), in the form of scores:
// scipy.stats.zmap(scores, compare, ddof=Ddof), ddof 0 by default.
func ZMap[D, C Data](scores D, compare C, o *DdofOptions) (D, error) {
	z, err := zmap(floats(scores), floats(compare), o, false)
	return like(scores, z), err
}

func zmap(scores, compare []float64, o *DdofOptions, self bool) ([]float64, error) {
	ddof, omit := ddofOf(o, 0)
	if len(compare) == 0 {
		return nil, fmt.Errorf("%w: ZMap against no observations", ErrData)
	}
	var mn, vr float64
	if omit {
		c, n := zeroNaN(compare)
		if n == 0 {
			return nil, fmt.Errorf("%w: ZMap against no observations", ErrData)
		}
		fn := float64(n)
		mn = sum(c) / fn
		for i, t := range compare {
			c[i] = 0
			if t == t {
				e := t - mn
				c[i] = float64(e * e)
			}
		}
		vr = sum(c) / fn
		vr *= scipyFactor(fn, ddof)
	} else {
		fn := float64(len(compare))
		mn = mean(compare)
		vr = mean(sqDev(compare, mn)) * scipyFactor(fn, ddof)
	}
	sd := math.Sqrt(vr)
	z := make([]float64, len(scores))
	zero := self && sd <= math.Abs(float64(eps*mn))
	for i, t := range scores {
		z[i] = (t - mn) / sd
		if zero {
			z[i] = math.NaN()
		}
	}
	return z, nil
}

// scipyFactor is _xp_var's correction n/(n - ddof), NaN when n - ddof <= 0.
func scipyFactor(n, ddof float64) float64 {
	if ddof == 0 {
		return 1
	}
	if nc := n - ddof; nc > 0 {
		return n / nc
	}
	return math.NaN()
}

// PowerMeanOptions are the options of GMean, HMean and PMean.
type PowerMeanOptions struct {
	// Weights, when not nil, weight the observations (SciPy's weights=).
	Weights []float64
	// OmitNaN ignores the NaNs (nan_policy="omit"); by default a NaN gives
	// NaN.
	OmitNaN bool
}

// xpMean is SciPy's _xp_mean: mean(x), or sum(x*w)/sum(w).
func xpMean(x, w []float64) float64 {
	if w == nil {
		return mean(x)
	}
	p := make([]float64, len(x))
	for i := range x {
		p[i] = float64(x[i] * w[i])
	}
	return sum(p) / sum(w)
}

func powerMeanData(x []float64, o *PowerMeanOptions, name string) (v, w []float64, err error) {
	if o == nil {
		o = &PowerMeanOptions{}
	}
	v, w = x, o.Weights
	if w != nil && len(w) != len(v) {
		return nil, nil, fmt.Errorf("%w: %s: %d weights for %d observations", ErrData, name, len(w), len(v))
	}
	if o.OmitNaN && (hasNaN(v) || hasNaN(w)) {
		var kv, kw []float64
		for i, t := range v {
			if t == t && (w == nil || w[i] == w[i]) {
				kv = append(kv, t)
				if w != nil {
					kw = append(kw, w[i])
				}
			}
		}
		v, w = kv, kw
	}
	if len(v) == 0 {
		return nil, nil, fmt.Errorf("%w: %s of no observations", ErrData, name)
	}
	return v, w, nil
}

// mapF returns f applied to each element.
func mapF(v []float64, f func(float64) float64) []float64 {
	out := make([]float64, len(v))
	for i, t := range v {
		out[i] = f(t)
	}
	return out
}

// GMean returns the geometric mean exp(mean(log x)): scipy.stats.gmean(x,
// weights=Weights). Negative data give NaN, a zero gives 0.
func GMean[D Data](x D, o *PowerMeanOptions) (float64, error) {
	v, w, err := powerMeanData(floats(x), o, "GMean")
	if err != nil {
		return math.NaN(), err
	}
	return gmean(v, w), nil
}

func gmean(v, w []float64) float64 {
	return crExp(xpMean(mapF(v, crLog), w))
}

// HMean returns the harmonic mean 1/mean(1/x): scipy.stats.hmean(x,
// weights=Weights). Negative data give NaN.
func HMean[D Data](x D, o *PowerMeanOptions) (float64, error) {
	v, w, err := powerMeanData(floats(x), o, "HMean")
	if err != nil {
		return math.NaN(), err
	}
	r := mapF(v, func(t float64) float64 {
		if t < 0 {
			return math.NaN()
		}
		return 1 / t
	})
	return 1 / xpMean(r, w), nil
}

// PMean returns the power mean mean(x^p)^(1/p): scipy.stats.pmean(x, p,
// weights=Weights); p = 0 is the geometric mean, and for |p| <= 2e-6 SciPy's
// first-order expansion about p = 0 is used. Negative data give NaN.
func PMean[D Data](x D, p float64, o *PowerMeanOptions) (float64, error) {
	if math.IsInf(p, 0) || math.IsNaN(p) {
		return math.NaN(), fmt.Errorf("%w: PMean: exponent %v is not finite", ErrParam, p)
	}
	v, w, err := powerMeanData(floats(x), o, "PMean")
	if err != nil {
		return math.NaN(), err
	}
	if p == 0 {
		return gmean(v, w), nil
	}
	v = mapF(v, func(t float64) float64 {
		if t < 0 {
			return math.NaN()
		}
		return t
	})
	if math.Abs(p) <= 2e-6 {
		m0 := gmean(v, w)
		la := mapF(v, crLog)
		lnAvg := xpMean(la, w)
		ln2Avg := xpMean(mapF(la, func(t float64) float64 { return t * t }), w)
		return m0 * (1 + float64(0.5*p*(ln2Avg-float64(lnAvg*lnAvg)))), nil
	}
	return crPow(xpMean(mapF(v, func(t float64) float64 { return crPow(t, p) }), w), 1/p), nil
}
