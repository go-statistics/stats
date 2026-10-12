package stats

import (
	"fmt"
	"math"
	"slices"
)

// Mode, interquartile range, median absolute deviation and the trimmed
// statistics.

// ModeResult is scipy.stats.mode's result.
type ModeResult struct {
	// Mode is the smallest of the most frequent values.
	Mode float64
	// Count is its number of occurrences.
	Count int
}

// NaNOptions are the options of the statistics whose only option is the
// NaN policy.
type NaNOptions struct {
	// OmitNaN ignores the NaNs (nan_policy="omit"); by default a NaN gives a
	// NaN result (for Mode, the NaNs count as one value, as in SciPy).
	OmitNaN bool
}

// Mode returns the most frequent value of the data (the smallest of them on
// a tie) and its count: scipy.stats.mode(x). All NaNs count as one value,
// ordered after every number.
func Mode[D Data](x D, o *NaNOptions) (ModeResult, error) {
	v := floats(x)
	if o != nil && o.OmitNaN {
		v = dropNaN(v)
	}
	if len(v) == 0 {
		return ModeResult{Mode: math.NaN()}, fmt.Errorf("%w: Mode of no observations", ErrData)
	}
	a := slices.Clone(v)
	n := nanLast(a)
	slices.Sort(a[:n])
	best := ModeResult{Mode: a[0]}
	for i := 0; i < n; {
		j := i + 1
		for j < n && a[j] == a[i] {
			j++
		}
		if j-i > best.Count {
			best = ModeResult{Mode: a[i], Count: j - i}
		}
		i = j
	}
	if k := len(a) - n; k > best.Count {
		best = ModeResult{Mode: math.NaN(), Count: k}
	}
	return best, nil
}

// Scales of IQR and MedianAbsDeviation that make them consistent
// estimators of the standard deviation of a normal sample (SciPy's
// scale="normal").
const (
	// IQRNormal is 2 Φ⁻¹(3/4), scipy.stats.iqr's scale="normal".
	IQRNormal = 1.3489795003921634
	// MADNormal is Φ⁻¹(3/4), scipy.stats.median_abs_deviation's
	// scale="normal".
	MADNormal = 0.6744897501960817
)

// IQROptions are the options of IQR.
type IQROptions struct {
	// Range is the pair of percentiles (SciPy's rng=); the zero value
	// means (25, 75).
	Range [2]float64
	// Scale divides the result (0 means 1); IQRNormal is scale="normal".
	Scale float64
	// Method is the quantile estimator (SciPy's interpolation=), computed
	// as scipy.stats.quantile does.
	Method QuantileMethod
	// OmitNaN ignores the NaNs; by default a NaN gives NaN.
	OmitNaN bool
}

// IQR returns the interquartile range (or the range between the two
// percentiles of Range): scipy.stats.iqr(x, rng=Range, scale=Scale,
// interpolation=Method).
func IQR[D Data](x D, o *IQROptions) (float64, error) {
	if o == nil {
		o = &IQROptions{}
	}
	r := o.Range
	if r == [2]float64{} {
		r = [2]float64{25, 75}
	}
	if math.IsNaN(r[0]) || math.IsNaN(r[1]) {
		return math.NaN(), fmt.Errorf("%w: IQR: range %v holds a NaN", ErrParam, r)
	}
	lo, hi := r[0]/100, r[1]/100
	if r[0] >= r[1] {
		lo, hi = r[1]/100, r[0]/100
	}
	if lo < 0 || hi > 1 {
		return math.NaN(), fmt.Errorf("%w: IQR: range %v outside [0, 100]", ErrParam, r)
	}
	if o.Method < 0 || o.Method > QuantileNearest {
		return math.NaN(), fmt.Errorf("%w: IQR: unknown method %d", ErrParam, o.Method)
	}
	v := floats(x)
	if o.OmitNaN {
		v = dropNaN(v)
	}
	if len(v) == 0 {
		return math.NaN(), fmt.Errorf("%w: IQR of no observations", ErrData)
	}
	if hasNaN(v) {
		return math.NaN(), nil
	}
	q := scipyQuantiles(v, []float64{lo, hi}, o.Method)
	out := q[1] - q[0]
	if o.Scale != 0 && o.Scale != 1 {
		out /= o.Scale
	}
	return out, nil
}

// scipyQuantiles is scipy.stats.quantile (which iqr calls) on NaN-free
// data: the Hyndman and Fan methods as (1-g) y[j] + g y[j+1], and the
// discontinuous ones on p(n-1).
func scipyQuantiles(v, ps []float64, m QuantileMethod) []float64 {
	n := float64(len(v))
	type ix struct {
		j, jp1 int
		g      float64
		mid    bool
	}
	plans := make([]ix, len(ps))
	var ks []int
	clip := func(t float64) int { return int(min(max(t, 0), n-1)) }
	for i, p := range ps {
		var pl ix
		switch m {
		case QuantileLower, QuantileHigher, QuantileMidpoint, QuantileNearest:
			ij := p * (n - 1)
			switch m {
			case QuantileLower:
				pl.j = int(math.Floor(ij))
				pl.jp1 = pl.j
			case QuantileHigher:
				pl.j = int(math.Ceil(ij))
				pl.jp1 = pl.j
			case QuantileNearest:
				pl.j = int(math.RoundToEven(ij))
				pl.jp1 = pl.j
			default:
				pl.j, pl.jp1, pl.mid = int(math.Floor(ij)), int(math.Ceil(ij)), true
			}
		default:
			var mm float64
			switch m {
			case QuantileClosestObservation:
				mm = -0.5
			case QuantileHazen:
				mm = 0.5
			case QuantileWeibull:
				mm = p
			case QuantileLinear:
				mm = 1 - p
			case QuantileMedianUnbiased:
				mm = p/3 + 1.0/3
			case QuantileNormalUnbiased:
				mm = p/4 + 3.0/8
			}
			jg := float64(p*n) + mm
			jp1 := math.Floor(jg)
			j := jp1 - 1
			g := pyMod(jg, 1)
			switch m {
			case QuantileInvertedCDF:
				g = b2f(g > 0)
			case QuantileAveragedInvertedCDF:
				g = (1 + b2f(g > 0)) / 2
			case QuantileClosestObservation:
				g = 1 - b2f(g == 0 && pyMod(j, 2) == 1)
			}
			switch m {
			case QuantileInvertedCDF, QuantileAveragedInvertedCDF, QuantileClosestObservation:
				if jg < 0 {
					g = 0
				}
			}
			if j < 0 {
				g = 0
			}
			pl.j, pl.jp1, pl.g = clip(j), clip(jp1), g
		}
		plans[i] = pl
		ks = append(ks, pl.j, pl.jp1)
	}
	a := orderStats(v, uniqueSorted(ks))
	out := make([]float64, len(ps))
	for i, pl := range plans {
		switch {
		case pl.mid:
			out[i] = (a[pl.j] + a[pl.jp1]) / 2
		case m >= QuantileLower:
			out[i] = a[pl.j]
		default:
			out[i] = float64((1-pl.g)*a[pl.j]) + float64(pl.g*a[pl.jp1])
		}
	}
	return out
}

func b2f(b bool) float64 {
	if b {
		return 1
	}
	return 0
}

// MADOptions are the options of MedianAbsDeviation.
type MADOptions struct {
	// Scale divides the result (0 means 1); MADNormal is scale="normal".
	Scale float64
	// Center, when not nil, computes the center from the (NaN-free) data;
	// by default the median.
	Center func([]float64) float64
	// OmitNaN ignores the NaNs; by default a NaN gives NaN.
	OmitNaN bool
}

// MedianAbsDeviation returns the median of |x - center| divided by Scale:
// scipy.stats.median_abs_deviation(x, center=Center, scale=Scale).
func MedianAbsDeviation[D Data](x D, o *MADOptions) (float64, error) {
	if o == nil {
		o = &MADOptions{}
	}
	v := floats(x)
	if o.OmitNaN {
		v = dropNaN(v)
	}
	if len(v) == 0 {
		return math.NaN(), fmt.Errorf("%w: MedianAbsDeviation of no observations", ErrData)
	}
	if hasNaN(v) {
		return math.NaN(), nil
	}
	c := 0.0
	if o.Center != nil {
		c = o.Center(v)
	} else {
		c = median(v)
	}
	mad := median(mapF(v, func(t float64) float64 { return math.Abs(t - c) }))
	s := o.Scale
	if s == 0 {
		s = 1
	}
	return mad / s, nil
}

// Limits restricts the trimmed statistics to the observations between
// Lower and Upper (use an infinity for an open side), SciPy's limits= and
// inclusive= arguments.
type Limits struct {
	Lower, Upper float64
	// LowerExclusive and UpperExclusive drop the observations equal to the
	// limit (SciPy's inclusive=(False, ...)).
	LowerExclusive, UpperExclusive bool
}

// TrimmedOptions are the options of TMean, TVar, TStd and TSem.
type TrimmedOptions struct {
	// Limits, when not nil, restrict the data; nil keeps all of them.
	Limits *Limits
	// Ddof is the delta degrees of freedom of TVar, TStd and TSem; nil
	// means 1, SciPy's default.
	Ddof *float64
	// OmitNaN ignores the NaNs; by default a NaN gives NaN.
	OmitNaN bool
}

// trimmedMask returns the data with the observations outside the limits
// zeroed in place, and the count of those kept (SciPy's
// _put_val_to_limits).
func trimmedMask(x []float64, o *TrimmedOptions, name string) (z []float64, kept []bool, n int, err error) {
	if o == nil {
		o = &TrimmedOptions{}
	}
	v := x
	if o.OmitNaN {
		v = dropNaN(v)
	}
	if len(v) == 0 {
		return nil, nil, 0, fmt.Errorf("%w: %s of no observations", ErrData, name)
	}
	z = make([]float64, len(v))
	kept = make([]bool, len(v))
	for i, t := range v {
		out := false
		if l := o.Limits; l != nil {
			out = t < l.Lower || t > l.Upper || (l.LowerExclusive && t == l.Lower) || (l.UpperExclusive && t == l.Upper)
		}
		if !out {
			z[i], kept[i] = t, true
			n++
		}
	}
	if n == 0 {
		return nil, nil, 0, fmt.Errorf("%w: %s: no observation within the limits", ErrData, name)
	}
	return z, kept, n, nil
}

// TMean returns the mean of the observations within the limits:
// scipy.stats.tmean(x, limits, inclusive).
func TMean[D Data](x D, o *TrimmedOptions) (float64, error) {
	z, _, n, err := trimmedMask(floats(x), o, "TMean")
	if err != nil {
		return math.NaN(), err
	}
	return sum(z) / float64(n), nil
}

// TVar returns the variance (ddof 1 by default) of the observations within
// the limits: scipy.stats.tvar(x, limits, inclusive, ddof=Ddof).
func TVar[D Data](x D, o *TrimmedOptions) (float64, error) {
	v, _, err := tvar(floats(x), o, "TVar")
	return v, err
}

func tvar(x []float64, o *TrimmedOptions, name string) (v, n float64, err error) {
	z, kept, k, err := trimmedMask(x, o, name)
	if err != nil {
		return math.NaN(), 0, err
	}
	ddof := 1.0
	if o != nil && o.Ddof != nil {
		ddof = *o.Ddof
	}
	n = float64(k)
	m := sum(z) / n
	for i, t := range z {
		if kept[i] {
			e := t - m
			z[i] = float64(e * e)
		}
	}
	return sum(z) / n * scipyFactor(n, ddof), n, nil
}

// TStd returns the square root of TVar: scipy.stats.tstd.
func TStd[D Data](x D, o *TrimmedOptions) (float64, error) {
	v, _, err := tvar(floats(x), o, "TStd")
	return math.Sqrt(v), err
}

// TSem returns the standard error of the mean of the observations within
// the limits: scipy.stats.tsem.
func TSem[D Data](x D, o *TrimmedOptions) (float64, error) {
	v, n, err := tvar(floats(x), o, "TSem")
	return math.Sqrt(v) / math.Sqrt(n), err
}

// BoundOptions are the options of TMin and TMax.
type BoundOptions struct {
	// Exclusive drops the observations equal to the limit (SciPy's
	// inclusive=False).
	Exclusive bool
	// OmitNaN ignores the NaNs; by default a NaN gives NaN.
	OmitNaN bool
}

// TMin returns the smallest observation not below lower (math.Inf(-1) for
// none): scipy.stats.tmin(x, lowerlimit, inclusive=!Exclusive).
func TMin[D Data](x D, lower float64, o *BoundOptions) (float64, error) {
	return tbound(floats(x), lower, o, -1, "TMin")
}

// TMax returns the largest observation not above upper (math.Inf(1) for
// none): scipy.stats.tmax(x, upperlimit, inclusive=!Exclusive).
func TMax[D Data](x D, upper float64, o *BoundOptions) (float64, error) {
	return tbound(floats(x), upper, o, 1, "TMax")
}

func tbound(x []float64, limit float64, o *BoundOptions, side float64, name string) (float64, error) {
	if o == nil {
		o = &BoundOptions{}
	}
	if o.OmitNaN {
		x = dropNaN(x)
	}
	if len(x) == 0 {
		return math.NaN(), fmt.Errorf("%w: %s of no observations", ErrData, name)
	}
	if hasNaN(x) {
		return math.NaN(), nil
	}
	best, found := 0.0, false
	for _, t := range x {
		d := side * (t - limit) // > 0 beyond the limit
		if d > 0 || (o.Exclusive && t == limit) {
			continue
		}
		if !found || side*(t-best) > 0 {
			best, found = t, true
		}
	}
	if !found {
		return math.NaN(), fmt.Errorf("%w: %s: no observation within the limit %v", ErrData, name, limit)
	}
	return best, nil
}

// trimCuts partitions a copy of the data so that the order statistics
// lowercut .. uppercut-1 are in place (NaN ordered last, as NumPy does).
func trimCuts(v []float64, lowercut, uppercut int) []float64 {
	return orderStats(v, uniqueSorted([]int{lowercut, uppercut - 1}))
}

// TrimMean returns the mean after cutting int(proportion n) observations
// from each end: scipy.stats.trim_mean(x, proportion).
func TrimMean[D Data](x D, proportion float64) (float64, error) {
	v := floats(x)
	if len(v) == 0 {
		return math.NaN(), fmt.Errorf("%w: TrimMean of no observations", ErrData)
	}
	n := len(v)
	lc := int(proportion * float64(n))
	uc := n - lc
	if lc > uc || proportion < 0 {
		return math.NaN(), fmt.Errorf("%w: TrimMean: proportion %v too big", ErrParam, proportion)
	}
	if hasNaN(v) {
		return math.NaN(), nil
	}
	if lc == uc {
		return math.NaN(), nil
	}
	return mean(trimCuts(v, lc, uc)[lc:uc]), nil
}

// TrimBoth returns the data without int(proportion n) observations at each
// end, in no particular order: scipy.stats.trimboth(x, proportion).
func TrimBoth[D Data](x D, proportion float64) (D, error) {
	v := floats(x)
	n := len(v)
	if n == 0 {
		return like(x, []float64{}), nil
	}
	lc := int(proportion * float64(n))
	uc := n - lc
	if lc >= uc || proportion < 0 {
		return like(x, nil), fmt.Errorf("%w: TrimBoth: proportion %v too big", ErrParam, proportion)
	}
	return like(x, slices.Clone(trimCuts(v, lc, uc)[lc:uc])), nil
}

// Tail selects the end that Trim1 cuts.
type Tail int

const (
	// TailRight cuts the largest observations (SciPy's default).
	TailRight Tail = iota
	// TailLeft cuts the smallest observations.
	TailLeft
)

// Trim1 returns the data without int(proportion n) observations at one
// end, in no particular order: scipy.stats.trim1(x, proportion, tail).
func Trim1[D Data](x D, proportion float64, tail Tail) (D, error) {
	v := floats(x)
	n := len(v)
	if proportion < 0 {
		return like(x, nil), fmt.Errorf("%w: Trim1: negative proportion %v", ErrParam, proportion)
	}
	if proportion >= 1 || n == 0 {
		return like(x, []float64{}), nil
	}
	lc, uc := 0, n-int(proportion*float64(n))
	if tail == TailLeft {
		lc, uc = int(proportion*float64(n)), n
	}
	return like(x, slices.Clone(trimCuts(v, lc, uc)[lc:uc])), nil
}
