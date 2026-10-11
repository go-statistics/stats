package stats

import (
	"fmt"
	"math"
	"slices"
	"sort"

	"github.com/go-statistics/stats/internal/dd"
)

// BinStatistic is the statistic BinnedStatistic computes in each bin.
type BinStatistic int

const (
	// BinMean is statistic="mean" (the zero value, SciPy's default).
	BinMean BinStatistic = iota
	// BinStd is statistic="std" (ddof 0).
	BinStd
	// BinMedian is statistic="median".
	BinMedian
	// BinCount is statistic="count".
	BinCount
	// BinSum is statistic="sum".
	BinSum
	// BinMin is statistic="min".
	BinMin
	// BinMax is statistic="max".
	BinMax
)

// BinnedOptions are the options of BinnedStatistic.
type BinnedOptions struct {
	// Statistic is the statistic of each bin (default BinMean).
	Statistic BinStatistic
	// Bins is the number of equal-width bins (0 means 10).
	Bins int
	// Edges, when not nil, are the bin edges (increasing); they take
	// precedence over Bins.
	Edges []float64
	// Range, when not nil, is the range of the equal-width bins; by default
	// the minimum and maximum of x.
	Range *[2]float64
}

// BinnedResult is scipy.stats.binned_statistic's result.
type BinnedResult struct {
	// Statistic is the statistic of the values in each bin (NaN for an
	// empty bin, except 0 for BinCount and BinSum).
	Statistic []float64
	// Edges are the len(Statistic)+1 bin edges.
	Edges []float64
	// BinNumber is the bin of each observation, 1-based: 0 below the first
	// edge and len(Edges) above the last.
	BinNumber []int
}

// BinnedStatistic computes a statistic of values in bins of x:
// scipy.stats.binned_statistic(x, values, statistic, bins, range). The bins
// are half-open [a, b) except the last, which is closed (to the precision
// SciPy rounds the last edge to).
func BinnedStatistic[D, V Data](x D, values V, o *BinnedOptions) (BinnedResult, error) {
	if o == nil {
		o = &BinnedOptions{}
	}
	xs, vs := floats(x), floats(values)
	if o.Statistic < BinMean || o.Statistic > BinMax {
		return BinnedResult{}, fmt.Errorf("%w: BinnedStatistic: unknown statistic %d", ErrParam, o.Statistic)
	}
	if o.Statistic != BinCount && len(vs) != len(xs) {
		return BinnedResult{}, fmt.Errorf("%w: BinnedStatistic: %d values for %d observations", ErrData, len(vs), len(xs))
	}
	if hasNaN(xs) {
		return BinnedResult{}, fmt.Errorf("%w: BinnedStatistic: x holds a NaN", ErrData)
	}
	var edges []float64
	if o.Edges != nil {
		edges = slices.Clone(o.Edges)
	} else {
		nb := o.Bins
		if nb == 0 {
			nb = 10
		}
		if nb < 0 {
			return BinnedResult{}, fmt.Errorf("%w: BinnedStatistic: %d bins", ErrParam, nb)
		}
		var lo, hi float64
		if o.Range != nil {
			lo, hi = o.Range[0], o.Range[1]
			if hi < lo {
				return BinnedResult{}, fmt.Errorf("%w: BinnedStatistic: range %v is decreasing", ErrParam, *o.Range)
			}
		} else {
			lo, hi = math.NaN(), math.NaN()
			for _, t := range xs {
				if !math.IsInf(t, 0) {
					if lo != lo || t < lo {
						lo = t
					}
					if hi != hi || t > hi {
						hi = t
					}
				}
			}
		}
		if lo == hi {
			lo, hi = lo-0.5, hi+0.5
		}
		edges = linspace(lo, hi, nb+1)
	}
	if len(edges) < 2 {
		return BinnedResult{}, fmt.Errorf("%w: BinnedStatistic: %d edges", ErrParam, len(edges))
	}
	dmin := math.Inf(1)
	for i := 1; i < len(edges); i++ {
		dmin = min(dmin, edges[i]-edges[i-1])
	}
	if dmin == 0 {
		return BinnedResult{}, fmt.Errorf("%w: BinnedStatistic: the smallest edge difference is 0", ErrParam)
	}
	nbin := len(edges) + 1
	last := edges[len(edges)-1]
	decimal := int(-crLog10(dmin)) + 6
	lastR := npAround(last, decimal)
	bins := make([]int, len(xs))
	for i, t := range xs {
		b := sort.Search(len(edges), func(j int) bool { return edges[j] > t })
		if t >= last && npAround(t, decimal) == lastR {
			b--
		}
		bins[i] = b
	}
	stat := binStat(bins, vs, nbin, o.Statistic)
	return BinnedResult{Statistic: stat[1 : nbin-1], Edges: edges, BinNumber: bins}, nil
}

// binStat is binned_statistic_dd's per-bin computation over all nbin bins
// (outliers included), with np.bincount's sequential sums.
func binStat(bins []int, vs []float64, nbin int, s BinStatistic) []float64 {
	res := make([]float64, nbin)
	count := make([]float64, nbin)
	for _, b := range bins {
		count[b]++
	}
	sums := func(f func(i int) float64) []float64 {
		r := make([]float64, nbin)
		for i, b := range bins {
			r[b] += f(i)
		}
		return r
	}
	switch s {
	case BinCount:
		return count
	case BinSum:
		return sums(func(i int) float64 { return vs[i] })
	}
	for i := range res {
		res[i] = math.NaN()
	}
	switch s {
	case BinMean, BinStd:
		sm := sums(func(i int) float64 { return vs[i] })
		if s == BinStd {
			sq := sums(func(i int) float64 {
				d := vs[i] - sm[bins[i]]/count[bins[i]]
				return float64(d * d)
			})
			for b := range res {
				if count[b] != 0 {
					res[b] = math.Sqrt(sq[b] / count[b])
				}
			}
			return res
		}
		for b := range res {
			if count[b] != 0 {
				res[b] = sm[b] / count[b]
			}
		}
	case BinMedian:
		per := make([][]float64, nbin)
		for i, b := range bins {
			per[b] = append(per[b], vs[i])
		}
		for b, v := range per {
			if len(v) > 0 {
				slices.SortFunc(v, func(p, q float64) int {
					switch {
					case lessNaN(p, q):
						return -1
					case lessNaN(q, p):
						return 1
					}
					return 0
				})
				mid := float64(len(v)-1) / 2
				res[b] = (v[int(math.Floor(mid))] + v[int(math.Ceil(mid))]) / 2
			}
		}
	default:
		// SciPy assigns the values in sorted order (descending for the
		// minimum), the last write winning: the minimum skips NaNs unless
		// all are NaN, the maximum is NaN as soon as one is.
		has := make([]bool, nbin)
		for i, b := range bins {
			t, r := vs[i], res[b]
			switch {
			case !has[b]:
				res[b] = t
			case s == BinMin && (t < r || r != r):
				res[b] = t
			case s == BinMax && (t > r || t != t):
				res[b] = t
			}
			has[b] = true
		}
	}
	return res
}

// crLog10 returns log10(x) correctly rounded (barring a near-tie).
func crLog10(x float64) float64 {
	if !(x > 0) || math.IsInf(x, 1) {
		return math.Log10(x)
	}
	return dd.Div(dd.LogAccurate(x), dd.LogAccurate(10)).Float()
}

// npAround is numpy.around(x, decimals): rint(x * 10^d) / 10^d (or with
// the scale inverted for negative d).
func npAround(x float64, d int) float64 {
	if d >= 0 {
		f := crPow(10, float64(d))
		return math.RoundToEven(x*f) / f
	}
	f := crPow(10, float64(-d))
	return math.RoundToEven(x/f) * f
}
