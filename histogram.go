package stats

import (
	"fmt"
	"math"
	"slices"
	"sort"
)

// BinRule is an estimator of the number of histogram bins, NumPy's
// bins="auto", "fd", ... The zero value BinsCount means "use Bins".
type BinRule int

const (
	// BinsCount uses HistogramOptions.Bins (or Edges).
	BinsCount BinRule = iota
	// BinsAuto is bins="auto": the smaller of the Sturges and the
	// Freedman-Diaconis widths (the latter at least half the sqrt width).
	BinsAuto
	// BinsFD is bins="fd" (Freedman-Diaconis): width 2 IQR n^(-1/3).
	BinsFD
	// BinsDoane is bins="doane".
	BinsDoane
	// BinsScott is bins="scott".
	BinsScott
	// BinsStone is bins="stone" (minimises the leave-one-out risk).
	BinsStone
	// BinsRice is bins="rice".
	BinsRice
	// BinsSturges is bins="sturges".
	BinsSturges
	// BinsSqrt is bins="sqrt".
	BinsSqrt
)

// HistogramOptions are the options of Histogram and HistogramBinEdges. The
// zero value is NumPy's default: 10 equal bins over the data's range.
type HistogramOptions struct {
	// Bins is the number of equal-width bins (0 means 10).
	Bins int
	// Edges, when not nil, are the bin edges (non-decreasing); they take
	// precedence over Bins and Rule.
	Edges []float64
	// Rule, when not BinsCount, estimates the bin width from the data.
	Rule BinRule
	// Range, when not nil, is the (finite) range of the equal-width bins;
	// observations outside it (and NaNs) are ignored. By default the
	// data's minimum and maximum.
	Range *[2]float64
	// Density normalises the counts to a probability density.
	Density bool
	// Weights, when not nil, weight the observations.
	Weights []float64
}

// HistogramResult is numpy.histogram's result.
type HistogramResult struct {
	// Counts are the (weighted) counts, or densities, of the bins.
	Counts []float64
	// Edges are the len(Counts)+1 bin edges.
	Edges []float64
}

// linspace is numpy.linspace(start, stop, num), to the bit.
func linspace(start, stop float64, num int) []float64 {
	y := make([]float64, num)
	div := num - 1
	delta := stop - start
	if div > 0 {
		step := delta / float64(div)
		for i := range y {
			if step == 0 {
				y[i] = float64(float64(i)/float64(div)*delta) + start
			} else {
				y[i] = float64(float64(i)*step) + start
			}
		}
		y[num-1] = stop
	} else if num == 1 {
		y[0] = float64(0*delta) + start
	}
	return y
}

// binEdges is numpy's _get_bin_edges: the edges, and for equal-width bins
// the number of bins (0 for explicit edges) and the outer edges.
func binEdges(v []float64, o *HistogramOptions) (edges []float64, nb int, first, last float64, err error) {
	if o.Edges != nil {
		if len(o.Edges) < 2 {
			return nil, 0, 0, 0, fmt.Errorf("%w: Histogram: %d edges", ErrParam, len(o.Edges))
		}
		for i := 1; i < len(o.Edges); i++ {
			if !(o.Edges[i-1] <= o.Edges[i]) {
				return nil, 0, 0, 0, fmt.Errorf("%w: Histogram: edges must increase monotonically", ErrParam)
			}
		}
		return o.Edges, 0, 0, 0, nil
	}
	if o.Rule < BinsCount || o.Rule > BinsSqrt {
		return nil, 0, 0, 0, fmt.Errorf("%w: Histogram: unknown bin rule %d", ErrParam, o.Rule)
	}
	if o.Bins < 0 {
		return nil, 0, 0, 0, fmt.Errorf("%w: Histogram: %d bins", ErrParam, o.Bins)
	}
	// outer edges (numpy's _get_outer_edges)
	if r := o.Range; r != nil {
		first, last = r[0], r[1]
		if first > last {
			return nil, 0, 0, 0, fmt.Errorf("%w: Histogram: range %v is decreasing", ErrParam, *r)
		}
		if math.IsInf(first, 0) || math.IsInf(last, 0) || math.IsNaN(first) || math.IsNaN(last) {
			return nil, 0, 0, 0, fmt.Errorf("%w: Histogram: range %v is not finite", ErrParam, *r)
		}
	} else if len(v) == 0 {
		first, last = 0, 1
	} else {
		first, last = v[0], v[0]
		for _, t := range v {
			if t != t {
				return nil, 0, 0, 0, fmt.Errorf("%w: Histogram: autodetected range of data with NaN", ErrData)
			}
			first, last = min(first, t), max(last, t)
		}
		if math.IsInf(first, 0) || math.IsInf(last, 0) {
			return nil, 0, 0, 0, fmt.Errorf("%w: Histogram: autodetected range [%v, %v] is not finite", ErrData, first, last)
		}
	}
	if first == last {
		first, last = first-0.5, last+0.5
	}
	switch {
	case o.Rule != BinsCount:
		if o.Weights != nil {
			return nil, 0, 0, 0, fmt.Errorf("%w: Histogram: a bin rule cannot estimate from weighted data", ErrParam)
		}
		a := v
		if o.Range != nil {
			a = nil
			for _, t := range v {
				if t >= first && t <= last {
					a = append(a, t)
				}
			}
		}
		nb = 1
		if len(a) > 0 {
			if w := binWidth(a, o.Rule, first, last); w != 0 {
				nb = int(math.Ceil((last - first) / w))
			}
		}
	case o.Bins == 0:
		nb = 10
	default:
		nb = o.Bins
	}
	edges = linspace(first, last, nb+1)
	for i := 1; i < len(edges); i++ {
		if !(edges[i-1] < edges[i]) {
			return nil, 0, 0, 0, fmt.Errorf("%w: Histogram: too many bins (%d) for the data range", ErrParam, nb)
		}
	}
	return edges, nb, first, last, nil
}

// binWidth is numpy's _hist_bin_* estimators on non-empty, finite data.
func binWidth(a []float64, rule BinRule, first, last float64) float64 {
	n := float64(len(a))
	lo, hi := slices.Min(a), slices.Max(a)
	ptp := hi - lo
	sqrtW := func() float64 { return ptp / math.Sqrt(n) }
	sturges := func() float64 { return ptp / (crLog2(n) + 1.0) }
	fd := func() float64 {
		q := quantilesNoNaN(a, []float64{0.75, 0.25})
		return float64(2.0*(q[0]-q[1])) * crPow(n, -1.0/3.0)
	}
	switch rule {
	case BinsSqrt:
		return sqrtW()
	case BinsSturges:
		return sturges()
	case BinsRice:
		return ptp / (2.0 * crPow(n, 1.0/3))
	case BinsScott:
		return crPow(24.0*math.Sqrt(math.Pi)/n, 1.0/3.0) * math.Sqrt(npVar(a, 0))
	case BinsFD:
		return fd()
	case BinsDoane:
		if n > 2 {
			sg1 := math.Sqrt(6.0 * (n - 2) / ((n + 1.0) * (n + 3)))
			sigma := math.Sqrt(npVar(a, 0))
			if sigma > 0 {
				m := mean(a)
				t := make([]float64, len(a))
				for i, x := range a {
					t[i] = crPow((x-m)/sigma, 3)
				}
				g1 := mean(t)
				return ptp / (1.0 + crLog2(n) + crLog2(1.0+math.Abs(g1)/sg1))
			}
		}
		return 0
	case BinsStone:
		return stoneWidth(a, ptp, first, last)
	}
	// BinsAuto
	return min(max(fd(), sqrtW()/2), sturges())
}

// quantilesNoNaN is numpy.percentile's linear method on NaN-free data.
func quantilesNoNaN(a, qs []float64) []float64 {
	r, _ := quantiles(a, qs, nil, "")
	return r
}

// stoneWidth is numpy's _hist_bin_stone.
func stoneWidth(a []float64, ptp, first, last float64) float64 {
	n := len(a)
	if n <= 1 || ptp == 0 {
		return 0
	}
	fn := float64(n)
	upper := max(100, int(math.Sqrt(fn)))
	best, bestJ := 0, math.Inf(1)
	for nb := 1; nb <= upper; nb++ {
		hh := ptp / float64(nb)
		edges := linspace(first, last, nb+1)
		counts := uniformCounts(a, nil, edges, first, last)
		dot := 0.0
		for _, c := range counts {
			p := c / fn
			dot += p * p
		}
		j := (2 - float64(fn+1)*dot) / hh
		if j < bestJ {
			best, bestJ = nb, j
		}
	}
	return ptp / float64(best)
}

// uniformCounts is numpy.histogram's equal-width branch: the bin of each
// observation from its scaled offset, corrected against the edges, the
// last bin closed; NaNs and observations outside [first, last] ignored.
// Weighted counts are accumulated in blocks of 65536 observations, as
// numpy.bincount does.
func uniformCounts(a, w, edges []float64, first, last float64) []float64 {
	nb := len(edges) - 1
	counts := make([]float64, nb)
	block := make([]float64, nb)
	norm := last - first
	for start := 0; start < len(a); start += 65536 {
		end := min(start+65536, len(a))
		clear(block)
		for i := start; i < end; i++ {
			t := a[i]
			if !(t >= first && t <= last) {
				continue
			}
			k := int((t - first) / norm * float64(nb))
			if k == nb {
				k--
			}
			if t < edges[k] {
				k--
			}
			if k != nb-1 && t >= edges[k+1] {
				k++
			}
			if w == nil {
				block[k]++
			} else {
				block[k] += w[i]
			}
		}
		for k := range counts {
			counts[k] += block[k]
		}
	}
	return counts
}

// edgeCounts is numpy.histogram's explicit-edges branch: cumulative
// (weighted) counts at the edges by binary search in the sorted data, the
// last edge inclusive, in blocks of 65536 observations.
func edgeCounts(a, w, edges []float64) []float64 {
	cum := make([]float64, len(edges))
	for start := 0; start < len(a); start += 65536 {
		end := min(start+65536, len(a))
		idx := make([]int, end-start)
		for i := range idx {
			idx[i] = start + i
		}
		sort.SliceStable(idx, func(i, j int) bool { return lessNaN(a[idx[i]], a[idx[j]]) })
		sa := make([]float64, len(idx))
		cw := make([]float64, len(idx)+1)
		for i, k := range idx {
			sa[i] = a[k]
			if w == nil {
				cw[i+1] = float64(i + 1)
			} else {
				cw[i+1] = cw[i] + w[k]
			}
		}
		for j, e := range edges {
			var pos int
			if j < len(edges)-1 {
				pos = sort.Search(len(sa), func(i int) bool { return !lessNaN(sa[i], e) })
			} else {
				pos = sort.Search(len(sa), func(i int) bool { return lessNaN(e, sa[i]) })
			}
			cum[j] += cw[pos]
		}
	}
	counts := make([]float64, len(edges)-1)
	for i := range counts {
		counts[i] = cum[i+1] - cum[i]
	}
	return counts
}

// lessNaN orders NaN after every number, as NumPy's sort does.
func lessNaN(a, b float64) bool { return a < b || (b != b && a == a) }

// HistogramBinEdges returns the bin edges numpy.histogram_bin_edges(x,
// bins, range, weights) computes, to the bit.
func HistogramBinEdges[D Data](x D, o *HistogramOptions) ([]float64, error) {
	if o == nil {
		o = &HistogramOptions{}
	}
	v := floats(x)
	if o.Weights != nil && len(o.Weights) != len(v) {
		return nil, fmt.Errorf("%w: Histogram: %d weights for %d observations", ErrData, len(o.Weights), len(v))
	}
	e, _, _, _, err := binEdges(v, o)
	if err != nil {
		return nil, err
	}
	return slices.Clone(e), nil
}

// Histogram returns the (weighted) counts of the data in bins and the bin
// edges: numpy.histogram(x, bins, range, density, weights), to the bit.
// Each bin is half-open [a, b) but the last, which is closed.
func Histogram[D Data](x D, o *HistogramOptions) (HistogramResult, error) {
	if o == nil {
		o = &HistogramOptions{}
	}
	v := floats(x)
	edges, err := HistogramBinEdges(x, o)
	if err != nil {
		return HistogramResult{}, err
	}
	var counts []float64
	if o.Edges == nil {
		first, last := edges[0], edges[len(edges)-1]
		if o.Range != nil {
			first, last = o.Range[0], o.Range[1]
			if first == last {
				first, last = first-0.5, last+0.5
			}
		}
		counts = uniformCounts(v, o.Weights, edges, first, last)
	} else {
		counts = edgeCounts(v, o.Weights, edges)
	}
	if o.Density {
		total := sum(counts)
		if o.Weights == nil {
			total = 0
			for _, c := range counts {
				total += c
			}
		}
		for i := range counts {
			counts[i] = counts[i] / (edges[i+1] - edges[i]) / total
		}
	}
	return HistogramResult{Counts: counts, Edges: edges}, nil
}

// FreqOptions are the options of CumFreq and RelFreq.
type FreqOptions struct {
	// NumBins is the number of bins (0 means 10).
	NumBins int
	// Limits, when not nil, are the lower and upper limits of the bins
	// (SciPy's defaultreallimits=); by default the data's range widened by
	// half a bin at each end.
	Limits *[2]float64
	// Weights, when not nil, weight the observations.
	Weights []float64
}

// FreqResult is the result of scipy.stats.cumfreq and relfreq.
type FreqResult struct {
	// Freq are the cumulative counts (CumFreq) or relative frequencies
	// (RelFreq) of the bins.
	Freq []float64
	// LowerLimit is the lower limit of the first bin.
	LowerLimit float64
	// BinSize is the width of the bins.
	BinSize float64
	// ExtraPoints counts the observations outside the bins.
	ExtraPoints int
}

// CumFreq returns the cumulative frequency histogram:
// scipy.stats.cumfreq(x, numbins, defaultreallimits, weights).
func CumFreq[D Data](x D, o *FreqOptions) (FreqResult, error) {
	return scipyHistogram(floats(x), o, false, true, "CumFreq")
}

// RelFreq returns the relative frequency histogram:
// scipy.stats.relfreq(x, numbins, defaultreallimits, weights).
func RelFreq[D Data](x D, o *FreqOptions) (FreqResult, error) {
	return scipyHistogram(floats(x), o, true, false, "RelFreq")
}

// scipyHistogram is scipy.stats._stats_py._histogram.
func scipyHistogram(a []float64, o *FreqOptions, density, cumulative bool, name string) (FreqResult, error) {
	if o == nil {
		o = &FreqOptions{}
	}
	nb := o.NumBins
	if nb == 0 {
		nb = 10
	}
	if nb < 0 {
		return FreqResult{}, fmt.Errorf("%w: %s: %d bins", ErrParam, name, nb)
	}
	w := o.Weights
	if w != nil && len(w) != len(a) {
		return FreqResult{}, fmt.Errorf("%w: %s: %d weights for %d observations", ErrData, name, len(w), len(a))
	}
	if hasNaN(a) || hasNaN(w) {
		return FreqResult{}, fmt.Errorf("%w: %s of data with NaN", ErrData, name)
	}
	for _, t := range w {
		if t < 0 {
			return FreqResult{}, fmt.Errorf("%w: %s: negative weight %v", ErrData, name, t)
		}
	}
	var lo, hi float64
	switch {
	case o.Limits != nil:
		lo, hi = o.Limits[0], o.Limits[1]
		if !(lo < hi) {
			return FreqResult{}, fmt.Errorf("%w: %s: limits %v are not increasing", ErrParam, name, *o.Limits)
		}
	case len(a) == 0:
		lo, hi = 0, 1
	default:
		mn, mx := slices.Min(a), slices.Max(a)
		s := (mx - mn) / (2. * (float64(nb) - 1.))
		lo, hi = mn-s, mx+s
	}
	edges := linspace(lo, hi, nb+1)
	n := len(a)
	idx := make([]int, n)
	for i := range idx {
		idx[i] = i
	}
	sort.SliceStable(idx, func(i, j int) bool { return a[idx[i]] < a[idx[j]] })
	sa := make([]float64, n)
	cw := make([]float64, n+1)
	for i, k := range idx {
		sa[i] = a[k]
		if w == nil {
			cw[i+1] = float64(i + 1)
		} else {
			cw[i+1] = cw[i] + w[k]
		}
	}
	pos := make([]int, len(edges))
	for j, e := range edges {
		pos[j] = sort.SearchFloat64s(sa, e)
	}
	hist := make([]float64, nb)
	for i := range hist {
		hist[i] = cw[pos[i+1]] - cw[pos[i]]
	}
	top := edges[nb]
	if w == nil {
		for _, t := range sa {
			if t == top {
				hist[nb-1]++
			}
		}
	} else {
		at := make([]float64, n)
		for i, t := range sa {
			if t == top {
				at[i] = w[idx[i]]
			}
		}
		hist[nb-1] += sum(at)
	}
	binned := 0
	if w == nil {
		for _, h := range hist {
			binned += int(h)
		}
	} else {
		for _, t := range a {
			if edges[0] <= t && t <= top {
				binned++
			}
		}
	}
	r := FreqResult{LowerLimit: lo, BinSize: edges[1] - edges[0], ExtraPoints: n - binned}
	if density {
		d := float64(n)
		if w != nil {
			d = cw[n]
		}
		for i := range hist {
			hist[i] /= d
		}
	}
	if cumulative {
		for i := 1; i < nb; i++ {
			hist[i] += hist[i-1]
		}
	}
	r.Freq = hist
	return r, nil
}
