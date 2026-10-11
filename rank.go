package stats

import (
	"fmt"
	"slices"
)

// RankMethod is how RankData ranks tied values: SciPy's rankdata method=
// argument. The zero value is RankAverage, SciPy's default.
type RankMethod int

const (
	// RankAverage gives each tied value the average of the ranks the group
	// spans (method="average").
	RankAverage RankMethod = iota
	// RankMin gives each tied value the lowest rank of its group
	// (method="min", "competition" ranking).
	RankMin
	// RankMax gives each tied value the highest rank of its group
	// (method="max").
	RankMax
	// RankDense ranks the distinct values 1, 2, 3, ... (method="dense").
	RankDense
	// RankOrdinal gives distinct ranks in order of appearance among ties
	// (method="ordinal").
	RankOrdinal
)

// RankData assigns ranks 1..n to the data, ties handled by method:
// scipy.stats.rankdata(x, method) (axis=None, nan_policy="propagate": an
// array is ranked in row-major order as a whole, and when any value is
// NaN every rank is NaN). The result has the form of x (a []float64, or an
// array of x's shape).
func RankData[D Data](x D, method RankMethod) (D, error) {
	if method < RankAverage || method > RankOrdinal {
		var zero D
		return zero, fmt.Errorf("%w: rank method %d", ErrParam, int(method))
	}
	v := floats(x)
	r, _ := rankWithTies(v, method, false)
	if hasNaN(v) {
		for i := range r {
			r[i] = nan
		}
	}
	return like(x, r), nil
}

// rankPair is a value and its position, the unit of the sort behind the
// ranks.
type rankPair struct {
	v float64
	i int
}

// sortedPairs returns the values with their positions sorted ascending,
// ties (and NaNs, which go last as in np.argsort) in order of position:
// the stable argsort SciPy's _rankdata uses.
func sortedPairs(x []float64) []rankPair {
	p := make([]rankPair, len(x))
	lo, hi := 0, len(x)
	// NaNs go to the end, in order of position; the rest is sorted with a
	// total order (value, then position), which makes the unstable sort
	// stable.
	nNaN := 0
	for _, v := range x {
		if v != v {
			nNaN++
		}
	}
	hi = len(x) - nNaN
	k := hi
	for i, v := range x {
		if v != v {
			p[k] = rankPair{v, i}
			k++
		} else {
			p[lo] = rankPair{v, i}
			lo++
		}
	}
	slices.SortFunc(p[:hi], func(a, b rankPair) int {
		switch {
		case a.v < b.v:
			return -1
		case a.v > b.v:
			return 1
		}
		return a.i - b.i
	})
	return p
}

// rankWithTies is SciPy's _rankdata(x, method, return_ties=True): the
// ranks in the order of x and, when ties is true, the tie counts in sorted
// order (t[k] is the size of the group starting at sorted position k, 0
// inside a group). NaNs sort last and each is its own group (NaN != NaN),
// as in SciPy.
func rankWithTies(x []float64, method RankMethod, ties bool) (ranks, t []float64) {
	n := len(x)
	p := sortedPairs(x)
	ranks = make([]float64, n)
	if ties {
		t = make([]float64, n)
	}
	if method == RankOrdinal {
		for k, q := range p {
			ranks[q.i] = float64(k + 1)
		}
		return ranks, t
	}
	dense := 0.0
	for start := 0; start < n; {
		end := start + 1
		for end < n && p[end].v == p[start].v {
			end++
		}
		cnt := float64(end - start)
		dense++
		var r float64
		switch method {
		case RankMin:
			r = float64(start + 1)
		case RankMax:
			r = float64(start+1) + cnt - 1
		case RankDense:
			r = dense
		default:
			r = float64(start+1) + (cnt-1)/2
		}
		for k := start; k < end; k++ {
			ranks[p[k].i] = r
		}
		if ties {
			t[start] = cnt
		}
		start = end
	}
	return ranks, t
}

// tieTerm is Σ (t³ - t) over the tie counts.
func tieTerm(t []float64) float64 {
	d := make([]float64, len(t))
	for i, c := range t {
		d[i] = c*c*c - c
	}
	return sum(d)
}
