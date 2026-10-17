package stats

import (
	"fmt"
	"math"

	"github.com/go-ndarray/ndarray"
	"github.com/go-ndarray/ndarray/random"
	"github.com/go-numerics/special"
	"github.com/go-statistics/stats/internal/dd"
)

// BootstrapMethod is the kind of bootstrap confidence interval. The zero
// value is BootstrapBCa, SciPy's default.
type BootstrapMethod int

const (
	// BootstrapBCa is the bias-corrected and accelerated interval
	// (method="BCa").
	BootstrapBCa BootstrapMethod = iota
	// BootstrapPercentile is the percentile interval (method="percentile").
	BootstrapPercentile
	// BootstrapBasic is the basic (reverse percentile) interval
	// (method="basic").
	BootstrapBasic
)

// String returns SciPy's name for the method.
func (m BootstrapMethod) String() string {
	switch m {
	case BootstrapBCa:
		return "BCa"
	case BootstrapPercentile:
		return "percentile"
	case BootstrapBasic:
		return "basic"
	}
	return "invalid"
}

// BootstrapOptions configures Bootstrap. The zero value is SciPy's
// default: 9999 resamples, a 95% two-sided BCa interval, independent
// samples, one batch.
type BootstrapOptions struct {
	// NResamples is the number of resamples; 0 is 9999.
	NResamples int
	// ConfidenceLevel is the confidence level, in (0, 1); 0 is 0.95.
	ConfidenceLevel float64
	// Method is the kind of interval.
	Method BootstrapMethod
	// Alternative makes the interval one-sided: Less gives (-∞, high],
	// Greater [low, ∞).
	Alternative Alternative
	// Paired resamples the samples together, with the same indices (they
	// must have the same length).
	Paired bool
	// Batch is the number of resamples drawn per call to the generator
	// (SciPy's batch); 0 draws them all at once. It changes the draws.
	Batch int
}

// BootstrapResult is the result of Bootstrap.
type BootstrapResult struct {
	// ConfidenceInterval is the bootstrap confidence interval.
	ConfidenceInterval ConfidenceInterval
	// BootstrapDistribution holds the statistic of each resample.
	BootstrapDistribution []float64
	// StandardError is the sample standard deviation (ddof 1) of the
	// bootstrap distribution.
	StandardError float64
}

// Bootstrap computes a bootstrap confidence interval for statistic, a
// function of one value per sample, from the given samples:
// scipy.stats.bootstrap(data, statistic, n_resamples, batch, paired,
// confidence_level, alternative, method, rng=g).
//
// The resamples are SciPy's for the same generator: for each batch and
// each sample, g.Integers(0, n, (batch, n)) indices (one index matrix for
// all samples when Paired). With statistic the NumPy-exact mean of this
// package (NumPy's pairwise sum along the last axis, as np.mean(axis=-1)
// computes it), the bootstrap distribution and the interval are SciPy's
// to the bit. A nil g is a generator seeded from the operating system.
//
// statistic is called once per resample (and, for BCa, once per jackknife
// sample) with slices it must not retain or modify. It returns an error
// wrapping ErrData when a sample has fewer than two observations or paired
// samples differ in length, and ErrParam for invalid options. The interval
// is NaN where SciPy's is (a degenerate bootstrap distribution under BCa,
// or a NaN statistic).
func Bootstrap[D Data](samples []D, statistic func(samples ...[]float64) float64, g *random.Generator, o *BootstrapOptions) (BootstrapResult, error) {
	var res BootstrapResult
	if o == nil {
		o = &BootstrapOptions{}
	}
	nres := o.NResamples
	if nres == 0 {
		nres = 9999
	}
	cl := o.ConfidenceLevel
	if cl == 0 {
		cl = 0.95
	}
	switch {
	case len(samples) == 0:
		return res, fmt.Errorf("%w: no samples", ErrData)
	case nres < 0:
		return res, fmt.Errorf("%w: NResamples must be nonnegative", ErrParam)
	case o.Batch < 0:
		return res, fmt.Errorf("%w: Batch must be nonnegative", ErrParam)
	case !(cl > 0 && cl < 1):
		return res, fmt.Errorf("%w: ConfidenceLevel must lie in (0, 1)", ErrParam)
	case o.Method < BootstrapBCa || o.Method > BootstrapBasic:
		return res, fmt.Errorf("%w: unknown bootstrap method %d", ErrParam, o.Method)
	case o.Alternative < TwoSided || o.Alternative > Greater:
		return res, fmt.Errorf("%w: unknown alternative %d", ErrParam, o.Alternative)
	}
	data := make([][]float64, len(samples))
	for i, s := range samples {
		data[i] = floats(s)
		if len(data[i]) <= 1 {
			return res, fmt.Errorf("%w: each sample needs at least two observations", ErrData)
		}
		if o.Paired && len(data[i]) != len(data[0]) {
			return res, fmt.Errorf("%w: paired samples must have the same length", ErrData)
		}
	}
	if g == nil {
		g = random.DefaultRNG()
	}
	b := &bootstrapper{data: data, stat: statistic, paired: o.Paired}
	theta := make([]float64, 0, nres)
	batch := o.Batch
	if batch == 0 {
		batch = nres
	}
	for k := 0; k < nres; k += batch {
		theta = b.resample(g, theta, min(batch, nres-k))
	}
	alpha := 1 - cl
	if o.Alternative == TwoSided {
		alpha = (1 - cl) / 2
	}
	pl, ph := alpha, 1-alpha
	if o.Method == BootstrapBCa {
		pl, ph = b.bcaLevels(theta, alpha)
	}
	sorted := append([]float64(nil), theta...)
	sortFloats(sorted)
	nan := false
	for _, t := range theta {
		nan = nan || math.IsNaN(t)
	}
	lo, hi := quantileLinear(sorted, pl, nan), quantileLinear(sorted, ph, nan)
	if o.Method == BootstrapBasic {
		t := b.full()
		lo, hi = 2*t-hi, 2*t-lo
	}
	switch o.Alternative {
	case Less:
		lo = math.Inf(-1)
	case Greater:
		hi = math.Inf(1)
	}
	res.ConfidenceInterval = ConfidenceInterval{Low: lo, High: hi}
	res.BootstrapDistribution = theta
	res.StandardError = math.Sqrt(variance(theta, 1))
	return res, nil
}

type bootstrapper struct {
	data   [][]float64
	stat   func(samples ...[]float64) float64
	paired bool
}

// full is the statistic of the original samples.
func (b *bootstrapper) full() float64 { return b.stat(b.data...) }

// resample appends the statistics of m resamples, drawing the indices as
// SciPy's _bootstrap_resample does.
func (b *bootstrapper) resample(g *random.Generator, theta []float64, m int) []float64 {
	k := len(b.data)
	draws := 1
	if !b.paired {
		draws = k
	}
	idx := make([][]int64, draws)
	for j := range draws {
		n := len(b.data[j])
		a, _ := g.Integers(0, int64(n), m, n)
		idx[j], _ = ndarray.Data[int64](a)
	}
	bufs := make([][]float64, k)
	for j := range k {
		bufs[j] = make([]float64, len(b.data[j]))
	}
	for r := range m {
		for j := range k {
			n := len(b.data[j])
			ix := idx[0]
			if !b.paired {
				ix = idx[j]
			}
			for i := range n {
				bufs[j][i] = b.data[j][ix[r*n+i]]
			}
		}
		theta = append(theta, b.stat(bufs...))
	}
	return theta
}

// jackknife returns, for each resampled unit (each sample, or the common
// index when paired), the statistics with each observation left out.
func (b *bootstrapper) jackknife() [][]float64 {
	k := len(b.data)
	units := k
	if b.paired {
		units = 1
	}
	out := make([][]float64, units)
	bufs := make([][]float64, k)
	for u := range units {
		n := len(b.data[u])
		for j := range k {
			bufs[j] = b.data[j]
		}
		for i := range n {
			for j := range k {
				if b.paired || j == u {
					bufs[j] = leaveOut(b.data[j], i)
				}
			}
			out[u] = append(out[u], b.stat(bufs...))
		}
	}
	return out
}

func leaveOut(x []float64, i int) []float64 {
	r := make([]float64, 0, len(x)-1)
	r = append(r, x[:i]...)
	return append(r, x[i+1:]...)
}

// bcaLevels returns the BCa quantile levels, SciPy's _bca_interval:
// the bias correction from the share of resampled statistics below the
// full-sample one, the acceleration from the jackknife.
func (b *bootstrapper) bcaLevels(theta []float64, alpha float64) (float64, float64) {
	t := b.full()
	cnt := 0
	for _, v := range theta {
		if v < t {
			cnt++
		}
		if v <= t {
			cnt++
		}
	}
	z0 := special.Ndtri(float64(cnt) / float64(2*len(theta)))
	var num, den float64
	for _, ti := range b.jackknife() {
		n := float64(len(ti))
		dot := mean(ti)
		u3 := make([]float64, len(ti))
		u2 := make([]float64, len(ti))
		for i, v := range ti {
			u := (n - 1) * (dot - v)
			u2[i] = float64(u * u)
			u3[i] = cube(u)
		}
		num += sum(u3) / cube(n)
		den += sum(u2) / float64(n*n)
	}
	// den**1.5 correctly rounded, as the C library's pow gives it.
	a := 1.0 / 6 * num / dd.MulF(dd.Sqrt(dd.F(den)), den).Float()
	za := special.Ndtri(alpha)
	n1 := z0 + za
	n2 := z0 - za
	return special.Ndtr(z0 + n1/(1-float64(a*n1))), special.Ndtr(z0 + n2/(1-float64(a*n2)))
}

// cube is x³ correctly rounded (NumPy's x**3 is the C library's pow).
func cube(x float64) float64 { return dd.MulF(dd.MulFF(x, x), x).Float() }

// quantileLinear is scipy.stats.quantile(y, p) with method="linear" for
// the sorted y: (1-g)·y[j] + g·y[j+1] with j + g = p·(n-1); NaN when p is
// outside [0, 1] or NaN, or when nan is set (a NaN in y propagates).
func quantileLinear(y []float64, p float64, nan bool) float64 {
	if nan || !(p >= 0 && p <= 1) {
		return math.NaN()
	}
	n := float64(len(y))
	jg := float64(p*n) + (1 - p) // in [1, n] for p in [0, 1]
	jp1 := math.Floor(jg)
	j := max(jp1-1, 0) // guards a rounding of jg below 1
	g := jg - jp1
	jp1 = min(jp1, n-1)
	return float64((1-g)*y[int(j)]) + float64(g*y[int(jp1)])
}
