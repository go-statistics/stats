package stats

import (
	"fmt"
	"math"
	"runtime"
	"sort"
	"sync"

	"github.com/go-ndarray/ndarray"
	"github.com/go-ndarray/ndarray/linalg"
	"github.com/go-ndarray/ndarray/random"
	"github.com/go-statistics/stats/internal/dd"
)

// BandwidthRule is a rule of thumb for the bandwidth factor of a
// GaussianKDE. The zero value is Scott's rule, SciPy's default.
type BandwidthRule int

const (
	// BandwidthScott is Scott's rule, neff^(-1/(d+4)) (bw_method="scott").
	BandwidthScott BandwidthRule = iota
	// BandwidthSilverman is Silverman's rule, (neff·(d+2)/4)^(-1/(d+4))
	// (bw_method="silverman").
	BandwidthSilverman
)

// KDEBandwidth selects the bandwidth factor of a GaussianKDE, SciPy's
// bw_method: Func if it is set (a callable), else Factor if it is positive
// (a scalar), else Rule. The zero value is Scott's rule.
type KDEBandwidth struct {
	Rule   BandwidthRule
	Factor float64
	Func   func(k *GaussianKDE) float64
}

// GaussianKDEOptions configures NewGaussianKDE. The zero value is SciPy's
// default: Scott's rule and equal weights.
type GaussianKDEOptions struct {
	Bandwidth KDEBandwidth
	// Weights are the weights of the data points (n of them, nonnegative,
	// not all zero); nil is equal weights. They are normalized to sum to 1.
	Weights []float64
}

// GaussianKDE is a kernel density estimate with Gaussian kernels:
// scipy.stats.gaussian_kde(dataset, bw_method, weights).
//
// The kernel covariance is the weighted data covariance times factor².
// Evaluation whitens the data and the points by the Cholesky factor of
// the kernel covariance, as SciPy does, but sums each estimate with
// compensated (Neumaier) summation, relative to its largest term, so the
// density and its logarithm are accurate to about one ULP of the computed
// kernel values and the logarithm stays finite far from the data. In one
// dimension the data are sorted once and each estimate sums only the
// points whose kernel exceeds 2^-60 of the nearest point's (the omitted
// terms together are below 2^-60 of the result). Points are evaluated in
// parallel; each result is computed by one goroutine in a fixed order, so
// results do not depend on GOMAXPROCS.
type GaussianKDE struct {
	d, n    int
	data    []float64 // d×n, as given (one row per dimension)
	weights []float64
	neff    float64
	bw      KDEBandwidth

	dataCov, dataChol []float64

	factor   float64
	cov, cho []float64
	logDet   float64

	// Evaluation: the points with positive weight, whitened, point-major
	// (n'×d), with log-weights, sorted in one dimension.
	white []float64
	ew    []float64
	elw   []float64
	wmax  float64
}

// NewGaussianKDE returns the kernel density estimate of dataset: a
// []float64 (one-dimensional data), a 1-D array, or a d×n array with one
// row per dimension and one column per point (SciPy's layout). It returns
// an error wrapping ErrData when there are fewer than two values, more
// dimensions than points, or the data lie in a lower-dimensional subspace
// (their covariance is singular), and ErrParam for invalid weights or
// bandwidth.
func NewGaussianKDE[D Data](dataset D, o *GaussianKDEOptions) (*GaussianKDE, error) {
	k := &GaussianKDE{}
	if a, ok := any(dataset).(*ndarray.Array); ok && a != nil && len(a.Shape()) == 2 {
		k.d, k.n = a.Shape()[0], a.Shape()[1]
		k.data = append([]float64(nil), arrayFloats(a)...)
	} else {
		if ok && a != nil && len(a.Shape()) > 2 {
			return nil, fmt.Errorf("%w: the dataset must have at most two axes", ErrData)
		}
		k.data = append([]float64(nil), floats(dataset)...)
		k.d, k.n = 1, len(k.data)
	}
	if k.d*k.n <= 1 {
		return nil, fmt.Errorf("%w: the dataset needs more than one element", ErrData)
	}
	if o == nil {
		o = &GaussianKDEOptions{}
	}
	if o.Weights != nil {
		if len(o.Weights) != k.n {
			return nil, fmt.Errorf("%w: %d weights for %d points", ErrParam, len(o.Weights), k.n)
		}
		s := sum(o.Weights)
		for _, w := range o.Weights {
			if !(w >= 0) || math.IsInf(w, 1) {
				return nil, fmt.Errorf("%w: weights must be nonnegative and finite", ErrParam)
			}
		}
		if !(s > 0) {
			return nil, fmt.Errorf("%w: the weights sum to zero", ErrParam)
		}
		k.weights = make([]float64, k.n)
		for i, w := range o.Weights {
			k.weights[i] = w / s
		}
	} else {
		k.weights = make([]float64, k.n)
		for i := range k.weights {
			k.weights[i] = 1 / float64(k.n)
		}
	}
	// neff = 1/Σw², correctly rounded: SciPy's int(neff) (the default
	// resample size) must not drop to 999 for 1000 equal weights.
	var sq dd.DD
	for _, w := range k.weights {
		sq = dd.Add(sq, dd.MulFF(w, w))
	}
	k.neff = dd.Div(dd.F(1), sq).Float()
	if k.d > k.n {
		return nil, fmt.Errorf("%w: %d dimensions but only %d points (gaussian_kde reads each column as a point)", ErrData, k.d, k.n)
	}
	k.dataCov = weightedCov(k.data, k.weights, k.d, k.n)
	var ok bool
	if k.dataChol, ok = cholesky(k.dataCov, k.d); !ok {
		return nil, fmt.Errorf("%w: the data lie in a lower-dimensional subspace (their covariance is singular)", ErrData)
	}
	if err := k.SetBandwidth(o.Bandwidth); err != nil {
		return nil, err
	}
	return k, nil
}

// weightedCov is np.cov(x, rowvar=True, bias=False, aweights=w) for the
// d×n x and weights w summing to 1.
func weightedCov(x, w []float64, d, n int) []float64 {
	ws := sum(w)
	ww := make([]float64, n)
	for i := range n {
		ww[i] = float64(w[i] * w[i])
	}
	fact := ws - sum(ww)/ws
	xc := make([]float64, d*n)
	t := make([]float64, n)
	for a := range d {
		row := x[a*n : (a+1)*n]
		for i := range n {
			t[i] = float64(row[i] * w[i])
		}
		avg := sum(t) / ws
		for i := range n {
			xc[a*n+i] = row[i] - avg
		}
	}
	c := make([]float64, d*d)
	inv := 1 / fact
	for a := range d {
		for b := range d {
			for i := range n {
				t[i] = float64(xc[a*n+i] * float64(xc[b*n+i]*w[i]))
			}
			c[a*d+b] = float64(sum(t) * inv)
		}
	}
	return c
}

// SetBandwidth changes the bandwidth factor and recomputes the kernel
// covariance: scipy.stats.gaussian_kde.set_bandwidth. It returns an error
// wrapping ErrParam if the factor is not positive and finite.
func (k *GaussianKDE) SetBandwidth(b KDEBandwidth) error {
	var f float64
	switch {
	case b.Func != nil:
		f = b.Func(k)
	case b.Factor != 0:
		f = b.Factor
	case b.Rule == BandwidthScott:
		f = k.ScottsFactor()
	case b.Rule == BandwidthSilverman:
		f = k.SilvermanFactor()
	default:
		return fmt.Errorf("%w: unknown bandwidth rule %d", ErrParam, b.Rule)
	}
	if !(f > 0) || math.IsInf(f, 1) {
		return fmt.Errorf("%w: the bandwidth factor must be positive and finite, it is %v", ErrParam, f)
	}
	k.bw, k.factor = b, f
	d := k.d
	k.cov = make([]float64, d*d)
	k.cho = make([]float64, d*d)
	f2 := f * f
	for i := range k.cov {
		k.cov[i] = k.dataCov[i] * f2
		k.cho[i] = k.dataChol[i] * f
	}
	logs := make([]float64, d)
	for i := range d {
		logs[i] = math.Log(k.cho[i*d+i] * math.Sqrt(2*math.Pi))
	}
	k.logDet = 2 * sum(logs)
	k.prepare()
	return nil
}

// prepare whitens the points of positive weight for evaluation (sorted by
// coordinate in one dimension).
func (k *GaussianKDE) prepare() {
	d := k.d
	var pts, w []float64
	for i, wi := range k.weights {
		if wi > 0 {
			for a := range d {
				pts = append(pts, k.data[a*k.n+i])
			}
			w = append(w, wi)
		}
	}
	z := k.whiten(pts)
	ord := make([]int, len(w))
	for j := range ord {
		ord[j] = j
	}
	if d == 1 {
		sort.SliceStable(ord, func(a, b int) bool { return z[ord[a]] < z[ord[b]] })
	}
	k.white = make([]float64, len(z))
	k.ew = make([]float64, len(w))
	k.elw = make([]float64, len(w))
	k.wmax = 0
	for j, o := range ord {
		copy(k.white[j*d:(j+1)*d], z[o*d:(o+1)*d])
		k.ew[j] = w[o]
		k.elw[j] = math.Log(w[o])
		k.wmax = math.Max(k.wmax, w[o])
	}
}

// whiten solves L·z = x for each point (point-major rows of d) with the
// kernel's Cholesky factor L.
func (k *GaussianKDE) whiten(pts []float64) []float64 {
	d := k.d
	z := make([]float64, len(pts))
	for p := 0; p < len(pts); p += d {
		for i := range d {
			t := pts[p+i]
			for j := range i {
				t -= float64(k.cho[i*d+j] * z[p+j])
			}
			z[p+i] = t / k.cho[i*d+i]
		}
	}
	return z
}

// D returns the dimension.
func (k *GaussianKDE) D() int { return k.d }

// N returns the number of data points.
func (k *GaussianKDE) N() int { return k.n }

// Dataset returns the data, d×n (a copy).
func (k *GaussianKDE) Dataset() *ndarray.Array { return mustArray(k.data, k.d, k.n) }

// Weights returns the normalized weights (a copy).
func (k *GaussianKDE) Weights() []float64 { return append([]float64(nil), k.weights...) }

// Neff returns the effective number of points, 1/Σw².
func (k *GaussianKDE) Neff() float64 { return k.neff }

// Factor returns the bandwidth factor.
func (k *GaussianKDE) Factor() float64 { return k.factor }

// Covariance returns the kernel covariance, the data covariance times
// factor², d×d.
func (k *GaussianKDE) Covariance() *ndarray.Array { return mustArray(k.cov, k.d, k.d) }

// ChoCov returns the lower Cholesky factor of the kernel covariance
// (SciPy's cho_cov), d×d.
func (k *GaussianKDE) ChoCov() *ndarray.Array { return mustArray(k.cho, k.d, k.d) }

// LogDet returns SciPy's log_det, 2·Σ log(Lᵢᵢ·√(2π)) = log det(2πΣ).
func (k *GaussianKDE) LogDet() float64 { return k.logDet }

// InvCov returns the inverse of the kernel covariance, d×d (SciPy's
// inv_cov).
func (k *GaussianKDE) InvCov() *ndarray.Array {
	inv, _ := linalg.Inv(mustArray(k.dataCov, k.d, k.d))
	v, _ := ndarray.Data[float64](inv)
	f2 := k.factor * k.factor
	out := make([]float64, len(v))
	for i := range v {
		out[i] = v[i] / f2
	}
	return mustArray(out, k.d, k.d)
}

// ScottsFactor returns neff^(-1/(d+4)).
func (k *GaussianKDE) ScottsFactor() float64 {
	return math.Pow(k.neff, -1/float64(k.d+4))
}

// SilvermanFactor returns (neff·(d+2)/4)^(-1/(d+4)).
func (k *GaussianKDE) SilvermanFactor() float64 {
	return math.Pow(k.neff*(float64(k.d)+2)/4, -1/float64(k.d+4))
}

// queryPoints reads points as SciPy's evaluate does: an array of one row
// per dimension and one column per point (a 1-D array is m points in one
// dimension, or one point of d coordinates). It returns them point-major.
func (k *GaussianKDE) queryPoints(points *ndarray.Array) ([]float64, int, error) {
	if points == nil {
		return nil, 0, fmt.Errorf("%w: no points", ErrData)
	}
	v := arrayFloats(points)
	s := points.Shape()
	var r, m int
	switch len(s) {
	case 0:
		r, m = 1, 1
	case 1:
		r, m = 1, s[0]
	case 2:
		r, m = s[0], s[1]
	default:
		return nil, 0, fmt.Errorf("%w: points must have at most two axes", ErrData)
	}
	if r != k.d {
		if r == 1 && m == k.d {
			return append([]float64(nil), v...), 1, nil
		}
		return nil, 0, fmt.Errorf("%w: points have dimension %d, the dataset %d", ErrData, r, k.d)
	}
	out := make([]float64, len(v))
	for a := range r {
		for j := range m {
			out[j*r+a] = v[a*m+j]
		}
	}
	return out, m, nil
}

// Evaluate returns the estimated density at each point:
// scipy.stats.gaussian_kde.evaluate (also pdf and __call__). Points are
// read as SciPy reads them: a d×m array, one column per point, or a 1-D
// array (m points when d is 1, else one point).
func (k *GaussianKDE) Evaluate(points *ndarray.Array) ([]float64, error) {
	return k.eval(points, false)
}

// LogPDF returns the logarithm of the estimated density at each point
// (see Evaluate): scipy.stats.gaussian_kde.logpdf. It is computed relative
// to the largest kernel term, so it is finite and accurate far from the
// data, where the density underflows.
func (k *GaussianKDE) LogPDF(points *ndarray.Array) ([]float64, error) {
	return k.eval(points, true)
}

// EvaluateSlice is Evaluate for points given as a []float64, read as a 1-D
// array.
func (k *GaussianKDE) EvaluateSlice(x []float64) ([]float64, error) {
	return k.eval(mustArray(x, len(x)), false)
}

// LogPDFSlice is LogPDF for points given as a []float64, read as a 1-D
// array.
func (k *GaussianKDE) LogPDFSlice(x []float64) ([]float64, error) {
	return k.eval(mustArray(x, len(x)), true)
}

func (k *GaussianKDE) eval(points *ndarray.Array, logp bool) ([]float64, error) {
	pts, m, err := k.queryPoints(points)
	if err != nil {
		return nil, err
	}
	z := k.whiten(pts)
	d := k.d
	logNorm := -float64(d)/2*math.Log(2*math.Pi) - sumLogDiag(k.cho, d)
	out := make([]float64, m)
	parallelChunks(m, func(lo, hi int) {
		var buf []float64
		if d > 1 {
			buf = make([]float64, len(k.ew))
		}
		for j := lo; j < hi; j++ {
			var a0, s float64
			if d == 1 {
				a0, s = k.sum1D(z[j])
			} else {
				a0, s = k.sumND(z[j*d:(j+1)*d], buf)
			}
			if logp {
				out[j] = math.Log(s) - a0 + logNorm
			} else {
				out[j] = math.Exp(-a0+logNorm) * s
			}
		}
	})
	return out, nil
}

func sumLogDiag(l []float64, d int) float64 {
	s := 0.0
	for i := range d {
		s += math.Log(l[i*d+i])
	}
	return s
}

// tailCut is 60·log 2: kernel terms below 2^-60 of the reference term
// (scaled by n·wmax/w) are omitted in one dimension.
const tailCut = 60 * math.Ln2

// sum1D returns a0 and s with Σ wᵢ·exp(-(zᵢ-t)²/2) = exp(-a0)·s, over the
// sorted whitened one-dimensional data, summing only the window of points
// that can matter (see GaussianKDE).
func (k *GaussianKDE) sum1D(t float64) (a0, s float64) {
	z := k.white
	n := len(z)
	if math.IsNaN(t) {
		return math.NaN(), math.NaN()
	}
	j := sort.SearchFloat64s(z, t)
	near := j
	if j == n || j > 0 && t-z[j-1] < z[j]-t {
		near = j - 1
	}
	dn := z[near] - t
	a0 = dn * dn / 2
	if math.IsInf(t, 0) {
		return math.Inf(1), 0
	}
	// Terms with a > a0 + cut contribute together less than 2^-60 of
	// w_near·exp(-a0) ≤ the sum.
	cut := a0 + tailCut + math.Log(float64(n)*k.wmax/k.ew[near])
	r := math.Sqrt(2 * cut)
	lo := sort.SearchFloat64s(z, t-r)
	hi := sort.SearchFloat64s(z, math.Nextafter(t+r, math.Inf(1)))
	// Kahan summation suffices: every term is positive.
	var c float64
	w := k.ew[lo:hi]
	for i, zi := range z[lo:hi] {
		u := zi - t
		y := w[i]*math.Exp(a0-u*u/2) - c
		tt := s + y
		c = (tt - s) - y
		s = tt
	}
	return a0, s
}

// sumND is sum1D for d > 1, over every point, relative to the largest
// weighted term.
func (k *GaussianKDE) sumND(t, buf []float64) (a0, s float64) {
	d := k.d
	mx := math.Inf(-1)
	nan := false
	for i := range k.ew {
		a := 0.0
		p := k.white[i*d : (i+1)*d]
		for c := range d {
			u := p[c] - t[c]
			a += float64(u * u)
		}
		b := k.elw[i] - a/2
		buf[i] = b
		// mx = math.Max(mx, b), without its call: NaN sticks, +0 beats -0
		switch {
		case b > mx:
			mx = b
		case b != b:
			nan = true
		case b == 0 && mx == 0 && !math.Signbit(b):
			mx = b
		}
	}
	if nan {
		mx = math.NaN()
	}
	if math.IsInf(mx, 0) || math.IsNaN(mx) {
		return -mx, 1
	}
	var c float64
	for _, b := range buf {
		s, c = neumaier(s, c, math.Exp(b-mx))
	}
	return -mx, s + c
}

// neumaier adds v to the compensated sum (s, c).
func neumaier(s, c, v float64) (float64, float64) {
	t := s + v
	if math.Abs(s) >= math.Abs(v) {
		c += (s - t) + v
	} else {
		c += (v - t) + s
	}
	return t, c
}

// parallelChunks runs f over [0, m) split into contiguous chunks, one
// goroutine per chunk.
func parallelChunks(m int, f func(lo, hi int)) {
	p := runtime.GOMAXPROCS(0)
	chunk := max((m+p-1)/p, 64)
	var wg sync.WaitGroup
	for lo := 0; lo < m; lo += chunk {
		hi := min(lo+chunk, m)
		wg.Add(1)
		go func() {
			defer wg.Done()
			f(lo, hi)
		}()
	}
	wg.Wait()
}

// IntegrateBox1D returns the integral of the one-dimensional estimate over
// [low, high]: scipy.stats.gaussian_kde.integrate_box_1d. Each kernel's
// mass is a difference of normal cdfs, or of survival functions above its
// centre, so that tail intervals keep their relative accuracy.
func (k *GaussianKDE) IntegrateBox1D(low, high float64) (float64, error) {
	if k.d != 1 {
		return math.NaN(), fmt.Errorf("%w: IntegrateBox1D needs one-dimensional data", ErrData)
	}
	sd := math.Sqrt(k.cov[0])
	t := make([]float64, k.n)
	for i, x := range k.data {
		a, b := (low-x)/sd, (high-x)/sd
		if a <= b {
			t[i] = k.weights[i] * normalInterval(a, b)
		} else {
			t[i] = -k.weights[i] * normalInterval(b, a)
		}
	}
	return sum(t), nil
}

// IntegrateGaussian returns the integral of the product of the estimate
// and the normal density with the given mean and covariance (d×d):
// scipy.stats.gaussian_kde.integrate_gaussian.
func (k *GaussianKDE) IntegrateGaussian(mean []float64, cov *ndarray.Array) (float64, error) {
	if len(mean) != k.d {
		return math.NaN(), fmt.Errorf("%w: mean has dimension %d, the estimate %d", ErrParam, len(mean), k.d)
	}
	c, err := matrixArg(cov, k.d, "cov")
	if err != nil {
		return math.NaN(), err
	}
	for i := range c {
		c[i] += k.cov[i]
	}
	return gaussOverlap(k.data, k.weights, k.d, k.n, c, func(i int) []float64 { return mean })
}

// gaussOverlap is Σᵢ wᵢ·N(xᵢ; μ(i), C) over the columns of the d×n data,
// with C = l·lᵀ positive definite.
func gaussOverlap(data, w []float64, d, n int, c []float64, mu func(i int) []float64) (float64, error) {
	l, ok := cholesky(c, d)
	if !ok {
		return math.NaN(), fmt.Errorf("%w: the summed covariance is not positive definite", ErrParam)
	}
	sqrtDet := 1.0
	for i := range d {
		sqrtDet *= l[i*d+i]
	}
	norm := math.Pow(2*math.Pi, float64(d)/2) * sqrtDet
	t := make([]float64, n)
	z := make([]float64, d)
	for i := range n {
		m := mu(i)
		e := 0.0
		for a := range d {
			v := data[a*n+i] - m[a]
			for b := range a {
				v -= float64(l[a*d+b] * z[b])
			}
			z[a] = v / l[a*d+a]
			e += float64(z[a] * z[a])
		}
		t[i] = float64(math.Exp(-e/2) * w[i])
	}
	return sum(t) / norm, nil
}

// IntegrateKDE returns the integral of the product of two estimates of the
// same dimension: scipy.stats.gaussian_kde.integrate_kde.
func (k *GaussianKDE) IntegrateKDE(other *GaussianKDE) (float64, error) {
	if other.d != k.d {
		return math.NaN(), fmt.Errorf("%w: the estimates have dimensions %d and %d", ErrParam, k.d, other.d)
	}
	small, large := k, other
	if other.n < k.n {
		small, large = other, k
	}
	c := make([]float64, k.d*k.d)
	for i := range c {
		c[i] = small.cov[i] + large.cov[i]
	}
	t := make([]float64, small.n)
	pt := make([]float64, k.d)
	for i := range small.n {
		for a := range k.d {
			pt[a] = small.data[a*small.n+i]
		}
		v, _ := gaussOverlap(large.data, large.weights, k.d, large.n, c, func(int) []float64 { return pt })
		t[i] = v * small.weights[i]
	}
	return sum(t), nil
}

// IntegrateBox returns the integral of the estimate over the box
// [low, high]: scipy.stats.gaussian_kde.integrate_box, with the
// multivariate normal cdf of each kernel (MultivariateNormal.CDF, options
// as there; one generator serves every kernel in turn, as in SciPy).
func (k *GaussianKDE) IntegrateBox(low, high []float64, o *MultivariateNormalCDFOptions) (float64, error) {
	if len(low) != k.d || len(high) != k.d {
		return math.NaN(), fmt.Errorf("%w: the box must have dimension %d", ErrParam, k.d)
	}
	mvn, _ := NewMultivariateNormal(nil, mustArray(k.cov, k.d, k.d), nil)
	oo := MultivariateNormalCDFOptions{}
	if o != nil {
		oo = *o
	}
	if oo.RNG == nil {
		oo.RNG = random.DefaultRNG(0)
	}
	t := make([]float64, k.n)
	x := make([]float64, k.d)
	lo := make([]float64, k.d)
	oo.Lower = lo
	for i := range k.n {
		for a := range k.d {
			p := k.data[a*k.n+i]
			x[a], lo[a] = high[a]-p, low[a]-p
		}
		v, _ := mvn.CDF(x, &oo)
		t[i] = float64(v * k.weights[i])
	}
	return sum(t), nil
}

// Resample draws size points from the estimate, a d×size array, as
// scipy.stats.gaussian_kde.resample(size, seed=g) does: first
// Generator.multivariate_normal(zeros(d), covariance, size), then
// Generator.choice(n, size, p=weights), and the chosen data points plus
// the normal draws. size 0 is int(neff), SciPy's default. The draws are
// SciPy's bit for bit in one dimension; in more, the normal parts agree to
// a few ULP (see MultivariateNormal.Rvs).
func (k *GaussianKDE) Resample(g *random.Generator, size int) (*ndarray.Array, error) {
	if size == 0 {
		size = int(k.neff)
	}
	if size < 0 {
		return nil, fmt.Errorf("%w: size must be nonnegative", ErrParam)
	}
	norm, _ := npMultivariateNormal(g, make([]float64, k.d), k.cov, k.d, size)
	ia, err := g.ChoiceN(int64(k.n), random.ChoiceOptions{P: k.weights}, size)
	if err != nil {
		return nil, fmt.Errorf("%w: %v", ErrParam, err)
	}
	idx, _ := ndarray.Data[int64](ia)
	out := make([]float64, k.d*size)
	for a := range k.d {
		for j := range size {
			out[a*size+j] = k.data[a*k.n+int(idx[j])] + norm[j*k.d+a]
		}
	}
	return mustArray(out, k.d, size), nil
}

// Marginal returns the estimate of the marginal distribution of the given
// dimensions (indices, negative ones counting from the end), with the same
// bandwidth factor and weights: scipy.stats.gaussian_kde.marginal.
func (k *GaussianKDE) Marginal(dims ...int) (*GaussianKDE, error) {
	seen := map[int]bool{}
	rows := make([]float64, 0, len(dims)*k.n)
	for _, d := range dims {
		if d < 0 {
			d += k.d
		}
		if d < 0 || d >= k.d || seen[d] {
			return nil, fmt.Errorf("%w: invalid or repeated dimension in %v", ErrParam, dims)
		}
		seen[d] = true
		rows = append(rows, k.data[d*k.n:(d+1)*k.n]...)
	}
	return NewGaussianKDE(mustArray(rows, len(dims), k.n), &GaussianKDEOptions{
		Bandwidth: KDEBandwidth{Factor: k.factor},
		Weights:   k.weights,
	})
}
