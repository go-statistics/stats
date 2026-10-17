package stats

import (
	"fmt"
	"math"

	"github.com/go-ndarray/ndarray"
	"github.com/go-ndarray/ndarray/random"
	"github.com/go-statistics/stats/internal/dd"
)

// MultivariateNormalOptions configures NewMultivariateNormal. The zero
// value is SciPy's default.
type MultivariateNormalOptions struct {
	// AllowSingular accepts a positive semidefinite covariance that is
	// singular (SciPy's allow_singular=True): the density is then taken on
	// the support, the affine subspace mean + range(cov), with the
	// pseudo-determinant.
	AllowSingular bool
}

// MultivariateNormal is the multivariate normal distribution with mean
// vector Mean and covariance matrix Cov: scipy.stats.multivariate_normal(
// mean, cov, allow_singular).
//
// The covariance is examined as SciPy's _PSD does, with the symmetric
// eigendecomposition of its lower triangle: an eigenvalue below
// -1e6·ε·max|λ| makes it invalid, one of magnitude at most 1e6·ε·max|λ|
// makes it singular. A nonsingular covariance is then factored again by a
// Cholesky decomposition carried in double-double, and the density solves
// with that factor (no inverse is formed), so the Mahalanobis distance, the
// log-determinant and the log-density are accurate to a few ULP even for
// ill-conditioned covariances, where the float64 eigendecomposition SciPy
// uses loses cond(Σ)·ε. A singular covariance (AllowSingular) is handled as
// SciPy does, with the pseudo-inverse from the eigendecomposition.
type MultivariateNormal struct {
	d    int
	mean []float64
	cov  []float64

	rank    int
	logPDet dd.DD

	// Nonsingular: the double-double lower Cholesky factor.
	chol []dd.DD

	// Singular: SciPy's whitening matrix U (d×d), the null-space basis
	// (d×(d-rank)) and the support tolerance.
	white   []float64
	null    []float64
	suppEps float64
}

// NewMultivariateNormal returns the multivariate normal distribution with
// the given mean (nil is the zero vector) and covariance: nil is the
// identity, a 0-d array s is s·I, a 1-D array the diagonal, a 2-D array the
// matrix (SciPy's conventions). The dimension is len(mean), or, when mean is
// nil, the side of cov. It returns an error wrapping ErrParam when cov is
// not symmetric positive semidefinite, is singular without AllowSingular,
// or does not match the mean.
func NewMultivariateNormal(mean []float64, cov *ndarray.Array, o *MultivariateNormalOptions) (*MultivariateNormal, error) {
	d := len(mean)
	if mean == nil {
		d = matrixDim(cov)
		mean = make([]float64, d)
	}
	if d == 0 {
		return nil, fmt.Errorf("%w: the mean is empty", ErrParam)
	}
	for _, m := range mean {
		if math.IsNaN(m) || math.IsInf(m, 0) {
			return nil, fmt.Errorf("%w: the mean must be finite", ErrParam)
		}
	}
	c, err := matrixArg(cov, d, "cov")
	if err != nil {
		return nil, err
	}
	m := &MultivariateNormal{d: d, mean: append([]float64(nil), mean...), cov: c}
	w, v := eighLower(c, d)
	eps := eigCutoff(w)
	if w[0] < -eps {
		return nil, fmt.Errorf("%w: the covariance must be symmetric positive semidefinite", ErrParam)
	}
	for _, x := range w {
		if x > eps {
			m.rank++
		}
	}
	if m.rank == d {
		m.chol = ddCholesky(c, d)
		for i := range d {
			m.logPDet = dd.Add(m.logPDet, dd.MulF(logDD(m.chol[i*d+i]), 2))
		}
		return m, nil
	}
	if o == nil || !o.AllowSingular {
		return nil, fmt.Errorf("%w: the covariance is singular (set AllowSingular to accept it)", ErrParam)
	}
	m.white = make([]float64, d*d)
	k := d - m.rank
	m.null = make([]float64, 0, d*k)
	var logs []float64
	for j, x := range w {
		if x > eps {
			logs = append(logs, math.Log(x))
			s := math.Sqrt(1 / x)
			for i := range d {
				m.white[i*d+j] = float64(v[i*d+j] * s)
			}
		}
	}
	for i := range d {
		for j, x := range w {
			if x <= eps {
				m.null = append(m.null, v[i*d+j])
			}
		}
	}
	m.logPDet = dd.F(sum(logs))
	m.suppEps = 1e3 * eps
	return m, nil
}

// Dim returns the dimension d.
func (m *MultivariateNormal) Dim() int { return m.d }

// Mean returns the mean vector (a copy).
func (m *MultivariateNormal) Mean() []float64 { return append([]float64(nil), m.mean...) }

// Cov returns the covariance matrix, d×d (a copy).
func (m *MultivariateNormal) Cov() *ndarray.Array { return mustArray(m.cov, m.d, m.d) }

// Rank returns the rank of the covariance (d unless it is singular).
func (m *MultivariateNormal) Rank() int { return m.rank }

// LogPDet returns the logarithm of the pseudo-determinant of the
// covariance: the sum of the logarithms of its nonnegligible eigenvalues
// (the log-determinant when it is nonsingular).
func (m *MultivariateNormal) LogPDet() float64 { return m.logPDet.Float() }

// Entropy returns the differential entropy (rank·(1 + log 2π) +
// log pdet)/2: scipy.stats.multivariate_normal.entropy.
func (m *MultivariateNormal) Entropy() float64 {
	r := float64(m.rank)
	t := dd.MulF(dd.AddF(dd.MulF(logSqrt2Pi, 2), 1), r)
	return dd.MulF(dd.Add(t, m.logPDet), 0.5).Float()
}

// logPDF is the log-density at the point x (len d) in double-double, with
// a flag for a non-finite result already decided (-Inf or NaN in r).
func (m *MultivariateNormal) logPDF(x []float64) (h dd.DD, r float64, special bool) {
	d := m.d
	for _, v := range x {
		if math.IsNaN(v) {
			return dd.DD{}, math.NaN(), true
		}
	}
	for _, v := range x {
		if math.IsInf(v, 0) {
			return dd.DD{}, math.Inf(-1), true
		}
	}
	if m.chol == nil {
		return m.logPDFSingular(x)
	}
	var zb [8]dd.DD
	z := append(zb[:0], make([]dd.DD, d)...) // on the stack up to d = 8
	var maha dd.DD
	for i := range d {
		hi, lo := dd.TwoSum(x[i], -m.mean[i])
		t := dd.DD{Hi: hi, Lo: lo}
		for k := range i {
			t = dd.Sub(t, dd.Mul(m.chol[i*d+k], z[k]))
		}
		z[i] = dd.Div(t, m.chol[i*d+i])
		if math.Abs(z[i].Hi) > 1e150 {
			return dd.DD{}, math.Inf(-1), true
		}
		maha = dd.Add(maha, dd.Mul(z[i], z[i]))
	}
	c := dd.Add(dd.MulF(logSqrt2Pi, 2*float64(d)), m.logPDet)
	return dd.MulF(dd.Add(c, maha), -0.5), 0, false
}

// logPDFSingular is SciPy's computation for a singular covariance:
// maha = Σ((x-μ)·U)², -∞ off the support.
func (m *MultivariateNormal) logPDFSingular(x []float64) (dd.DD, float64, bool) {
	d := m.d
	dev := make([]float64, d)
	for i := range d {
		dev[i] = x[i] - m.mean[i]
	}
	k := d - m.rank
	res := make([]float64, k)
	for j := range k {
		t := 0.0
		for i := range d {
			t += float64(dev[i] * m.null[i*k+j])
		}
		res[j] = float64(t * t)
	}
	if !(math.Sqrt(sum(res)) < m.suppEps) {
		return dd.DD{}, math.Inf(-1), true
	}
	sq := make([]float64, d)
	for j := range d {
		t := 0.0
		for i := range d {
			t += float64(dev[i] * m.white[i*d+j])
		}
		sq[j] = float64(t * t)
	}
	maha := sum(sq)
	v := -0.5 * (float64(m.rank)*1.8378770664093453 + m.logPDet.Hi + maha)
	return dd.F(v), 0, false
}

// LogPDF returns the log-density at the point x, which must have d
// coordinates: scipy.stats.multivariate_normal.logpdf.
func (m *MultivariateNormal) LogPDF(x []float64) (float64, error) {
	if len(x) != m.d {
		return math.NaN(), fmt.Errorf("%w: a point of dimension %d, the distribution has dimension %d", ErrData, len(x), m.d)
	}
	h, r, s := m.logPDF(x)
	if s {
		return r, nil
	}
	return h.Float(), nil
}

// PDF returns the density at the point x, which must have d coordinates:
// scipy.stats.multivariate_normal.pdf. It is the exponential of the
// double-double log-density, so it is accurate to about one ULP.
func (m *MultivariateNormal) PDF(x []float64) (float64, error) {
	if len(x) != m.d {
		return math.NaN(), fmt.Errorf("%w: a point of dimension %d, the distribution has dimension %d", ErrData, len(x), m.d)
	}
	h, r, s := m.logPDF(x)
	if s {
		return math.Exp(r), nil
	}
	return expDD(h), nil
}

// LogPDFPoints returns the log-density at each point of x, read as SciPy
// reads quantiles: coordinates along the last axis (a 1-D x is n points
// when d is 1, one point otherwise). The result has one value per point,
// in row-major order of the leading axes.
func (m *MultivariateNormal) LogPDFPoints(x *ndarray.Array) ([]float64, error) {
	return m.points(x, m.LogPDF)
}

// PDFPoints is PDF at each point of x, read as in LogPDFPoints.
func (m *MultivariateNormal) PDFPoints(x *ndarray.Array) ([]float64, error) {
	return m.points(x, m.PDF)
}

func (m *MultivariateNormal) points(x *ndarray.Array, f func([]float64) (float64, error)) ([]float64, error) {
	v, n, err := mvPoints(x, m.d)
	if err != nil {
		return nil, err
	}
	out := make([]float64, n)
	for i := range n {
		out[i], _ = f(v[i*m.d : (i+1)*m.d])
	}
	return out, nil
}

// Rvs draws size points, an array of shape (size, d), as
// scipy.stats.multivariate_normal.rvs(mean, cov, size, random_state=g)
// does: NumPy's Generator.multivariate_normal with its SVD factor. The
// standard normal draws are NumPy's bit for bit; the points are NumPy's
// exactly for d = 1 and to a few ULP for d ≥ 2 (the two SVDs round
// differently; very rarely they choose opposite signs for a singular
// vector, and the points then differ, though they have the same
// distribution).
func (m *MultivariateNormal) Rvs(g *random.Generator, size int) (*ndarray.Array, error) {
	v, err := npMultivariateNormal(g, m.mean, m.cov, m.d, size)
	if err != nil {
		return nil, err
	}
	return mustArray(v, size, m.d), nil
}

// MultivariateNormalFitOptions configures FitMultivariateNormal.
type MultivariateNormalFitOptions struct {
	// FixMean, when not nil, is the mean (length d) instead of the sample
	// mean: SciPy's fix_mean.
	FixMean []float64
	// FixCov, when not nil, is the covariance (d×d, symmetric positive
	// semidefinite) instead of the estimate: SciPy's fix_cov.
	FixCov *ndarray.Array
}

// FitMultivariateNormal returns the maximum likelihood estimates of the
// mean and the covariance (with divisor n, ddof 0) from x, n points of
// dimension d as the rows of an n×d array: scipy.stats.multivariate_normal.
// fit(x, fix_mean, fix_cov). The mean is NumPy's column mean to the bit.
func FitMultivariateNormal(x *ndarray.Array, o *MultivariateNormalFitOptions) (mean []float64, cov *ndarray.Array, err error) {
	if x == nil || len(x.Shape()) != 2 {
		return nil, nil, fmt.Errorf("%w: x must be two-dimensional", ErrData)
	}
	s := x.Shape()
	n, d := s[0], s[1]
	v := arrayFloats(x)
	if o == nil {
		o = &MultivariateNormalFitOptions{}
	}
	if o.FixMean != nil {
		if len(o.FixMean) != d {
			return nil, nil, fmt.Errorf("%w: FixMean has length %d, the data dimension %d", ErrParam, len(o.FixMean), d)
		}
		mean = append([]float64(nil), o.FixMean...)
	} else {
		mean = columnMeans(v, n, d)
	}
	if o.FixCov != nil {
		fs := o.FixCov.Shape()
		for len(fs) < 2 {
			fs = append([]int{1}, fs...)
		}
		if len(fs) != 2 || fs[0] != d || fs[1] != d {
			return nil, nil, fmt.Errorf("%w: FixCov must be %d×%d", ErrParam, d, d)
		}
		c := arrayFloats(o.FixCov)
		w, _ := eighLower(c, d)
		if w[0] < -eigCutoff(w) {
			return nil, nil, fmt.Errorf("%w: FixCov must be symmetric positive semidefinite", ErrParam)
		}
		return mean, mustArray(c, d, d), nil
	}
	cv := make([]float64, d*d)
	cen := make([]float64, n*d)
	for i := range n {
		for j := range d {
			cen[i*d+j] = v[i*d+j] - mean[j]
		}
	}
	col := make([]float64, n)
	for a := range d {
		for b := range d {
			for i := range n {
				col[i] = float64(cen[i*d+a] * cen[i*d+b])
			}
			cv[a*d+b] = sum(col) / float64(n)
		}
	}
	return mean, mustArray(cv, d, d), nil
}

// columnMeans is x.mean(axis=0) for an n×d row-major x as NumPy computes
// it: a reduction over the outer axis adds the rows in order, one
// accumulator per column.
func columnMeans(v []float64, n, d int) []float64 {
	m := make([]float64, d)
	if d == 1 {
		m[0] = mean(v)
		return m
	}
	for i := range n {
		for j := range d {
			m[j] += v[i*d+j]
		}
	}
	for j := range d {
		m[j] /= float64(n)
	}
	return m
}
