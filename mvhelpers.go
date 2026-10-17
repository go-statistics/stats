package stats

import (
	"fmt"
	"math"

	"github.com/go-ndarray/ndarray"
	"github.com/go-ndarray/ndarray/linalg"
	"github.com/go-ndarray/ndarray/random"
	"github.com/go-statistics/stats/internal/dd"
)

// Helpers shared by the multivariate distributions, the kernel density
// estimate and the bootstrap: point arrays, matrices and the draws NumPy's
// Generator.multivariate_normal makes.

// mvPoints reads x as points of dimension d the way SciPy's multivariate
// distributions read their quantiles (_process_quantiles): a 0-d array is one
// coordinate, a 1-D array is n points when d is 1 and one point otherwise,
// and an array of two or more axes has the coordinates along its last axis.
// It returns the coordinates row-major and the number of points.
func mvPoints(x *ndarray.Array, d int) ([]float64, int, error) {
	if x == nil {
		return nil, 0, fmt.Errorf("%w: no points", ErrData)
	}
	v := arrayFloats(x)
	shape := x.Shape()
	last := 1 // a 0-d array, or a 1-D array of points in one dimension
	if len(shape) >= 2 || len(shape) == 1 && d > 1 {
		last = shape[len(shape)-1]
	}
	if last != d {
		return nil, 0, fmt.Errorf("%w: points of dimension %d, the distribution has dimension %d", ErrData, last, d)
	}
	return v, len(v) / d, nil
}

// matrixArg reads a covariance-like argument the way SciPy's
// _process_parameters_psd does: nil is the identity, a 0-d array s is s·I,
// a 1-D array is a diagonal, and a 2-D array is the matrix itself, which
// must be d×d.
func matrixArg(c *ndarray.Array, d int, name string) ([]float64, error) {
	m := make([]float64, d*d)
	if c == nil {
		for i := range d {
			m[i*d+i] = 1
		}
		return m, nil
	}
	v := arrayFloats(c)
	shape := c.Shape()
	switch len(shape) {
	case 0:
		for i := range d {
			m[i*d+i] = v[0]
		}
	case 1:
		if shape[0] != d {
			return nil, fmt.Errorf("%w: %s has %d diagonal entries for dimension %d", ErrParam, name, shape[0], d)
		}
		for i := range d {
			m[i*d+i] = v[i]
		}
	case 2:
		if shape[0] != d || shape[1] != d {
			return nil, fmt.Errorf("%w: %s has shape %v for dimension %d", ErrParam, name, shape, d)
		}
		copy(m, v)
	default:
		return nil, fmt.Errorf("%w: %s must have at most two axes, it has %d", ErrParam, name, len(shape))
	}
	for _, x := range m {
		if math.IsNaN(x) || math.IsInf(x, 0) {
			return nil, fmt.Errorf("%w: %s must be finite", ErrParam, name)
		}
	}
	return m, nil
}

// matrixDim is the dimension SciPy infers from a covariance argument when
// no mean is given: the side of a matrix, the length of a diagonal, else 1.
func matrixDim(c *ndarray.Array) int {
	if c == nil {
		return 1
	}
	if s := c.Shape(); len(s) >= 1 {
		return s[0]
	}
	return 1
}

// mustArray wraps v in an array of the given shape; the lengths always
// agree where it is called.
func mustArray(v []float64, shape ...int) *ndarray.Array {
	a, _ := ndarray.FromData(v, shape...)
	return a
}

// eighLower is the symmetric eigendecomposition of the d×d matrix m read
// from its lower triangle (scipy.linalg.eigh(m, lower=True)): eigenvalues
// ascending and eigenvectors as the columns of a row-major d×d matrix.
func eighLower(m []float64, d int) (w, v []float64) {
	wa, va, _ := linalg.Eigh(mustArray(m, d, d), linalg.Lower)
	w, _ = ndarray.Data[float64](wa)
	v, _ = ndarray.Data[float64](va)
	return w, v
}

// eigCutoff is SciPy's _eigvalsh_to_eps for float64: eigenvalues of
// magnitude at most 1e6·ε·max|λ| are negligible.
func eigCutoff(w []float64) float64 {
	m := 0.0
	for _, x := range w {
		m = max(m, math.Abs(x))
	}
	return 1e6 * 0x1p-52 * m
}

// ddCholesky is the lower Cholesky factor of the d×d matrix m, read from its
// lower triangle, computed in double-double. The caller has checked that m
// is positive definite well above rounding (its smallest eigenvalue exceeds
// 1e6·ε times its largest), so every pivot is positive.
func ddCholesky(m []float64, d int) []dd.DD {
	l := make([]dd.DD, d*d)
	for j := range d {
		s := dd.F(m[j*d+j])
		for k := range j {
			s = dd.Sub(s, dd.Mul(l[j*d+k], l[j*d+k]))
		}
		ljj := dd.Sqrt(s)
		l[j*d+j] = ljj
		for i := j + 1; i < d; i++ {
			t := dd.F(m[i*d+j])
			for k := range j {
				t = dd.Sub(t, dd.Mul(l[i*d+k], l[j*d+k]))
			}
			l[i*d+j] = dd.Div(t, ljj)
		}
	}
	return l
}

// cholesky is the float64 lower Cholesky factor of a positive definite
// matrix (scipy.linalg.cholesky(m, lower=True)), with ok false when a pivot
// is not positive.
func cholesky(m []float64, d int) ([]float64, bool) {
	l := make([]float64, d*d)
	for j := range d {
		s := m[j*d+j]
		for k := range j {
			s -= float64(l[j*d+k] * l[j*d+k])
		}
		if !(s > 0) {
			return nil, false
		}
		ljj := math.Sqrt(s)
		l[j*d+j] = ljj
		for i := j + 1; i < d; i++ {
			t := m[i*d+j]
			for k := range j {
				t -= float64(l[i*d+k] * l[j*d+k])
			}
			l[i*d+j] = t / ljj
		}
	}
	return l, true
}

// npMultivariateNormal draws size points as NumPy's
// Generator.multivariate_normal(mean, cov, size) does with its default
// method="svd": standard normals of shape (size, d), the SVD cov = U·S·Vᵀ,
// and mean + x @ (U·√S)ᵀ. The draws are NumPy's; the factor is the same up to
// the rounding of the two SVD implementations (and, rarely, the sign of a
// singular vector), so the points agree with NumPy's only to a few ULP for
// d ≥ 2 and exactly for d = 1.
func npMultivariateNormal(g *random.Generator, mean, cov []float64, d, size int) ([]float64, error) {
	xa, err := g.StandardNormal(size, d)
	if err != nil {
		return nil, fmt.Errorf("%w: %v", ErrParam, err)
	}
	x, _ := ndarray.Data[float64](xa)
	ua, sa, _, _ := linalg.SVD(mustArray(cov, d, d), true) // cov is finite
	u, _ := ndarray.Data[float64](ua)
	s, _ := ndarray.Data[float64](sa)
	f := make([]float64, d*d)
	for i := range d {
		for k := range d {
			f[i*d+k] = float64(u[i*d+k] * math.Sqrt(math.Abs(s[k])))
		}
	}
	out := make([]float64, size*d)
	for i := range size {
		row := x[i*d : i*d+d]
		for j := range d {
			t := 0.0
			for k := range d {
				t += float64(row[k] * f[j*d+k])
			}
			out[i*d+j] = mean[j] + t
		}
	}
	return out, nil
}
