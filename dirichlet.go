package stats

import (
	"fmt"
	"math"

	"github.com/go-ndarray/ndarray"
	"github.com/go-ndarray/ndarray/random"
	"github.com/go-numerics/special"
	"github.com/go-statistics/stats/internal/dd"
)

// Dirichlet is the Dirichlet distribution on the (K-1)-simplex with
// concentration parameters Alpha (K of them, all positive):
// scipy.stats.dirichlet(alpha).
//
// The log-density -log B(α) + Σ(αᵢ-1)·log xᵢ is summed in double-double
// with double-double logarithms, so that its only rounding errors are
// those of the log-gamma values; SciPy sums the float64 terms.
type Dirichlet struct {
	alpha []float64
	lnB   dd.DD
}

// NewDirichlet returns the Dirichlet distribution with the given
// concentrations. It returns an error wrapping ErrParam unless there are
// at least one and all are positive and finite.
func NewDirichlet(alpha []float64) (*Dirichlet, error) {
	if len(alpha) == 0 {
		return nil, fmt.Errorf("%w: alpha is empty", ErrParam)
	}
	for _, a := range alpha {
		if !(a > 0) || math.IsInf(a, 1) {
			return nil, fmt.Errorf("%w: every alpha must be positive and finite", ErrParam)
		}
	}
	d := &Dirichlet{alpha: append([]float64(nil), alpha...)}
	for _, a := range alpha {
		d.lnB = dd.AddF(d.lnB, special.LogGamma(a))
	}
	d.lnB = dd.AddF(d.lnB, -special.LogGamma(sum(alpha)))
	return d, nil
}

// Alpha returns the concentration parameters (a copy).
func (d *Dirichlet) Alpha() []float64 { return append([]float64(nil), d.alpha...) }

// point completes x (K or K-1 coordinates) and checks it as SciPy's
// _dirichlet_check_input does.
func (d *Dirichlet) point(x []float64) ([]float64, error) {
	k := len(d.alpha)
	switch len(x) {
	case k:
		x = append([]float64(nil), x...)
	case k - 1:
		x = append(append([]float64(nil), x...), 1-sum(x))
	default:
		return nil, fmt.Errorf("%w: a point needs %d or %d coordinates, it has %d", ErrData, k, k-1, len(x))
	}
	for i, v := range x {
		switch {
		case v < 0:
			return nil, fmt.Errorf("%w: each coordinate must be at least 0", ErrData)
		case v > 1:
			return nil, fmt.Errorf("%w: each coordinate must be at most 1", ErrData)
		case v == 0 && d.alpha[i] < 1:
			return nil, fmt.Errorf("%w: a coordinate must be positive where its alpha is below 1", ErrData)
		case math.IsNaN(v):
			return nil, fmt.Errorf("%w: a coordinate is NaN", ErrData)
		}
	}
	if math.Abs(sum(x)-1) > 10e-10 {
		return nil, fmt.Errorf("%w: the point must lie on the simplex (its coordinates sum to %v)", ErrData, sum(x))
	}
	return x, nil
}

func (d *Dirichlet) logPDF(x []float64) dd.DD {
	s := dd.Neg(d.lnB)
	for i, v := range x {
		a := d.alpha[i]
		if a == 1 {
			continue // xlogy(0, x) = 0, also at x = 0
		}
		if v == 0 { // here a > 1: the density is 0
			return dd.F(math.Inf(-1))
		}
		hi, lo := dd.TwoSum(a, -1)
		s = dd.Add(s, dd.Mul(dd.DD{Hi: hi, Lo: lo}, dd.LogAccurate(v)))
	}
	return s
}

// LogPDF returns the log-density at x, given by its K coordinates or the
// first K-1 (the last is then 1 minus their sum):
// scipy.stats.dirichlet.logpdf. A point off the simplex is an error
// wrapping ErrData, as SciPy raises.
func (d *Dirichlet) LogPDF(x []float64) (float64, error) {
	x, err := d.point(x)
	if err != nil {
		return math.NaN(), err
	}
	h := d.logPDF(x)
	return h.Hi + h.Lo, nil
}

// PDF returns the density at x (see LogPDF): scipy.stats.dirichlet.pdf.
func (d *Dirichlet) PDF(x []float64) (float64, error) {
	x, err := d.point(x)
	if err != nil {
		return math.NaN(), err
	}
	h := d.logPDF(x)
	if math.IsInf(h.Hi, -1) {
		return 0, nil
	}
	return expDD(h), nil
}

// LogPDFPoints returns LogPDF at each row of x, an (n, K) or (n, K-1)
// array (a 1-D x is one point). SciPy's dirichlet takes the transpose,
// components along the first axis; here points are rows as everywhere in
// this package.
func (d *Dirichlet) LogPDFPoints(x *ndarray.Array) ([]float64, error) {
	return d.rows(x, d.LogPDF)
}

// PDFPoints is PDF at each row of x (see LogPDFPoints).
func (d *Dirichlet) PDFPoints(x *ndarray.Array) ([]float64, error) {
	return d.rows(x, d.PDF)
}

func (d *Dirichlet) rows(x *ndarray.Array, f func([]float64) (float64, error)) ([]float64, error) {
	if x == nil {
		return nil, fmt.Errorf("%w: no points", ErrData)
	}
	s := x.Shape()
	if len(s) == 0 || len(s) > 2 {
		return nil, fmt.Errorf("%w: points must be a 1-D or 2-D array", ErrData)
	}
	v := arrayFloats(x)
	w := s[len(s)-1]
	n := len(v) / w
	out := make([]float64, n)
	for i := range n {
		r, err := f(v[i*w : (i+1)*w])
		if err != nil {
			return nil, err
		}
		out[i] = r
	}
	return out, nil
}

// Mean returns α/α₀ with α₀ = Σα: scipy.stats.dirichlet.mean.
func (d *Dirichlet) Mean() []float64 {
	a0 := sum(d.alpha)
	out := make([]float64, len(d.alpha))
	for i, a := range d.alpha {
		out[i] = a / a0
	}
	return out
}

// Var returns the variances α(α₀-α)/(α₀²(α₀+1)): scipy.stats.dirichlet.var.
func (d *Dirichlet) Var() []float64 {
	a0 := sum(d.alpha)
	out := make([]float64, len(d.alpha))
	for i, a := range d.alpha {
		out[i] = float64(a*(a0-a)) / float64(float64(a0*a0)*(a0+1))
	}
	return out
}

// Cov returns the K×K covariance matrix (diag(a) - a·aᵀ)/(α₀+1) with
// a = α/α₀: scipy.stats.dirichlet.cov.
func (d *Dirichlet) Cov() *ndarray.Array {
	a := d.Mean()
	k := len(a)
	den := sum(d.alpha) + 1
	c := make([]float64, k*k)
	for i := range k {
		for j := range k {
			v := -float64(a[i] * a[j])
			if i == j {
				v = a[i] - float64(a[i]*a[i])
			}
			c[i*k+j] = v / den
		}
	}
	return mustArray(c, k, k)
}

// Entropy returns the differential entropy log B(α) + (α₀-K)ψ(α₀) -
// Σ(αᵢ-1)ψ(αᵢ): scipy.stats.dirichlet.entropy.
func (d *Dirichlet) Entropy() float64 {
	a0 := sum(d.alpha)
	k := float64(len(d.alpha))
	t := make([]float64, len(d.alpha))
	for i, a := range d.alpha {
		t[i] = float64((a - 1) * special.Digamma(a))
	}
	return d.lnB.Hi + d.lnB.Lo + float64((a0-k)*special.Digamma(a0)) - sum(t)
}

// Rvs draws size points, an array of shape (size, K), as
// scipy.stats.dirichlet.rvs(alpha, size, random_state=g) does: NumPy's
// Generator.dirichlet, reproduced bit for bit by go-ndarray.
func (d *Dirichlet) Rvs(g *random.Generator, size int) (*ndarray.Array, error) {
	r, err := g.Dirichlet(d.alpha, size)
	if err != nil {
		return nil, fmt.Errorf("%w: %v", ErrParam, err)
	}
	return r, nil
}
