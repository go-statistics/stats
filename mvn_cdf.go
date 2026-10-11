package stats

import (
	"fmt"
	"math"
	"math/cmplx"

	"github.com/go-ndarray/ndarray"
	"github.com/go-ndarray/ndarray/random"
	"github.com/go-numerics/special"
)

// MultivariateNormalCDFOptions configures MultivariateNormal.CDF and
// LogCDF. The zero value is SciPy's default.
type MultivariateNormalCDFOptions struct {
	// Lower is the lower limit of integration (SciPy's lower_limit); nil is
	// -∞ in every coordinate. Where a lower limit exceeds the upper one the
	// two are swapped and the sign of the result flips, as in SciPy.
	Lower []float64
	// MaxPts bounds the number of integration points for d ≥ 3; 0 is
	// 1000000·d (SciPy's maxpts).
	MaxPts int
	// AbsEps is the target error for d ≥ 3; 0 is 1e-5 (SciPy's abseps;
	// SciPy's releps is not used by its integrator and has no counterpart).
	AbsEps float64
	// RNG randomizes the lattice rule for d ≥ 3 (SciPy's rng). nil uses a
	// fresh random.DefaultRNG(0) for each call, so that results are
	// deterministic; pass a generator to reproduce SciPy's value for the
	// same seed (it is drawn from point after point, as SciPy does).
	RNG *random.Generator
}

// CDFEstimate is a multivariate normal probability with its error estimate.
type CDFEstimate struct {
	// Value is the probability.
	Value float64
	// Error is the error estimate: three standard errors of the randomized
	// lattice batches for d ≥ 3, 1e-15 (nominal, as SciPy reports) for
	// d ≤ 2 where the computation is deterministic.
	Error float64
	// NPoints is the number of integration points used (0 for d ≤ 2).
	NPoints int
}

// CDF returns P(Lower < X ≤ x): scipy.stats.multivariate_normal.cdf(x,
// mean, cov, maxpts, abseps, lower_limit=Lower, rng=RNG).
//
// For d = 1 it is the normal cdf, special.Ndtr((x-μ)/σ) (or the difference
// of two survival functions above the mean). For d = 2 it is Genz's
// bivariate normal algorithm (as in SciPy), but applied to the orthant or
// rectangle reflected towards the lower-left tail, so that a small
// probability is not the cancelling difference of numbers near 1: SciPy
// returns 0 for P(X ≤ -10, Y ≤ -10); this returns the probability to a
// relative accuracy of about 1e-14. For d ≥ 3 it is the Genz–Bretz
// randomized lattice rule of SciPy's _qmvn with its automatic refinement:
// with the same generator the lattice, the shifts and the value are
// SciPy's.
func (m *MultivariateNormal) CDF(x []float64, o *MultivariateNormalCDFOptions) (float64, error) {
	e, err := m.CDFEstimate(x, o)
	return e.Value, err
}

// LogCDF returns the logarithm of CDF (NaN where a swapped limit makes the
// probability negative; SciPy returns a complex number there). For d = 1
// with no lower limit it is special.LogNdtr, accurate far in the tail.
func (m *MultivariateNormal) LogCDF(x []float64, o *MultivariateNormalCDFOptions) (float64, error) {
	if m.d == 1 && (o == nil || o.Lower == nil) && len(x) == 1 {
		return special.LogNdtr((x[0] - m.mean[0]) / math.Sqrt(m.cov[0])), nil
	}
	p, err := m.CDF(x, o)
	return math.Log(p), err
}

// CDFEstimate is CDF with the error estimate and the number of points.
func (m *MultivariateNormal) CDFEstimate(x []float64, o *MultivariateNormalCDFOptions) (CDFEstimate, error) {
	d := m.d
	if len(x) != d {
		return CDFEstimate{Value: math.NaN()}, fmt.Errorf("%w: a point of dimension %d, the distribution has dimension %d", ErrData, len(x), d)
	}
	if o == nil {
		o = &MultivariateNormalCDFOptions{}
	}
	if o.Lower != nil && len(o.Lower) != d {
		return CDFEstimate{Value: math.NaN()}, fmt.Errorf("%w: Lower has dimension %d, the distribution %d", ErrParam, len(o.Lower), d)
	}
	maxpts := o.MaxPts
	if maxpts <= 0 {
		maxpts = 1000000 * d
	}
	abseps := o.AbsEps
	if abseps <= 0 {
		abseps = 1e-5
	}
	g := o.RNG
	if g == nil {
		g = random.DefaultRNG(0)
	}
	a := make([]float64, d)
	b := make([]float64, d)
	sign := 1.0
	for i := range d {
		lo := math.Inf(-1)
		if o.Lower != nil {
			lo = o.Lower[i]
		}
		b[i], a[i] = x[i]-m.mean[i], lo-m.mean[i]
		if b[i] < a[i] {
			a[i], b[i] = b[i], a[i]
			sign = -sign
		}
	}
	var e CDFEstimate
	switch d {
	case 1:
		s := math.Sqrt(m.cov[0])
		e = CDFEstimate{Value: normalInterval(a[0]/s, b[0]/s), Error: 1e-15}
	case 2:
		e = CDFEstimate{Value: bvnRect(a, b, m.cov), Error: 1e-15}
	default:
		e.Value, e.Error, e.NPoints = qauto(m.cov, a, b, g, abseps, maxpts)
	}
	e.Value *= sign
	return e, nil
}

// CDFPoints is CDF at each point of x (read as in LogPDFPoints), drawing
// from one generator point after point as SciPy does.
func (m *MultivariateNormal) CDFPoints(x *ndarray.Array, o *MultivariateNormalCDFOptions) ([]float64, error) {
	v, n, err := mvPoints(x, m.d)
	if err != nil {
		return nil, err
	}
	oo := MultivariateNormalCDFOptions{}
	if o != nil {
		oo = *o
	}
	if oo.RNG == nil {
		oo.RNG = random.DefaultRNG(0)
	}
	out := make([]float64, n)
	for i := range n {
		out[i], err = m.CDF(v[i*m.d:(i+1)*m.d], &oo)
		if err != nil {
			return nil, err
		}
	}
	return out, nil
}

// normalInterval is P(a < Z ≤ b) for a standard normal Z, a ≤ b, as the
// difference of the two survival functions when a > 0.
func normalInterval(a, b float64) float64 {
	if a > 0 {
		return special.Ndtr(-a) - special.Ndtr(-b)
	}
	return special.Ndtr(b) - special.Ndtr(a)
}

// bvnRect is P(a < X ≤ b) for a centred bivariate normal with covariance c
// (2×2 row-major), a ≤ b. Each coordinate whose interval lies mostly above
// zero is reflected (which flips the sign of the correlation), so that the
// four orthant probabilities of the inclusion–exclusion are lower-tail
// ones and a small rectangle far out is not a difference of numbers near 1.
// The result is clipped to [0, 1] as in SciPy.
func bvnRect(a, b, c []float64) float64 {
	s1, s2 := math.Sqrt(c[0]), math.Sqrt(c[3])
	r := c[1] / (s1 * s2)
	xl, xu := a[0]/s1, b[0]/s1
	yl, yu := a[1]/s2, b[1]/s2
	if xl+xu > 0 {
		xl, xu = -xu, -xl
		r = -r
	}
	if yl+yu > 0 {
		yl, yu = -yu, -yl
		r = -r
	}
	p := bvnl(xu, yu, r) - bvnl(xl, yu, r) - bvnl(xu, yl, r) + bvnl(xl, yl, r)
	return math.Max(0, math.Min(p, 1))
}

// bvnl is the lower orthant probability P(X ≤ h, Y ≤ k) of a standard
// bivariate normal with correlation r.
func bvnl(h, k, r float64) float64 {
	switch {
	case h == math.Inf(-1) || k == math.Inf(-1):
		return 0
	case h == math.Inf(1):
		return special.Ndtr(k)
	case k == math.Inf(1):
		return special.Ndtr(h)
	}
	return bvnu(-h, -k, r)
}

// Gauss–Legendre nodes (the negative half) and weights of orders 6, 12
// and 20 for bvnu.
var (
	glX = [3][]float64{
		{-0.932469514203152, -0.6612093864662645, -0.2386191860831969},
		{-0.9815606342467192, -0.9041172563704749, -0.7699026741943047, -0.5873179542866175, -0.3678314989981802, -0.1252334085114689},
		{-0.9931285991850949, -0.9639719272779138, -0.912234428251326, -0.8391169718222188, -0.7463319064601508, -0.636053680726515, -0.5108670019508271, -0.37370608871541955, -0.22778585114164507, -0.07652652113349734},
	}
	glW = [3][]float64{
		{0.17132449237917036, 0.3607615730481386, 0.46791393457269104},
		{0.04717533638651183, 0.10693932599531843, 0.16007832854334622, 0.20316742672306592, 0.2334925365383548, 0.24914704581340277},
		{0.017614007139152118, 0.04060142980038694, 0.06267204833410907, 0.08327674157670475, 0.10193011981724044, 0.11819453196151841, 0.13168863844917664, 0.14209610931838204, 0.14917298647260374, 0.15275338713072584},
	}
)

// bvnu is P(X > h, Y > k) for a standard bivariate normal with correlation
// r, by Genz's BVNU (Drezner and Wesolowsky's method with Gauss–Legendre
// quadrature; A. Genz, "Numerical computation of rectangular bivariate and
// trivariate normal and t probabilities", Statistics and Computing 14,
// 2004), for finite h and k.
func bvnu(h, k, r float64) float64 {
	const twoPi = 2 * math.Pi
	ng := 2
	switch {
	case math.Abs(r) < 0.3:
		ng = 0
	case math.Abs(r) < 0.75:
		ng = 1
	}
	xs, ws := glX[ng], glW[ng]
	hk := h * k
	bvn := 0.0
	if math.Abs(r) < 0.925 {
		hs := (h*h + k*k) / 2
		asr := math.Asin(r)
		for i := range xs {
			sn := math.Sin(asr * (xs[i] + 1) / 2)
			bvn += ws[i] * math.Exp((sn*hk-hs)/(1-sn*sn))
			sn = math.Sin(asr * (-xs[i] + 1) / 2)
			bvn += ws[i] * math.Exp((sn*hk-hs)/(1-sn*sn))
		}
		return bvn*asr/(2*twoPi) + special.Ndtr(-h)*special.Ndtr(-k)
	}
	if r < 0 {
		k = -k
		hk = -hk
	}
	if math.Abs(r) < 1 {
		as := (1 - r) * (1 + r)
		a := math.Sqrt(as)
		bs := (h - k) * (h - k)
		c := (4 - hk) / 8
		d := (12 - hk) / 16
		bvn = a * math.Exp(-(bs/as+hk)/2) * (1 - c*(bs-as)*(1-d*bs/5)/3 + c*d*as*as/5)
		if hk > -160 {
			b := math.Sqrt(bs)
			bvn -= math.Exp(-hk/2) * math.Sqrt(twoPi) * special.Ndtr(-b/a) * b * (1 - c*bs*(1-d*bs/5)/3)
		}
		a /= 2
		for i := range xs {
			xs2 := a * (xs[i] + 1)
			xs2 *= xs2
			rs := math.Sqrt(1 - xs2)
			bvn += a * ws[i] * (math.Exp(-bs/(2*xs2)-hk/(1+rs))/rs - math.Exp(-(bs/xs2+hk)/2)*(1+c*xs2*(1+d*xs2)))
			xs2 = as * (-xs[i] + 1) * (-xs[i] + 1) / 4
			rs = math.Sqrt(1 - xs2)
			bvn += a * ws[i] * math.Exp(-(bs/xs2+hk)/2) * (math.Exp(-hk*(1-rs)/(2*(1+rs)))/rs - (1 + c*xs2*(1+d*xs2)))
		}
		bvn = -bvn / twoPi
	}
	if r > 0 {
		return bvn + special.Ndtr(-math.Max(h, k))
	}
	return -bvn + math.Max(0, special.Ndtr(-h)-special.Ndtr(-k))
}

// qauto is SciPy's _qauto with _qmvn: rerun the randomized lattice rule
// with √2 times more points until the error estimate is at most target or
// limit points have been used, merging the estimates by inverse variance.
func qauto(cov, low, high []float64, g *random.Generator, target float64, limit int) (prob, est float64, ns int) {
	n := len(low)
	mi := min(limit, n*1000)
	est = 1
	for est > target && ns < limit {
		mi = int(math.RoundToEven(math.Sqrt2 * float64(mi)))
		pi, ei, ni := qmvn(mi, cov, low, high, g)
		ns += ni
		wt := 1 / (1 + (ei/est)*(ei/est))
		prob += wt * (pi - prob)
		est = math.Sqrt(wt) * ei
	}
	return prob, est, ns
}

// qmvn is SciPy's _qmvn: m points in 10 batches of a rank-1 lattice
// (fast CBC construction) shifted by uniform draws, over the
// Genz-transformed integrand with the permuted, scaled Cholesky factor.
func qmvn(m int, cov, low, high []float64, g *random.Generator) (prob, est float64, ns int) {
	const batches = 10
	cho, lo, hi := permutedCholesky(cov, low, high)
	n := len(lo)
	q, nq := cbcLattice(n-1, max(m/batches, 3))
	ra, _ := g.Random(batches, n)
	rnd, _ := ndarray.Data[float64](ra)
	ct := cho[0]
	ci := special.Ndtr(lo[0] / ct)
	dci := special.Ndtr(hi[0]/ct) - ci
	y := make([]float64, (n-1)*nq)
	c := make([]float64, nq)
	dc := make([]float64, nq)
	pv := make([]float64, nq)
	ev := 0.0
	for j := range batches {
		for s := range nq {
			c[s], dc[s], pv[s] = ci, dci, dci
		}
		for i := 1; i < n; i++ {
			sh := rnd[j*n+i]
			row := cho[i*n : i*n+i]
			lo, hi, ct := lo[i], hi[i], cho[i*n+i]
			yi := y[(i-1)*nq : i*nq]
			for s := range nq {
				z := float64(q[i-1]*float64(s+1)) + sh
				z -= math.Trunc(z)
				xv := math.Abs(2*z - 1)
				yi[s] = special.Ndtri(c[s] + float64(xv*dc[s]))
				t := 0.0
				for k := range i {
					t += float64(row[k] * y[k*nq+s])
				}
				cs := special.Ndtr((lo - t) / ct)
				ds := special.Ndtr((hi - t) / ct)
				c[s] = cs
				dc[s] = ds - cs
				pv[s] *= dc[s]
			}
		}
		d := (mean(pv) - prob) / float64(j+1)
		prob += d
		ev = float64(j-1)*ev/float64(j+1) + d*d
	}
	return prob, 3 * math.Sqrt(ev), nq * batches
}

// permutedCholesky is SciPy's _permuted_cholesky: the Cholesky factor of
// the covariance scaled to unit diagonal, with the variables reordered
// (smallest expected interval probability first) and the bounds scaled
// and permuted alike. Negligible pivots (≤ (k+1)·1e-10) zero their column.
func permutedCholesky(cov, low, high []float64) (cho, lo, hi []float64) {
	const tol = 1e-10
	n := len(low)
	cho = append([]float64(nil), cov...)
	lo = append([]float64(nil), low...)
	hi = append([]float64(nil), high...)
	dc := make([]float64, n)
	for i := range n {
		dc[i] = math.Sqrt(math.Max(cho[i*n+i], 0))
		if dc[i] == 0 {
			dc[i] = 1
		}
		lo[i] /= dc[i]
		hi[i] /= dc[i]
	}
	for i := range n {
		for j := range n {
			cho[i*n+j] = cho[i*n+j] / dc[j] / dc[i]
		}
	}
	y := make([]float64, n)
	sqtp := math.Sqrt(2 * math.Pi)
	for k := range n {
		epk := float64(k+1) * tol
		im := k
		ck, dem, s, loM, hiM := 0.0, 1.0, 0.0, 0.0, 0.0
		for i := k; i < n; i++ {
			if cho[i*n+i] > tol {
				ci := math.Sqrt(cho[i*n+i])
				if i > 0 {
					s = 0
					for j := range k {
						s += float64(cho[i*n+j] * y[j])
					}
				}
				loI := (lo[i] - s) / ci
				hiI := (hi[i] - s) / ci
				de := special.Ndtr(hiI) - special.Ndtr(loI)
				if de <= dem {
					ck, dem, loM, hiM, im = ci, de, loI, hiI, i
				}
			}
		}
		if im > k {
			cho[im*n+im] = cho[k*n+k]
			for j := range k {
				cho[im*n+j], cho[k*n+j] = cho[k*n+j], cho[im*n+j]
			}
			for i := im + 1; i < n; i++ {
				cho[i*n+im], cho[i*n+k] = cho[i*n+k], cho[i*n+im]
			}
			for i := k + 1; i < im; i++ {
				cho[i*n+k], cho[im*n+i] = cho[im*n+i], cho[i*n+k]
			}
			lo[k], lo[im] = lo[im], lo[k]
			hi[k], hi[im] = hi[im], hi[k]
		}
		if ck > epk {
			cho[k*n+k] = ck
			for j := k + 1; j < n; j++ {
				cho[k*n+j] = 0
			}
			for i := k + 1; i < n; i++ {
				cho[i*n+k] /= ck
				for j := k + 1; j <= i; j++ {
					cho[i*n+j] -= float64(cho[i*n+k] * cho[j*n+k])
				}
			}
			if math.Abs(dem) > tol {
				y[k] = (math.Exp(-loM*loM/2) - math.Exp(-hiM*hiM/2)) / (sqtp * dem)
			} else {
				y[k] = (loM + hiM) / 2
				if loM < -10 {
					y[k] = hiM
				} else if hiM > 10 {
					y[k] = loM
				}
			}
			for j := 0; j <= k; j++ {
				cho[k*n+j] /= ck
			}
			lo[k] /= ck
			hi[k] /= ck
		} else {
			for i := k; i < n; i++ {
				cho[i*n+k] = 0
			}
			y[k] = (lo[k] + hi[k]) / 2
		}
	}
	return cho, lo, hi
}

// cbcLattice is SciPy's _cbc_lattice: the generating vector of a rank-1
// lattice rule of nq points (nq the largest prime ≤ m) in dim dimensions,
// by the fast component-by-component construction of Nuyens and Cools.
func cbcLattice(dim, m int) (q []float64, nq int) {
	nq = largestPrime(m)
	half := (nq - 1) / 2
	gen := primitiveRoot(nq)
	perm := make([]int, half)
	perm[0] = 1
	for j := 0; j < half-1; j++ {
		perm[j+1] = gen * perm[j] % nq
	}
	c := make([]float64, half)
	for j, p := range perm {
		perm[j] = min(nq-p, p)
		pn := float64(perm[j]) / float64(nq)
		c[j] = float64(pn*pn) - pn + 1.0/6
	}
	fc := fft(toComplex(c), false)
	z := make([]int, dim)
	for i := range z {
		z[i] = i + 1
	}
	qq := make([]float64, half)
	for i := range qq {
		qq[i] = 1
	}
	w := 0
	for s := 1; s < dim; s++ {
		gm := 1.0 // SciPy's gm = [1, 0.8**0, 0.8**1, ...][s-1]
		if s >= 2 {
			gm = math.Pow(0.8, float64(s-2))
		}
		re := make([]float64, 0, half)
		for j := w; j >= 0; j-- {
			re = append(re, c[j])
		}
		for j := half - 1; j > w; j-- {
			re = append(re, c[j])
		}
		for j := range qq {
			qq[j] *= 1 + float64(gm*re[j])
		}
		fq := fft(toComplex(qq), false)
		for j := range fq {
			fq[j] *= fc[j]
		}
		conv := fft(fq, true)
		w = 0
		for j := range conv {
			if real(conv[j]) < real(conv[w]) {
				w = j
			}
		}
		z[s] = perm[w]
	}
	q = make([]float64, dim)
	for i := range q {
		q[i] = float64(z[i]) / float64(nq)
	}
	return q, nq
}

// largestPrime is the largest prime ≤ m (m ≥ 3).
func largestPrime(m int) int {
	sieve := make([]bool, m+1)
	p := 2
	for i := 2; i <= m; i++ {
		if !sieve[i] {
			p = i
			for j := i * i; j <= m; j += i {
				sieve[j] = true
			}
		}
	}
	return p
}

// primitiveRoot is the smallest primitive root of the prime p.
func primitiveRoot(p int) int {
	pm := p - 1
	var fs []int
	n := pm
	for f := 2; f*f <= n; f++ {
		if n%f == 0 {
			fs = append(fs, f)
			for n%f == 0 {
				n /= f
			}
		}
	}
	if n > 1 {
		fs = append(fs, n)
	}
	for r := 2; ; r++ {
		ok := true
		for _, f := range fs {
			if powMod(r, pm/f, p) == 1 {
				ok = false
				break
			}
		}
		if ok {
			return r
		}
	}
}

func powMod(b, e, m int) int {
	r := 1
	b %= m
	for ; e > 0; e >>= 1 {
		if e&1 == 1 {
			r = r * b % m
		}
		b = b * b % m
	}
	return r
}

func toComplex(v []float64) []complex128 {
	c := make([]complex128, len(v))
	for i, x := range v {
		c[i] = complex(x, 0)
	}
	return c
}

// fft is the discrete Fourier transform of any length (the inverse,
// scaled by 1/n, when inv is set): radix 2 for powers of two, Bluestein's
// chirp z-transform otherwise.
func fft(x []complex128, inv bool) []complex128 {
	n := len(x)
	sign := -1.0
	if inv {
		sign = 1
	}
	var out []complex128
	if n&(n-1) == 0 {
		out = append([]complex128(nil), x...)
		fft2(out, sign)
	} else {
		out = bluestein(x, sign)
	}
	if inv {
		for i := range out {
			out[i] /= complex(float64(n), 0)
		}
	}
	return out
}

// fft2 is the in-place radix-2 transform with exponent sign sign.
func fft2(a []complex128, sign float64) {
	n := len(a)
	for i, j := 1, 0; i < n; i++ {
		bit := n >> 1
		for ; j&bit != 0; bit >>= 1 {
			j ^= bit
		}
		j ^= bit
		if i < j {
			a[i], a[j] = a[j], a[i]
		}
	}
	for l := 2; l <= n; l <<= 1 {
		for k := 0; k < l/2; k++ {
			w := cmplx.Rect(1, sign*2*math.Pi*float64(k)/float64(l))
			for i := k; i < n; i += l {
				u, v := a[i], a[i+l/2]*w
				a[i], a[i+l/2] = u+v, u-v
			}
		}
	}
}

// bluestein is the length-n transform as a convolution of power-of-two
// length.
func bluestein(x []complex128, sign float64) []complex128 {
	n := len(x)
	m := 1
	for m < 2*n-1 {
		m <<= 1
	}
	w := make([]complex128, n)
	for k := range n {
		// k² mod 2n keeps the angle small and exact.
		kk := (k * k) % (2 * n)
		w[k] = cmplx.Rect(1, sign*math.Pi*float64(kk)/float64(n))
	}
	a := make([]complex128, m)
	b := make([]complex128, m)
	for k := range n {
		a[k] = x[k] * w[k]
	}
	b[0] = cmplx.Conj(w[0])
	for k := 1; k < n; k++ {
		b[k] = cmplx.Conj(w[k])
		b[m-k] = b[k]
	}
	fft2(a, -1)
	fft2(b, -1)
	for i := range a {
		a[i] *= b[i]
	}
	fft2(a, 1)
	out := make([]complex128, n)
	for k := range n {
		out[k] = a[k] * w[k] / complex(float64(m), 0)
	}
	return out
}
