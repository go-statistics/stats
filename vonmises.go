package stats

import (
	"fmt"
	"math"
	"sync/atomic"

	"github.com/go-ndarray/ndarray"
	"github.com/go-ndarray/ndarray/random"
	"github.com/go-numerics/special"
	"github.com/go-statistics/stats/internal/dd"
)

//stats:continuous VonMises vonmises kappa=Kappa

// VonMises is the von Mises (circular normal) distribution with
// concentration Kappa >= 0 and mean direction Loc: density
// exp(Kappa cos x)/(2π I0(Kappa)) in standard form:
// scipy.stats.vonmises(kappa, loc, scale).
//
// As SciPy's vonmises, it is a circular distribution laid on the real
// line: the density is 2π-periodic, the support is the whole line, and
// CDF counts the turns, CDF(x + 2π) = CDF(x) + 1, so that it lies in
// [0, 1] only on [Loc - π, Loc + π]; PPF and ISF return points of that
// interval, and the moments, as SciPy's, are those of the interval.
// Draws are wrapped onto [-π, π] after Loc and Scale are applied, as
// SciPy wraps them. (SciPy's vonmises_line, the same density restricted
// to [-π, π], is not provided.)
//
// The density carries cos(x) - 1 = -2 sin²(x/2) in double-double. SciPy
// computes the CDF by a series truncated for 12 significant digits (and a
// normal approximation from Kappa = 50); here the tail probability
// P(X <= x) for x in [-π, 0] is the density at x times
// ∫_0^{x+π} exp(-Δ(τ)) dτ, Δ(τ) = 2 Kappa sin(τ/2) sin(τ/2 - x) >= 0 the
// exact difference of the exponents, by tanh-sinh quadrature: full
// relative accuracy out to the far tail, at the cost of about 250
// exponentials per evaluation. PPF solves the CDF by Newton's method on
// its logarithm.
type VonMises struct {
	Kappa, Loc, Scale float64
}

func (d VonMises) std() vonmisesStd { return vonmisesStd{k: d.Kappa} }

type vonmisesStd struct{ k float64 }

// c0 is the density's factor 1/(2π I0e(k)) (the density being
// c0·exp(k(cos x - 1))).
func (s vonmisesStd) c0() dd.DD { return dd.Div(dd.F(1), dd.MulF(twoPiDD, s.i0e())) }

// vmI0e caches the last I0e(k) computed: special.I0e costs microseconds,
// and a distribution is usually evaluated many times at one Kappa.
var vmI0e atomic.Pointer[[2]float64]

// i0e returns I0e(k), from the cache when k is the last one asked for.
func (s vonmisesStd) i0e() float64 {
	if c := vmI0e.Load(); c != nil && c[0] == s.k {
		return c[1]
	}
	v := special.I0e(s.k)
	vmI0e.Store(&[2]float64{s.k, v})
	return v
}

func (s vonmisesStd) valid() bool             { return s.k >= 0 && !math.IsInf(s.k, 1) }
func (s vonmisesStd) support() (a, b float64) { return math.Inf(-1), math.Inf(1) }

// vmReduce returns r = z - 2πn in [-π, π] as a double-double, and n:
// with 2π in double-double below 2^20, to 1280 bits above
// (reduceTwoPiHuge), where the double-double product 2πn no longer
// carries enough bits for the remainder.
func vmReduce(z float64) (dd.DD, float64) {
	if reduceTwoPiHugeOK(z) {
		return reduceTwoPiHuge(z)
	}
	n := math.RoundToEven(z / twoPiDD.Hi)
	r := dd.Sub(dd.F(z), dd.MulF(twoPiDD, n))
	switch {
	case r.Hi > math.Pi:
		r = dd.Sub(r, twoPiDD)
		n++
	case r.Hi < -math.Pi:
		r = dd.Add(r, twoPiDD)
		n--
	}
	return r, n
}

// vmCosm1 returns cos(r) - 1 = -2 sin²(r/2) for |r| <= π.
func vmCosm1(r dd.DD) dd.DD {
	h := dd.MulF(r, 0.5)
	if h.Hi < 0 {
		h = dd.Neg(h)
	}
	sn, _ := sincosD(h)
	return dd.MulF(dd.Mul(sn, sn), -2)
}

func (s vonmisesStd) pdf(z float64) float64 {
	r, _ := vmReduce(z)
	return expScaledR(dd.MulF(vmCosm1(r), s.k), s.c0())
}

func (s vonmisesStd) logpdf(z float64) float64 {
	r, _ := vmReduce(z)
	return dd.Sub(dd.MulF(vmCosm1(r), s.k), dd.Add(log2PiDD, dd.Log(s.i0e()))).Float()
}

// tsRule is the tanh-sinh rule with step 1/16 on [0, 1]: nodes
// 1/(1 + exp(-π sinh(u))) and weights, u = j/16 for |j| <= tsK.
const tsK = 56

var tsNode, tsWeight [2*tsK + 1]float64

// tsNode2 and tsWeight2 are the same rule with step 1/64, for the
// moments, where the kurtosis needs the integrals to well below 2^-53.
var tsNode2, tsWeight2 [8*tsK + 1]float64

func init() {
	tsRule(tsNode[:], tsWeight[:], 16)
	tsRule(tsNode2[:], tsWeight2[:], 64)
}

// tsRule fills the tanh-sinh nodes and weights on [0, 1] with step 1/m.
func tsRule(node, weight []float64, m float64) {
	k := len(node) / 2
	for j := -k; j <= k; j++ {
		u := float64(j) / m
		sh := math.Pi / 2 * math.Sinh(u)
		ch := math.Cosh(sh)
		node[j+k] = 1 / (1 + math.Exp(-2*sh))
		weight[j+k] = (math.Pi / 2 * math.Cosh(u)) / (2 * ch * ch) / m
	}
}

// tanhSinh integrates f over [a, a+w] with the rule above (every stride-th
// node, for a coarser rule), summing in double-double.
func tanhSinh(f func(t float64) float64, a, w float64, stride int) float64 {
	var sh, sl float64
	for i := 0; i < len(tsNode); i += stride {
		v := tsWeight[i] * f(a+w*tsNode[i])
		t, e := dd.TwoSum(sh, v)
		sh, sl = t, sl+e
	}
	return (sh + sl) * w * float64(stride)
}

// vmCut is the exponent beyond which the integrands are negligible
// (e^-50 against a maximum of 1), where the quadrature splits its range.
const vmCut = 50

// expIntegral returns ∫_0^L exp(-Δ(τ)) dτ for an increasing Δ >= 0 with
// Δ(0) = 0: the fine rule up to about where Δ reaches vmCut (found by a
// short bisection: the split need not be exact), a coarse one beyond,
// where the integrand is below e^-40.
func expIntegral(delta func(t float64) float64, l float64) float64 {
	f := func(t float64) float64 { return math.Exp(-delta(t)) }
	if !(delta(l) > vmCut) {
		return tanhSinh(f, 0, l, 1)
	}
	lo, hi := 0.0, l
	for i := 0; i < 16; i++ {
		m := (lo + hi) / 2
		if delta(m) < vmCut {
			lo = m
		} else {
			hi = m
		}
	}
	return tanhSinh(f, 0, hi, 1) + tanhSinh(f, hi, l-hi, 4)
}

// lowerTail returns P(X <= x) within the period, for x in [-π, 0], and
// logLowerTail its logarithm (exact also where P underflows).
func (s vonmisesStd) lowerTail(x dd.DD) dd.DD {
	e, j := s.lowerParts(x)
	if j == 0 {
		return dd.DD{}
	}
	return dd.MulF(dd.Mul(expD(e), s.c0()), j)
}

func (s vonmisesStd) logLowerTail(x dd.DD) float64 {
	e, j := s.lowerParts(x)
	if j == 0 {
		return math.Inf(-1)
	}
	return dd.Sub(dd.Add(e, dd.Log(j)), dd.Add(log2PiDD, dd.Log(s.i0e()))).Float()
}

// lowerParts returns E and J with P(X <= x) = c0·exp(E)·J: E = k(cos x - 1)
// and J = ∫_0^{x+π} exp(-Δ(τ)) dτ.
func (s vonmisesStd) lowerParts(x dd.DD) (dd.DD, float64) {
	l := dd.Add(x, piDD)
	if !(l.Hi > 0) {
		return dd.DD{}, 0
	}
	if s.k == 0 {
		return dd.DD{}, l.Float()
	}
	// sin(τ/2 - x) = sin(τ/2) cos x - cos(τ/2) sin x, with sin x and cos x
	// from x + π where x is near -π (exact there in double-double)
	var sx, cx float64
	if l.Hi < math.Pi/2 {
		sx, cx = math.Sincos(l.Float())
		sx, cx = -sx, -cx
	} else {
		sx, cx = math.Sincos(x.Hi)
	}
	k := s.k
	j := expIntegral(func(t float64) float64 {
		st, ct := math.Sincos(t / 2)
		return 2 * k * st * (st*cx - ct*sx)
	}, l.Float())
	return dd.MulF(vmCosm1(x), k), j
}

// f0 returns the CDF within the period, for r in [-π, π].
func (s vonmisesStd) f0(r dd.DD) dd.DD {
	if r.Hi <= 0 {
		return s.lowerTail(r)
	}
	return dd.Sub(dd.F(1), s.lowerTail(dd.Neg(r)))
}

func (s vonmisesStd) cdfD(z float64) dd.DD {
	r, n := vmReduce(z)
	return dd.AddF(s.f0(r), n)
}

func (s vonmisesStd) sfD(z float64) dd.DD {
	r, n := vmReduce(z)
	return dd.AddF(s.f0(dd.Neg(r)), -n)
}

func (s vonmisesStd) cdf(z float64) float64 { return s.cdfD(z).Float() }
func (s vonmisesStd) sf(z float64) float64  { return s.sfD(z).Float() }

// logcdf and logsf: within the first period, the logarithm of the
// smaller tail in log space (it underflows long before its logarithm
// does at large Kappa) and log1p of minus it for the larger; elsewhere
// the logarithm of a value beyond [0, 1] (NaN below 0, as SciPy).
func (s vonmisesStd) logcdf(z float64) float64 {
	r, n := vmReduce(z)
	return s.logF0(r, n)
}

func (s vonmisesStd) logsf(z float64) float64 {
	r, n := vmReduce(z)
	return s.logF0(dd.Neg(r), -n)
}

// logF0 returns log(F0(r) + n).
func (s vonmisesStd) logF0(r dd.DD, n float64) float64 {
	switch {
	case n != 0:
		return math.Log(dd.AddF(s.f0(r), n).Float())
	case r.Hi <= 0:
		return s.logLowerTail(r)
	}
	return log1pDD(dd.Neg(s.lowerTail(dd.Neg(r)))).Float()
}

// solveLower returns x in [-π, 0] with lowerTail(x) = p, 0 < p <= 1/2:
// below 1/4 by Newton's method on log lowerTail from the normal
// approximation (large Kappa) or the uniform one; above, as minus the
// point y with ∫_0^y f = 1/2 - p (exact), so that a quantile near 0 keeps
// its relative accuracy.
func (s vonmisesStd) solveLower(p float64) float64 {
	if p >= 0.25 {
		return -s.solveCentral(0.5 - p)
	}
	x := -math.Pi + 2*math.Pi*p
	if s.k >= 2 {
		if v := special.Ndtri(p) * s.i0e() / sqrt2OverPi.Hi; v > -1 {
			x = 2 * math.Asin(v)
		}
	}
	return newtonLog(func(x float64) float64 { return s.logLowerTail(dd.F(x)) }, s.logpdf, math.Log(p), -math.Pi, 0, x)
}

// centralLog returns log ∫_0^y f for 0 < y <= π.
func (s vonmisesStd) centralLog(y float64) float64 {
	if y == 0 {
		return math.Inf(-1)
	}
	k := s.k
	j := y
	if k > 0 {
		j = expIntegral(func(t float64) float64 {
			sn := math.Sin(t / 2)
			return 2 * k * sn * sn
		}, y)
	}
	return dd.Sub(dd.Log(j), dd.Add(log2PiDD, dd.Log(s.i0e()))).Float()
}

// solveCentral returns y in [0, π] with ∫_0^y f = q, 0 < q <= 1/4.
func (s vonmisesStd) solveCentral(q float64) float64 {
	if q == 0 {
		return 0
	}
	y := min(q*2*math.Pi*s.i0e(), math.Pi/2) // f(0) = 1/(2π I0e(k))
	return newtonLog(s.centralLog, s.logpdf, math.Log(q), 0, math.Pi, y)
}

// newtonLog solves logF(x) = lp for an increasing F on [lo, hi] by a
// safeguarded Newton iteration on the logarithm (the step is
// (logF - lp)·F/f, F/f = exp(logF - logf)), then steps to the
// neighbouring float while that is closer.
func newtonLog(logF, logf func(float64) float64, lp, lo, hi, x float64) float64 {
	for i := 0; i < 100; i++ {
		lf := logF(x)
		gx := lf - lp
		if gx > 0 {
			hi = x
		} else {
			lo = x
		}
		nx := x - gx*math.Exp(lf-logf(x))
		if !(nx >= lo && nx <= hi) {
			nx = lo + (hi-lo)/2
		}
		if math.Abs(nx-x) <= 4*0x1p-52*math.Abs(x) {
			x = nx
			break
		}
		x = nx
	}
	gx := logF(x) - lp
	for i := 0; i < 8 && gx != 0; i++ {
		y := math.Nextafter(x, math.Copysign(math.Inf(1), -gx))
		gy := logF(y) - lp
		if !(math.Abs(gy) < math.Abs(gx)) {
			break
		}
		x, gx = y, gy
	}
	return x
}

func (s vonmisesStd) ppf(p float64) float64 {
	if p > 0.5 {
		return -s.solveLower(1 - p)
	}
	return s.solveLower(p)
}

func (s vonmisesStd) isf(q float64) float64 { return -s.ppf(q) }

// moments returns ∫_0^π exp(k(cos θ - 1)) θ^n dθ for n = 0, 2, 4 and
// ∫_0^π exp(k(cos θ - 1)) (1 - cos θ) dθ, by the quadrature of lowerTail
// carried in double-double throughout (integrand, powers, products and
// sums), so that the kurtosis, m0 m4/m2² - 3, keeps its digits where it
// is small (at k = 100 it is 0.0102: the difference cancels 8 bits).
func (s vonmisesStd) moments() (m0, m2, m4, c dd.DD) {
	k := s.k
	add := func(lo, w float64, stride int) {
		for i := 0; i < len(tsNode2); i += stride {
			t := lo + w*tsNode2[i]
			sn, _ := sincosD(dd.F(t / 2))
			om := dd.MulF(dd.Mul(sn, sn), 2) // 1 - cos t
			f := dd.MulF(expD(dd.MulF(om, -k)), tsWeight2[i]*w*float64(stride))
			t2 := dd.MulFF(t, t)
			m0 = dd.Add(m0, f)
			f2 := dd.Mul(f, t2)
			m2 = dd.Add(m2, f2)
			m4 = dd.Add(m4, dd.Mul(f2, t2))
			c = dd.Add(c, dd.Mul(f, om))
		}
	}
	if !(2*k > vmCut) {
		add(0, math.Pi, 1)
		return
	}
	cut := 2 * math.Asin(math.Sqrt(vmCut/(2*k)))
	add(0, cut, 1)
	add(cut, math.Pi-cut, 4)
	return
}

// munp is E[X^n] over [-π, π]: 0 for odd n, the ratio of quadratures
// ∫ θ^n f / ∫ f for even n (n = 2 and 4 in double-double).
func (s vonmisesStd) munp(n int) float64 {
	m0, m2, m4, _ := s.moments()
	switch {
	case n%2 == 1:
		return 0
	case n == 2:
		return dd.Div(m2, m0).Float()
	case n == 4:
		return dd.Div(m4, m0).Float()
	}
	k, fn := s.k, float64(n)
	var num, den float64
	f := func(lo, w float64, stride int) {
		for i := 0; i < len(tsNode); i += stride {
			t := lo + w*tsNode[i]
			sn := math.Sin(t / 2)
			e := math.Exp(-2*k*sn*sn) * tsWeight[i] * w * float64(stride)
			num += e * math.Pow(t, fn)
			den += e
		}
	}
	if !(2*k > vmCut) {
		f(0, math.Pi, 1)
	} else {
		cut := 2 * math.Asin(math.Sqrt(vmCut/(2*k)))
		f(0, cut, 1)
		f(cut, math.Pi-cut, 4)
	}
	return num / den
}

func (s vonmisesStd) mean() float64     { return 0 }
func (s vonmisesStd) variance() float64 { return s.munp(2) }
func (s vonmisesStd) skew() float64     { return 0 }
func (s vonmisesStd) kurtosis() float64 {
	m0, m2, m4, _ := s.moments()
	return dd.AddF(dd.Div(dd.Mul(m4, m0), dd.Mul(m2, m2)), -3).Float()
}
func (s vonmisesStd) median() float64 { return 0 }

// entropy is k(1 - I1(k)/I0(k)) + log(2π I0e(k)), with
// 1 - I1/I0 = ∫ (1 - cos θ) f / ∫ f by quadrature (SciPy's
// -k I1e/I0e + k cancels for large k).
func (s vonmisesStd) entropy() float64 {
	m0, _, _, c := s.moments()
	return dd.Add(dd.MulF(dd.Div(c, m0), s.k), dd.Add(log2PiDD, dd.Log(s.i0e()))).Float()
}

// rvs draws standard variates as SciPy's _rvs (numpy's vonmises(0, k));
// VonMises.Rvs then wraps them.
func (s vonmisesStd) rvs(g *random.Generator, size []int) (*ndarray.Array, error) {
	return g.VonMises(0, s.k, size...)
}

// Rvs draws random variates of the given shape:
// scipy.stats.vonmises.rvs(kappa, loc, scale, size=size, random_state=g):
// numpy's von Mises draws, moved by Loc and Scale, then wrapped onto
// [-π, π) as SciPy wraps them (np.mod(x + π, 2π) - π).
func (d VonMises) Rvs(g *random.Generator, size ...int) (*ndarray.Array, error) {
	if err := d.Validate(); err != nil {
		return nil, err
	}
	a, err := d.std().rvs(g, size)
	if err != nil {
		return nil, err
	}
	locScaleDraws(a, d.Loc, d.Scale)
	v, _ := ndarray.Data[float64](a)
	for i, x := range v {
		v[i] = npMod(x+math.Pi, 2*math.Pi) - math.Pi
	}
	return a, nil
}

// npMod is NumPy's floating remainder np.mod(a, b), b > 0: fmod moved
// into [0, b).
func npMod(a, b float64) float64 {
	m := math.Mod(a, b)
	switch {
	case m == 0:
		return 0
	case m < 0:
		return m + b
	}
	return m
}

// FitVonMises returns the maximum-likelihood von Mises distribution for x,
// read as angles: scipy.stats.vonmises.fit(x, fkappa=, floc=). Loc is the
// circular mean and Kappa solves I1(Kappa)/I0(Kappa) = mean cos(x - Loc)
// (zero if that is not positive), bracketed as SciPy brackets it but
// found to full precision; Loc is returned in [-π, π) and Scale is 1 (as
// SciPy, which ignores a fixed scale; fixing Scale to anything but 1 is
// an error here).
func FitVonMises[D Data](x D, o *FitOptions) (VonMises, error) {
	v := floats(x)
	fx, err := o.fixed("VonMises", "Kappa", "Loc", "Scale")
	if err != nil {
		return VonMises{}, err
	}
	if err := checkFitData(v); err != nil {
		return VonMises{}, err
	}
	if sc, ok := fx["Scale"]; ok && sc != 1 {
		return VonMises{}, fmt.Errorf("%w: the von Mises fit has Scale 1; it cannot be fixed to %v", ErrParam, sc)
	}
	w := make([]float64, len(v))
	for i, t := range v {
		w[i] = npMod(t, 2*math.Pi)
	}
	t := make([]float64, len(v))
	loc, ok := fx["Loc"]
	if !ok {
		c := make([]float64, len(v))
		for i, a := range w {
			t[i], c[i] = math.Sincos(a)
		}
		loc = npMod(math.Atan2(sum(t), sum(c)), 2*math.Pi)
	}
	kappa, ok := fx["Kappa"]
	if !ok {
		for i, a := range w {
			t[i] = math.Cos(loc - a)
		}
		kappa = vmKappa(sum(t) / float64(len(v)))
	}
	d := VonMises{Kappa: kappa, Loc: npMod(loc+math.Pi, 2*math.Pi) - math.Pi, Scale: 1}
	return d, d.Validate()
}

// vmKappa solves I1(k)/I0(k) = r for k, as SciPy's fit: 1e16 for r = 1,
// the smallest normal float64 for r <= 0, else the root between Amos's
// bounds r/(1-r²) and twice that (or the bound itself where the ratio
// computed at it already crosses r).
func vmKappa(r float64) float64 {
	switch {
	case r == 1:
		return 1e16
	case !(r > 0):
		return 0x1p-1022
	}
	f := func(k float64) float64 { return special.I1e(k)/special.I0e(k) - r }
	lb := r / (1 - r) / (1 + r)
	ub := 2 * lb
	if f(lb) >= 0 {
		return lb
	}
	if f(ub) <= 0 {
		return ub
	}
	k, _ := brentRoot(f, lb, ub)
	return k
}
