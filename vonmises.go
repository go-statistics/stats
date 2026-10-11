package stats

import (
	"fmt"
	"math"

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

func (d VonMises) std() vonmisesStd {
	s := vonmisesStd{k: d.Kappa}
	if s.valid() {
		s.i0e = special.I0e(s.k)
		s.c0 = dd.Div(dd.F(1), dd.MulF(twoPiDD, s.i0e))
	}
	return s
}

// vonmisesStd carries the concentration k, I0e(k) and the density's
// factor c0 = 1/(2π I0e(k)).
type vonmisesStd struct {
	k, i0e float64
	c0     dd.DD
}

func (s vonmisesStd) valid() bool             { return s.k >= 0 && !math.IsInf(s.k, 1) }
func (s vonmisesStd) support() (a, b float64) { return math.Inf(-1), math.Inf(1) }

// vmReduce returns r = z - 2πn in [-π, π] as a double-double, and n.
func vmReduce(z float64) (dd.DD, float64) {
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
	return expScaledR(dd.MulF(vmCosm1(r), s.k), s.c0)
}

func (s vonmisesStd) logpdf(z float64) float64 {
	r, _ := vmReduce(z)
	return dd.Sub(dd.MulF(vmCosm1(r), s.k), dd.Add(log2PiDD, dd.Log(s.i0e))).Float()
}

// tsRule is the tanh-sinh rule with step 1/16 on [0, 1]: nodes
// 1/(1 + exp(-π sinh(u))) and weights, u = j/16 for |j| <= tsK.
const tsK = 56

var tsNode, tsWeight [2*tsK + 1]float64

func init() {
	for j := -tsK; j <= tsK; j++ {
		u := float64(j) / 16
		sh := math.Pi / 2 * math.Sinh(u)
		ch := math.Cosh(sh)
		tsNode[j+tsK] = 1 / (1 + math.Exp(-2*sh))
		tsWeight[j+tsK] = (math.Pi / 2 * math.Cosh(u)) / (2 * ch * ch) / 16
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
// Δ(0) = 0: the fine rule up to where Δ reaches vmCut (found by
// bisection), a coarse one beyond, where the integrand is below e^-50.
func expIntegral(delta func(t float64) float64, l float64) float64 {
	f := func(t float64) float64 { return math.Exp(-delta(t)) }
	if !(delta(l) > vmCut) {
		return tanhSinh(f, 0, l, 1)
	}
	lo, hi := 0.0, l
	for i := 0; i < 64; i++ {
		m := (lo + hi) / 2
		if delta(m) < vmCut {
			lo = m
		} else {
			hi = m
		}
	}
	return tanhSinh(f, 0, hi, 1) + tanhSinh(f, hi, l-hi, 4)
}

// lowerTail returns P(X <= x) within the period, for x in [-π, 0].
func (s vonmisesStd) lowerTail(x dd.DD) dd.DD {
	l := dd.Add(x, piDD)
	if !(l.Hi > 0) {
		return dd.DD{}
	}
	if s.k == 0 {
		return dd.Div(l, twoPiDD)
	}
	xh := x.Hi
	k := s.k
	j := expIntegral(func(t float64) float64 {
		return 2 * k * math.Sin(t/2) * math.Sin(t/2-xh)
	}, l.Float())
	return dd.MulF(dd.Mul(expD(dd.MulF(vmCosm1(x), k)), s.c0), j)
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

// logcdf and logsf: within the first period through logPair; elsewhere
// the logarithm of a value beyond [0, 1] (NaN below 0, as SciPy).
func (s vonmisesStd) logcdf(z float64) float64 { return s.logOf(s.cdfD(z), s.sfD(z)) }
func (s vonmisesStd) logsf(z float64) float64  { return s.logOf(s.sfD(z), s.cdfD(z)) }

func (s vonmisesStd) logOf(p, q dd.DD) float64 {
	if p.Hi < 0 || q.Hi < 0 {
		return math.Log(p.Float())
	}
	return logPair(p, q)
}

// solveLower returns x in [-π, 0] with lowerTail(x) = p, 0 < p <= 1/2, by
// a safeguarded Newton iteration on log lowerTail from the normal
// approximation (large Kappa) or the uniform one.
func (s vonmisesStd) solveLower(p float64) float64 {
	if p == 0.5 {
		return 0
	}
	lo, hi := -math.Pi, 0.0
	x := -math.Pi + 2*math.Pi*p
	if s.k >= 2 {
		if v := special.Ndtri(p) * s.i0e / sqrt2OverPi.Hi; v > -1 {
			x = 2 * math.Asin(v)
		}
	}
	lp := math.Log(p)
	for i := 0; i < 100; i++ {
		f := s.lowerTail(dd.F(x))
		g := logD(f).Float() - lp
		if g > 0 {
			hi = x
		} else {
			lo = x
		}
		if g == 0 {
			return x
		}
		dx := -g * f.Float() / s.pdf(x)
		nx := x + dx
		if !(nx > lo && nx < hi) {
			nx = lo + (hi-lo)/2
		}
		if math.Abs(nx-x) <= 0x1p-60*math.Abs(x) || nx == x {
			return nx
		}
		x = nx
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

// integral returns ∫_0^π exp(k(cos θ - 1)) g(θ) dθ by the quadrature of
// lowerTail (Δ = 2k sin²(θ/2)).
func (s vonmisesStd) integral(g func(t float64) float64) float64 {
	k := s.k
	delta := func(t float64) float64 {
		sn := math.Sin(t / 2)
		return 2 * k * sn * sn
	}
	f := func(t float64) float64 { return math.Exp(-delta(t)) * g(t) }
	if !(delta(math.Pi) > vmCut) {
		return tanhSinh(f, 0, math.Pi, 1)
	}
	c := 2 * math.Asin(math.Sqrt(vmCut/(2*k)))
	return tanhSinh(f, 0, c, 1) + tanhSinh(f, c, math.Pi-c, 4)
}

// munp is E[X^n] over [-π, π]: 0 for odd n, the ratio of quadratures
// ∫ θ^n f / ∫ f for even n.
func (s vonmisesStd) munp(n int) float64 {
	if n%2 == 1 {
		return 0
	}
	fn := float64(n)
	return s.integral(func(t float64) float64 { return math.Pow(t, fn) }) /
		s.integral(func(float64) float64 { return 1 })
}

func (s vonmisesStd) mean() float64     { return 0 }
func (s vonmisesStd) variance() float64 { return s.munp(2) }
func (s vonmisesStd) skew() float64     { return 0 }
func (s vonmisesStd) kurtosis() float64 {
	v := s.munp(2)
	return s.munp(4)/(v*v) - 3
}
func (s vonmisesStd) median() float64 { return 0 }

// entropy is k(1 - I1(k)/I0(k)) + log(2π I0e(k)), with
// 1 - I1/I0 = ∫ (1 - cos θ) f / ∫ f by quadrature (SciPy's
// -k I1e/I0e + k cancels for large k).
func (s vonmisesStd) entropy() float64 {
	a := s.integral(func(t float64) float64 {
		sn := math.Sin(t / 2)
		return 2 * sn * sn
	}) / s.integral(func(float64) float64 { return 1 })
	return dd.Add(dd.F(s.k*a), dd.Add(log2PiDD, dd.Log(s.i0e))).Float()
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
