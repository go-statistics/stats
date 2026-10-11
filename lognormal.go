package stats

import (
	"fmt"
	"math"

	"github.com/go-ndarray/ndarray"
	"github.com/go-ndarray/ndarray/random"
	"github.com/go-numerics/special"
	"github.com/go-statistics/stats/internal/dd"
)

//stats:continuous LogNormal lognorm s=S

// LogNormal is the log-normal distribution: log X is normal with standard
// deviation S and mean log(Scale): density exp(-log²(x)/(2S²))/(S x √(2π))
// in standard form: scipy.stats.lognorm(s, loc, scale).
//
// z = log(x)/S is carried in double-double: the density takes its square
// exactly into a double-double exponent (SciPy's exp(-(log x/s)²/2)
// loses about log2(z²) bits), and CDF, SF and their logarithms add the
// first-order term of the low part of z to special.Ndtr and
// special.LogNdtr. PPF and ISF refine special.Ndtri by one Newton step in
// the tails before exp(S·z), whose relative error is S·z times that of z.
// The moments use expm1(S²), exact where SciPy's exp(S²) - 1 cancels for
// small S.
type LogNormal struct {
	S, Loc, Scale float64
}

func (d LogNormal) std() lognormStd { return lognormStd{s: d.S} }

type lognormStd struct{ s float64 }

func (s lognormStd) valid() bool               { return s.s > 0 && !math.IsInf(s.s, 1) }
func (s lognormStd) support() (lo, hi float64) { return 0, math.Inf(1) }

// z returns log(x)/s as a double-double and log x.
func (s lognormStd) z(x float64) (z, l dd.DD) {
	l = dd.Log(x)
	return dd.DivF(l, s.s), l
}

func (s lognormStd) pdf(x float64) float64 {
	if x == 0 {
		return 0
	}
	z, l := s.z(x)
	if math.Abs(z.Hi) > 1e150 {
		return 0
	}
	e := dd.Sub(dd.MulF(dd.Mul(z, z), -0.5), l)
	return expScaled(e, dd.DivF(invSqrt2Pi, s.s))
}

func (s lognormStd) logpdf(x float64) float64 {
	if x == 0 {
		return math.Inf(-1)
	}
	z, l := s.z(x)
	if math.Abs(z.Hi) > 1e150 {
		return math.Inf(-1)
	}
	e := dd.Sub(dd.MulF(dd.Mul(z, z), -0.5), l)
	return dd.Sub(e, dd.Add(dd.Log(s.s), logSqrt2Pi)).Float()
}

// millsInv returns φ(z)/Φ(z), the derivative of log Φ at z.
func millsInv(z float64) float64 {
	return math.Exp(-0.5*z*z - logSqrt2Pi.Hi - special.LogNdtr(z))
}

func (s lognormStd) cdf(x float64) float64 {
	z, _ := s.z(x)
	return special.Ndtr(z.Hi) + z.Lo*normalStd{}.pdf(z.Hi)
}

func (s lognormStd) sf(x float64) float64 {
	z, _ := s.z(x)
	return special.Ndtr(-z.Hi) - z.Lo*normalStd{}.pdf(z.Hi)
}

func (s lognormStd) logcdf(x float64) float64 {
	z, _ := s.z(x)
	return special.LogNdtr(z.Hi) + z.Lo*millsInv(z.Hi)
}

func (s lognormStd) logsf(x float64) float64 {
	z, _ := s.z(x)
	return special.LogNdtr(-z.Hi) - z.Lo*millsInv(-z.Hi)
}

// ndtriD returns the standard normal quantile of p (lower tail, or upper
// tail when upper is set: the quantile of 1-p) as a double-double: Ndtri,
// refined in the tails by one Newton step whose residual is formed in the
// tail that p measures, so that the low part is exact to about ε/z².
func ndtriD(p float64, upper bool) dd.DD {
	z := special.Ndtri(p)
	if upper {
		z = -z
	}
	if !(math.Abs(z) > 1 && math.Abs(z) < 37.5) {
		return dd.F(z)
	}
	phi := normalStd{}.pdf(z)
	var dz float64
	if upper {
		dz = (special.Ndtr(-z) - p) / phi
	} else {
		dz = -(special.Ndtr(z) - p) / phi
	}
	h, l := dd.TwoSum(z, dz)
	return dd.DD{Hi: h, Lo: l}
}

func (s lognormStd) ppf(p float64) float64 {
	if p > 0.5 {
		return dd.Exp(dd.MulF(ndtriD(1-p, true), s.s))
	}
	return dd.Exp(dd.MulF(ndtriD(p, false), s.s))
}

func (s lognormStd) isf(q float64) float64 {
	if q > 0.5 {
		return dd.Exp(dd.MulF(ndtriD(1-q, false), s.s))
	}
	return dd.Exp(dd.MulF(ndtriD(q, true), s.s))
}

// The moments with e = expm1(s²), p = 1 + e: mean exp(s²/2), variance
// p·e, skewness √e (2 + p), excess kurtosis p⁴ + 2p³ + 3p² - 6 expanded in
// e so that nothing cancels for small s.
func (s lognormStd) e() float64    { return expm1D(dd.MulFF(s.s, s.s)).Float() }
func (s lognormStd) mean() float64 { return dd.Exp(halfSquare(s.s)) }
func (s lognormStd) variance() float64 {
	sq := dd.MulFF(s.s, s.s)
	return dd.Mul(expD(sq), expm1D(sq)).Float()
}
func (s lognormStd) skew() float64 {
	e := s.e()
	return math.Sqrt(e) * (3 + e)
}
func (s lognormStd) kurtosis() float64 {
	e := s.e()
	return e * (16 + e*(15+e*(6+e)))
}
func (s lognormStd) median() float64 { return 1 }

// entropy is 1/2 + log(2π)/2 + log s.
func (s lognormStd) entropy() float64 {
	return dd.Add(dd.AddF(halfLog2Pi, 0.5), dd.Log(s.s)).Float()
}

// munp is exp(n² s²/2).
func (s lognormStd) munp(n int) float64 {
	fn := float64(n)
	return dd.Exp(dd.MulF(halfSquare(s.s), fn*fn))
}

// rvs is SciPy's exp(s·Z) for standard normal Z.
func (s lognormStd) rvs(g *random.Generator, size []int) (*ndarray.Array, error) {
	a, err := g.StandardNormal(size...)
	if err != nil {
		return nil, err
	}
	v, _ := ndarray.Data[float64](a)
	for i, z := range v {
		v[i] = crExp(float64(s.s * z))
	}
	return a, nil
}

// FitLogNormal returns the maximum-likelihood log-normal distribution for
// x: scipy.stats.lognorm.fit(x, fs=, floc=, fscale=). With Loc fixed, S
// and Scale have closed forms (the root mean square and the exponential
// of the mean of log(x - Loc)), reproduced to the bit; with Loc free, Loc
// solves the likelihood equation ∂/∂Loc = 0 with S and Scale substituted
// (Cohen and Whitten 1980), found as SciPy finds it (the same brackets)
// but to full precision, and kept only where its likelihood beats the
// limit Loc → min(x).
func FitLogNormal[D Data](x D, o *FitOptions) (LogNormal, error) {
	v := floats(x)
	fx, err := o.fixed("LogNormal", "S", "Loc", "Scale")
	if err != nil {
		return LogNormal{}, err
	}
	if err := checkFitData(v); err != nil {
		return LogNormal{}, err
	}
	fs, haveS := fx["S"]
	fscale, haveScale := fx["Scale"]
	lo, _ := minMax(v)
	shapeScale := func(loc float64) (float64, float64) {
		var lnd []float64
		if !haveS || !haveScale {
			lnd = make([]float64, len(v))
			for i, t := range v {
				lnd[i] = crLog(t - loc)
			}
		}
		scale, shape := fscale, fs
		if !haveScale {
			scale = crExp(mean(lnd))
		}
		if !haveS {
			ls := crLog(scale)
			for i := range lnd {
				d := lnd[i] - ls
				lnd[i] = float64(d * d)
			}
			shape = math.Sqrt(mean(lnd))
		}
		return shape, scale
	}
	dl := func(loc float64) float64 {
		shape, scale := shapeScale(loc)
		t := make([]float64, len(v))
		for i, y := range v {
			sh := y - loc
			t[i] = (1 + crLog(sh/scale)/(shape*shape)) / sh
		}
		return sum(t)
	}
	ll := func(loc float64) float64 {
		shape, scale := shapeScale(loc)
		return contLogLik(v, lognormStd{s: shape}, loc, scale)
	}
	fallback := func() (LogNormal, error) {
		sh, loc, scale, err := fitContinuous(v, fx, []string{"S"}, []bool{true}, []float64{1},
			func(s []float64) contStd { return lognormStd{s: s[0]} })
		if err != nil {
			return LogNormal{}, err
		}
		d := LogNormal{S: sh[0], Loc: loc, Scale: scale}
		return d, d.Validate()
	}
	loc, haveLoc := fx["Loc"]
	if haveLoc {
		if loc >= lo {
			return LogNormal{}, fmt.Errorf("%w: log-normal fit with Loc=%v needs every value above it", ErrData, loc)
		}
	} else {
		spacing := math.Nextafter(lo, math.Copysign(math.Inf(1), lo)) - lo
		rb := lo - spacing
		dr := dl(rb)
		llr := ll(rb)
		delta := 2 * spacing
		for dr >= -1e-6 && finite(rb) && finite(dr) {
			rb = lo - delta
			dr = dl(rb)
			delta *= 2
		}
		if !finite(rb) || !finite(dr) {
			return fallback()
		}
		lb := math.Min(math.Nextafter(rb, math.Inf(-1)), rb-1)
		dlb := dl(lb)
		delta = 2 * (rb - lb)
		for finite(lb) && finite(dlb) && math.Signbit(dlb) == math.Signbit(dr) {
			lb = rb - delta
			dlb = dl(lb)
			delta *= 2
		}
		if !finite(lb) || !finite(dlb) {
			return fallback()
		}
		root, ok := brentRoot(dl, lb, rb)
		if !ok {
			return fallback()
		}
		loc = lo - spacing
		if ll(root) > llr {
			loc = root
		}
	}
	shape, scale := shapeScale(loc)
	d := LogNormal{S: shape, Loc: loc, Scale: scale}
	if d.Validate() != nil {
		return fallback()
	}
	return d, nil
}
