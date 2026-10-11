package stats

import (
	"math"

	"github.com/go-ndarray/ndarray"
	"github.com/go-ndarray/ndarray/random"
	"github.com/go-numerics/special"
	"github.com/go-statistics/stats/internal/dd"
)

//stats:continuous StudentT t df=Df

// StudentT is Student's t distribution with Df degrees of freedom (Df may
// be +Inf, the normal distribution): scipy.stats.t(df, loc, scale).
//
// The distribution functions go through the incomplete beta function as in
// Boost (I_z(1/2, ν/2) or I_z(ν/2, 1/2), whichever argument is not close to
// 1), the quantile through its inverse with both z and 1-z obtained without
// cancellation. The density is c(ν)·(1+t²/ν)^(-(ν+1)/2) with the power in
// double-double and c(ν) from the gamma ratio (an asymptotic series for
// ν >= 100).
type StudentT struct {
	Df, Loc, Scale float64
}

func (d StudentT) std() tStd { return tStd{df: d.Df} }

type tStd struct{ df float64 }

func (s tStd) valid() bool               { return s.df > 0 }
func (s tStd) support() (lo, hi float64) { return math.Inf(-1), math.Inf(1) }

// tConst returns c(ν) = Γ((ν+1)/2) / (Γ(ν/2) √(νπ)) as a double-double.
func tConst(nu float64) dd.DD {
	return dd.Div(gammaRatioHalf(nu/2), dd.Sqrt(dd.Mul(dd.F(nu), dd.DD{Hi: math.Pi, Lo: 1.2246467991473532e-16})))
}

// tLogKernel returns -(ν+1)/2 · log(1 + t²/ν) as a double-double.
func tLogKernel(t, nu float64) dd.DD {
	t2h, t2l := dd.TwoProd(t, t)
	u := dd.DivF(dd.DD{Hi: t2h, Lo: t2l}, nu)
	nh, nl := dd.TwoSum(nu, 1)
	return dd.Neg(dd.Mul(log1pDD(u), dd.DD{Hi: nh * 0.5, Lo: nl * 0.5}))
}

func (s tStd) pdf(t float64) float64 {
	if math.IsInf(s.df, 1) {
		return normalStd{}.pdf(t)
	}
	return expScaled(tLogKernel(t, s.df), tConst(s.df))
}

func (s tStd) logpdf(t float64) float64 {
	if math.IsInf(s.df, 1) {
		return normalStd{}.logpdf(t)
	}
	return dd.Add(tLogKernel(t, s.df), logDD(tConst(s.df))).Float()
}

// tail returns P(T < -|t|) (Boost's students_t cdf): with z = t²/(ν+t²),
// ½·I^c_z(1/2, ν/2) when ν > 2t², else ½·I_{1-z}(ν/2, 1/2). The argument
// is formed in double-double and the rounding of its float64 value is
// corrected to first order with the beta density (dI_x/dx), which matters
// in the tails where the result is ν/2 times as sensitive to x as x is
// rounded.
func (s tStd) tail(t float64) float64 {
	nu := s.df
	h, l := dd.TwoProd(t, t)
	t2 := dd.DD{Hi: h, Lo: l}
	den := dd.AddF(t2, nu)
	z := dd.Div(t2, den)       // t²/(ν+t²)
	w := dd.Div(dd.F(nu), den) // ν/(ν+t²)
	if nu > 2*h {
		p := special.BetaIncC(0.5, nu/2, z.Hi)
		if z.Lo != 0 {
			p -= betaDensity(0.5, nu/2, z.Hi, w.Hi, dd.F(z.Lo))
		}
		return p / 2
	}
	p := special.BetaInc(nu/2, 0.5, w.Hi)
	return (p + betaDensity(nu/2, 0.5, w.Hi, z.Hi, dd.F(w.Lo))) / 2
}

func (s tStd) cdf(t float64) float64 {
	if math.IsInf(s.df, 1) {
		return special.Ndtr(t)
	}
	if t > 0 {
		return 1 - s.tail(t)
	}
	return s.tail(t)
}

func (s tStd) sf(t float64) float64 { return s.cdf(-t) }

func (s tStd) logcdf(t float64) float64 {
	if math.IsInf(s.df, 1) {
		return special.LogNdtr(t)
	}
	if t > 0 {
		return math.Log1p(-s.tail(t))
	}
	return math.Log(s.tail(t))
}

func (s tStd) logsf(t float64) float64 { return s.logcdf(-t) }

// lowerQuantile returns the t < 0 with P(T < t) = p, for 0 < p <= 1/2: from
// z = I^-1_{2p}(ν/2, 1/2) = ν/(ν+t²) when z is small, from
// x = 1-z = (I_c)^-1_{2p}(1/2, ν/2) when z is not, so that t² = ν x/z never
// takes a difference of nearly equal numbers.
func (s tStd) lowerQuantile(p float64) float64 {
	nu := s.df
	z := special.BetaIncInv(nu/2, 0.5, 2*p)
	var x float64
	if z < 0.5 {
		x = 1 - z
	} else {
		x = special.BetaIncCInv(0.5, nu/2, 2*p)
		z = 1 - x
	}
	return -math.Sqrt(nu * x / z)
}

func (s tStd) ppf(p float64) float64 {
	switch {
	case math.IsInf(s.df, 1):
		return special.Ndtri(p)
	case p <= 0.5:
		return s.lowerQuantile(p)
	}
	return -s.lowerQuantile(1 - p) // 1-p exact for p >= 1/2
}

func (s tStd) isf(q float64) float64 { return -s.ppf(q) }

func (s tStd) mean() float64 {
	if s.df > 1 {
		return 0
	}
	return math.Inf(1)
}

func (s tStd) variance() float64 {
	switch {
	case math.IsInf(s.df, 1):
		return 1
	case s.df > 2:
		return s.df / (s.df - 2)
	case s.df > 1:
		return math.Inf(1)
	}
	return math.NaN()
}

func (s tStd) skew() float64 {
	if s.df > 3 {
		return 0
	}
	return math.NaN()
}

func (s tStd) kurtosis() float64 {
	switch {
	case math.IsInf(s.df, 1):
		return 0
	case s.df > 4:
		return 6 / (s.df - 4)
	case s.df > 2:
		return math.Inf(1)
	}
	return math.NaN()
}

func (s tStd) median() float64 { return 0 }

// entropy is SciPy's formula, asymptotic from ν = 100.
func (s tStd) entropy() float64 {
	nu := s.df
	if nu >= 100 {
		return 1.4189385332046727 + 1/nu + math.Pow(nu, -2)/4 - math.Pow(nu, -3)/6 - math.Pow(nu, -4)/8 + 3.0/10*math.Pow(nu, -5) + math.Pow(nu, -6)/4
	}
	half, half1 := nu/2, (nu+1)/2
	return half1*(special.Digamma(half1)-special.Digamma(half)) + math.Log(math.Sqrt(nu)*special.Beta(half, 0.5))
}

// munp: E[T^n] is 0 for odd n < ν and ν^(n/2) Γ((n+1)/2) Γ((ν-n)/2) /
// (√π Γ(ν/2)) for even n < ν; it does not exist (NaN for odd, +Inf for even
// n) from n >= ν.
func (s tStd) munp(n int) float64 {
	nu := s.df
	fn := float64(n)
	if math.IsInf(nu, 1) {
		return normalStd{}.munp(n)
	}
	if fn >= nu {
		if n%2 == 1 {
			return math.NaN()
		}
		return math.Inf(1)
	}
	if n%2 == 1 {
		return 0
	}
	lg := fn/2*math.Log(nu) + special.LogGamma((fn+1)/2) + special.LogGamma((nu-fn)/2) - 0.5*math.Log(math.Pi) - special.LogGamma(nu/2)
	return math.Exp(lg)
}

func (s tStd) rvs(g *random.Generator, size []int) (*ndarray.Array, error) {
	return g.StandardT(s.df, size...)
}

// FitStudentT returns the maximum-likelihood t distribution for x:
// scipy.stats.t.fit(x, fdf=, floc=, fscale=), numerically from SciPy's
// starting point (Df = 1).
func FitStudentT[D Data](x D, o *FitOptions) (StudentT, error) {
	v := floats(x)
	fx, err := o.fixed("StudentT", "Df", "Loc", "Scale")
	if err != nil {
		return StudentT{}, err
	}
	if err := checkFitData(v); err != nil {
		return StudentT{}, err
	}
	sh, loc, scale, err := fitContinuous(v, fx, []string{"Df"}, []bool{true}, []float64{1},
		func(s []float64) contStd { return tStd{df: s[0]} })
	if err != nil {
		return StudentT{}, err
	}
	d := StudentT{Df: sh[0], Loc: loc, Scale: scale}
	return d, d.Validate()
}
