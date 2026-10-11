package stats

import (
	"math"

	"github.com/go-ndarray/ndarray"
	"github.com/go-ndarray/ndarray/random"
	"github.com/go-numerics/special"
	"github.com/go-statistics/stats/internal/dd"
)

//stats:continuous F f dfn=Dfn dfd=Dfd

// F is Fisher's F distribution (the variance ratio) with Dfn and Dfd
// degrees of freedom: scipy.stats.f(dfn, dfd, loc, scale).
//
// With v = Dfn·x, the distribution functions are I_{v/(Dfd+v)}(Dfn/2,
// Dfd/2) or its complement through I_{Dfd/(Dfd+v)}(Dfd/2, Dfn/2), the
// argument chosen below 1/2 as Boost does; the quantile inverts the
// incomplete beta function for x or 1-x, whichever is small, and the
// density is the beta density at the same argument times the Jacobian.
type F struct {
	Dfn, Dfd, Loc, Scale float64
}

func (d F) std() fStd { return fStd{m: d.Dfn, n: d.Dfd} }

type fStd struct{ m, n float64 }

func (s fStd) valid() bool {
	return s.m > 0 && s.n > 0 && !math.IsInf(s.m, 1) && !math.IsInf(s.n, 1)
}
func (s fStd) support() (lo, hi float64) { return 0, math.Inf(1) }

// args returns z = v/(n+v) and w = n/(n+v) = 1-z for v = m·x, as
// double-doubles (v exact through TwoProd), so that the distribution
// functions can correct the rounding of their float64 argument.
func (s fStd) args(x float64) (z, w dd.DD) {
	vh, vl := dd.TwoProd(s.m, x)
	v := dd.DD{Hi: vh, Lo: vl}
	den := dd.AddF(v, s.n)
	return dd.Div(v, den), dd.Div(dd.F(s.n), den)
}

// ibeta returns I_x(a, b) (or its complement when upper) at the
// double-double x, y = 1-x: the incomplete function at x.Hi plus the first
// order correction density·x.Lo.
func ibeta(a, b float64, x, y dd.DD, upper bool) float64 {
	if upper {
		p := special.BetaIncC(a, b, x.Hi)
		if x.Lo != 0 {
			p -= betaDensity(a, b, dd.F(x.Hi), dd.AddF(y, x.Lo), dd.F(x.Lo))
		}
		return p
	}
	p := special.BetaInc(a, b, x.Hi)
	if x.Lo != 0 {
		p += betaDensity(a, b, dd.F(x.Hi), dd.AddF(y, x.Lo), dd.F(x.Lo))
	}
	return p
}

func (s fStd) pdf(x float64) float64 {
	if x == 0 { // f ∝ x^(m/2-1); at m = 2, f(0) = (2/n)/B(1, n/2) = 1
		return gammaAtZero(s.m/2, 1)
	}
	z, w := s.args(x)
	// f(x) = beta_pdf(z; m/2, n/2) · dz/dx, dz/dx = m·n/(n+v)² = z·w/x.
	return betaDensity(s.m/2, s.n/2, z, w, dd.Div(dd.Mul(z, w), dd.F(x)))
}

func (s fStd) logpdf(x float64) float64 {
	if x == 0 {
		return math.Log(s.pdf(0))
	}
	z, w := s.args(x)
	l := betaLogDensity(s.m/2, s.n/2, z, w)
	return dd.Add(l, dd.Sub(dd.Add(logDD(z), logDD(w)), dd.Log(x))).Float()
}

func (s fStd) cdf(x float64) float64 {
	z, w := s.args(x)
	if z.Hi > 0.5 {
		return ibeta(s.n/2, s.m/2, w, z, true)
	}
	return ibeta(s.m/2, s.n/2, z, w, false)
}

func (s fStd) sf(x float64) float64 {
	z, w := s.args(x)
	if z.Hi > 0.5 {
		return ibeta(s.n/2, s.m/2, w, z, false)
	}
	return ibeta(s.m/2, s.n/2, z, w, true)
}

func (s fStd) logcdf(x float64) float64 { return logPQ(s.cdf(x), s.sf(x)) }
func (s fStd) logsf(x float64) float64  { return logPQ(s.sf(x), s.cdf(x)) }

// fromBeta returns x with z = v/(n+v) and w = 1-z: x = n·z/(m·w).
func (s fStd) fromBeta(z, w float64) float64 { return s.n * z / (s.m * w) }

func (s fStd) ppf(p float64) float64 {
	z := special.BetaIncInv(s.m/2, s.n/2, p)
	if z <= 0.5 {
		return s.fromBeta(z, 1-z)
	}
	w := special.BetaIncCInv(s.n/2, s.m/2, p)
	return s.fromBeta(1-w, w)
}

func (s fStd) isf(q float64) float64 {
	z := special.BetaIncCInv(s.m/2, s.n/2, q)
	if z <= 0.5 {
		return s.fromBeta(z, 1-z)
	}
	w := special.BetaIncInv(s.n/2, s.m/2, q)
	return s.fromBeta(1-w, w)
}

// mean, variance, skew and kurtosis are SciPy's formulas.
func (s fStd) mean() float64 {
	if s.n > 2 {
		return s.n / (s.n - 2)
	}
	return math.Inf(1)
}

func (s fStd) variance() float64 {
	v1, v2 := s.m, s.n
	if v2 > 4 {
		return 2 * v2 * v2 * (v1 + v2 - 2) / (v1 * (v2 - 2) * (v2 - 2) * (v2 - 4))
	}
	return math.Inf(1)
}

func (s fStd) skew() float64 {
	v1, v2 := s.m, s.n
	if v2 > 6 {
		return (2*v1 + v2 - 2) / (v2 - 6) * math.Sqrt((v2-4)/(v1*(v1+v2-2))) * math.Sqrt(8)
	}
	return math.NaN()
}

func (s fStd) kurtosis() float64 {
	v2 := s.n
	if v2 > 8 {
		g1 := s.skew()
		return (8 + g1*g1*(v2-6)) / (v2 - 8) * 1.5
	}
	return math.NaN()
}

func (s fStd) median() float64 { return s.ppf(0.5) }

func (s fStd) entropy() float64 {
	hn, hd, hs := 0.5*s.m, 0.5*s.n, 0.5*(s.m+s.n)
	return math.Log(s.n) - math.Log(s.m) + special.LogBeta(hn, hd) + (1-hn)*special.Digamma(hn) - (1+hd)*special.Digamma(hd) + hs*special.Digamma(hs)
}

// munp: E[X^k] = (n/m)^k Γ(m/2+k) Γ(n/2-k) / (Γ(m/2) Γ(n/2)) for 2k < n,
// +Inf from 2k >= n.
func (s fStd) munp(k int) float64 {
	fk := float64(k)
	if 2*fk >= s.n {
		return math.Inf(1)
	}
	r := math.Pow(s.n/s.m, fk) * poch(s.m/2, k)
	for j := 1; j <= k; j++ {
		r /= s.n/2 - float64(j)
	}
	return r
}

func (s fStd) rvs(g *random.Generator, size []int) (*ndarray.Array, error) {
	return g.F(s.m, s.n, size...)
}

// FitF returns the maximum-likelihood F distribution for x:
// scipy.stats.f.fit(x, fdfn=, fdfd=, floc=, fscale=), numerically from
// SciPy's starting point.
func FitF[D Data](x D, o *FitOptions) (F, error) {
	v := floats(x)
	fx, err := o.fixed("F", "Dfn", "Dfd", "Loc", "Scale")
	if err != nil {
		return F{}, err
	}
	if err := checkFitData(v); err != nil {
		return F{}, err
	}
	sh, loc, scale, err := fitContinuous(v, fx, []string{"Dfn", "Dfd"}, []bool{true, true}, []float64{1, 1},
		func(s []float64) contStd { return fStd{m: s[0], n: s[1]} })
	if err != nil {
		return F{}, err
	}
	d := F{Dfn: sh[0], Dfd: sh[1], Loc: loc, Scale: scale}
	return d, d.Validate()
}
