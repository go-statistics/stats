package stats

import (
	"fmt"
	"math"

	"github.com/go-ndarray/ndarray"
	"github.com/go-ndarray/ndarray/random"
	"github.com/go-numerics/optimize"
	"github.com/go-numerics/special"
	"github.com/go-statistics/stats/internal/dd"
)

//stats:continuous Beta beta a=A b=B

// Beta is the beta distribution on [0, 1] with shapes A and B: density
// x^(A-1) (1-x)^(B-1) / B(A, B) in standard form: scipy.stats.beta(a, b,
// loc, scale).
//
// CDF and SF are special.BetaInc and special.BetaIncC, PPF and ISF their
// inverses; the density is betaDensity (double-double exponent, Stirling
// form for large shapes), 1-x being exact on [1/2, 1].
type Beta struct {
	A, B, Loc, Scale float64
}

func (d Beta) std() betaStd { return betaStd{a: d.A, b: d.B} }

type betaStd struct{ a, b float64 }

func (s betaStd) valid() bool {
	return s.a > 0 && s.b > 0 && !math.IsInf(s.a, 1) && !math.IsInf(s.b, 1)
}
func (s betaStd) support() (lo, hi float64) { return 0, 1 }

// edge returns the density at x = 0 (or, with the shapes swapped, at 1):
// +Inf for a < 1, 0 for a > 1, and b at a = 1.
func betaEdge(a, b float64) float64 {
	switch {
	case a < 1:
		return math.Inf(1)
	case a > 1:
		return 0
	}
	return b
}

func (s betaStd) pdf(x float64) float64 {
	switch x {
	case 0:
		return betaEdge(s.a, s.b)
	case 1:
		return betaEdge(s.b, s.a)
	}
	return betaDensity(s.a, s.b, dd.F(x), oneMinus(x), dd.F(1))
}

func (s betaStd) logpdf(x float64) float64 {
	switch x {
	case 0:
		return math.Log(betaEdge(s.a, s.b))
	case 1:
		return math.Log(betaEdge(s.b, s.a))
	}
	return betaLogDensity(s.a, s.b, dd.F(x), oneMinus(x)).Float()
}

func (s betaStd) cdf(x float64) float64 { return special.BetaInc(s.a, s.b, x) }
func (s betaStd) sf(x float64) float64  { return special.BetaIncC(s.a, s.b, x) }
func (s betaStd) logcdf(x float64) float64 {
	return logPQ(special.BetaInc(s.a, s.b, x), special.BetaIncC(s.a, s.b, x))
}
func (s betaStd) logsf(x float64) float64 {
	return logPQ(special.BetaIncC(s.a, s.b, x), special.BetaInc(s.a, s.b, x))
}
func (s betaStd) ppf(p float64) float64 { return special.BetaIncInv(s.a, s.b, p) }
func (s betaStd) isf(q float64) float64 { return special.BetaIncCInv(s.a, s.b, q) }

func (s betaStd) mean() float64 { return s.a / (s.a + s.b) }
func (s betaStd) variance() float64 {
	ab := s.a + s.b
	return s.a * s.b / (ab * ab * (ab + 1))
}
func (s betaStd) skew() float64 {
	a, b := s.a, s.b
	return 2 * (b - a) * math.Sqrt(a+b+1) / ((a + b + 2) * math.Sqrt(a*b))
}
func (s betaStd) kurtosis() float64 {
	a, b := s.a, s.b
	ab := a + b
	n := 6 * ((a-b)*(a-b)*(ab+1) - a*b*(ab+2))
	return n / (a * b * (ab + 2) * (ab + 3))
}
func (s betaStd) median() float64 { return special.BetaIncInv(s.a, s.b, 0.5) }

// entropy is betaEntropy (the large terms cancelled analytically).
func (s betaStd) entropy() float64 { return betaEntropy(s.a, s.b) }

// munp is Π_{k<n} (a+k)/(a+b+k).
func (s betaStd) munp(n int) float64 {
	r := 1.0
	for k := 0; k < n; k++ {
		r *= (s.a + float64(k)) / (s.a + s.b + float64(k))
	}
	return r
}

func (s betaStd) rvs(g *random.Generator, size []int) (*ndarray.Array, error) {
	return g.Beta(s.a, s.b, size...)
}

// FitBeta returns the maximum-likelihood beta distribution for x:
// scipy.stats.beta.fit(x, fa=, fb=, floc=, fscale=). With Loc and Scale
// both fixed it solves the likelihood equations
// ψ(a) - ψ(a+b) = mean log y, ψ(b) - ψ(a+b) = mean log(1-y) as SciPy does
// (to full precision); otherwise it maximises the likelihood numerically
// from SciPy's starting point.
func FitBeta[D Data](x D, o *FitOptions) (Beta, error) {
	v := floats(x)
	fx, err := o.fixed("Beta", "A", "B", "Loc", "Scale")
	if err != nil {
		return Beta{}, err
	}
	if err := checkFitData(v); err != nil {
		return Beta{}, err
	}
	loc, haveLoc := fx["Loc"]
	scale, haveScale := fx["Scale"]
	if haveLoc && haveScale {
		return fitBetaShapes(v, loc, scale, fx)
	}
	a0, b0 := betaMomentStart(v)
	sh, loc, scale, err := fitContinuous(v, fx, []string{"A", "B"}, []bool{true, true}, []float64{a0, b0},
		func(s []float64) contStd { return betaStd{a: s[0], b: s[1]} })
	if err != nil {
		return Beta{}, err
	}
	d := Beta{A: sh[0], B: sh[1], Loc: loc, Scale: scale}
	return d, d.Validate()
}

// betaMomentStart is SciPy's beta._fitstart: the shapes whose skewness and
// kurtosis match the sample's (solved here by Newton's method from (1, 1)
// on the same equations), falling back to (1, 1).
func betaMomentStart(v []float64) (a, b float64) {
	g1 := skewOf(v)
	g2 := kurtosisOf(v)
	f := func(x, out []float64) {
		a, b := x[0], x[1]
		sk := 2 * (b - a) * math.Sqrt(a+b+1) / (a + b + 2) / math.Sqrt(a*b)
		ku := a*a*a - a*a*(2*b-1) + b*b*(b+1) - 2*a*b*(b+2)
		ku /= a * b * (a + b + 2) * (a + b + 3)
		ku *= 6
		out[0], out[1] = sk-g1, ku-g2
	}
	r, err := optimize.Fsolve(optimize.SystemProblem{Func: f}, []float64{1, 1}, nil)
	if err != nil || !(r.X[0] > 0 && r.X[1] > 0) {
		return 1, 1
	}
	return r.X[0], r.X[1]
}

// kurtosisOf is the biased excess kurtosis (scipy.stats.kurtosis(x)).
func kurtosisOf(v []float64) float64 {
	m := mean(v)
	d2 := make([]float64, len(v))
	d4 := make([]float64, len(v))
	for i, t := range v {
		e := t - m
		d2[i] = float64(e * e)
		d4[i] = float64(d2[i] * d2[i])
	}
	m2, m4 := mean(d2), mean(d4)
	return m4/(m2*m2) - 3
}

func fitBetaShapes(v []float64, loc, scale float64, fx map[string]float64) (Beta, error) {
	n := float64(len(v))
	ly := make([]float64, len(v))
	l1y := make([]float64, len(v))
	y := make([]float64, len(v))
	for i, t := range v {
		u := (t - loc) / scale
		if !(u > 0 && u < 1) {
			return Beta{}, fmt.Errorf("%w: beta fit with Loc=%v, Scale=%v needs every value inside (Loc, Loc+Scale)", ErrData, loc, scale)
		}
		y[i] = u
		ly[i] = math.Log(u)
		l1y[i] = math.Log1p(-u)
	}
	s1, s2 := sum(ly)/n, sum(l1y)/n
	a, haveA := fx["A"]
	b, haveB := fx["B"]
	switch {
	case haveA:
		// ψ(b) - ψ(a+b) = s2
		b = solveDigammaDiff(a, s2, b0FromMean(1-mean(y), a))
	case haveB:
		a = solveDigammaDiff(b, s1, b0FromMean(mean(y), b))
	default:
		xbar := mean(y)
		fac := xbar*(1-xbar)/variance(y, 0) - 1
		a, b = xbar*fac, (1-xbar)*fac
		if !(a > 0 && b > 0) {
			a, b = 1, 1
		}
		// Newton on (log a, log b) for ψ(a)-ψ(a+b) = s1, ψ(b)-ψ(a+b) = s2.
		for i := 0; i < 200; i++ {
			pab := special.Digamma(a + b)
			f1 := special.Digamma(a) - pab - s1
			f2 := special.Digamma(b) - pab - s2
			t1, t2, tab := special.Polygamma(1, a), special.Polygamma(1, b), special.Polygamma(1, a+b)
			// Jacobian in (a, b)
			j11, j12, j21, j22 := t1-tab, -tab, -tab, t2-tab
			det := j11*j22 - j12*j21
			da := (f1*j22 - f2*j12) / det
			db := (j11*f2 - j21*f1) / det
			na, nb := a-da, b-db
			if !(na > 0) {
				na = a / 2
			}
			if !(nb > 0) {
				nb = b / 2
			}
			done := math.Abs(na-a) <= 4e-16*a && math.Abs(nb-b) <= 4e-16*b
			a, b = na, nb
			if done {
				break
			}
		}
	}
	d := Beta{A: a, B: b, Loc: loc, Scale: scale}
	return d, d.Validate()
}

// b0FromMean is the start for the free shape given the fixed one: the
// shape whose beta mean a/(a+b) matches m.
func b0FromMean(m, fixed float64) float64 { return fixed * m / (1 - m) }

// solveDigammaDiff solves ψ(u) - ψ(u + c) = s for u > 0 by Newton's method.
func solveDigammaDiff(c, s, u float64) float64 {
	if !(u > 0) || math.IsInf(u, 0) {
		u = 1
	}
	for i := 0; i < 200; i++ {
		f := special.Digamma(u) - special.Digamma(u+c) - s
		fp := special.Polygamma(1, u) - special.Polygamma(1, u+c)
		nu := u - f/fp
		if !(nu > 0) {
			nu = u / 2
		}
		if math.Abs(nu-u) <= 4e-16*u {
			return nu
		}
		u = nu
	}
	return u
}
