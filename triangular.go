package stats

import (
	"math"

	"github.com/go-ndarray/ndarray"
	"github.com/go-ndarray/ndarray/random"
	"github.com/go-statistics/stats/internal/dd"
)

//stats:continuous Triangular triang c=C

// Triangular is the triangular distribution on [0, 1] with mode C,
// 0 <= C <= 1, in standard form: scipy.stats.triang(c, loc, scale), the
// distribution on [Loc, Loc+Scale] with mode Loc + C·Scale.
//
// The two branches of every function are formed in double-double from
// the exact differences 1 - x and 1 - C: CDF x²/C below the mode and
// 1 - (1-x)²/(1-C) above, SF the complements, so that each tail keeps full
// relative accuracy (SciPy's (x² - 2x + c)/(c - 1) cancels near x = 1);
// PPF √(C p) and 1 - √((1-C)(1-p)).
type Triangular struct {
	C, Loc, Scale float64
}

func (d Triangular) std() triangStd { return triangStd{c: d.C} }

type triangStd struct{ c float64 }

func (s triangStd) valid() bool               { return s.c >= 0 && s.c <= 1 }
func (s triangStd) support() (lo, hi float64) { return 0, 1 }

func (s triangStd) pdfD(x float64) dd.DD {
	switch {
	case x < s.c:
		return dd.DivF(dd.F(2*x), s.c)
	case s.c == 1:
		return dd.F(2 * x)
	}
	return dd.MulF(dd.Div(oneMinus(x), oneMinus(s.c)), 2)
}

func (s triangStd) pdf(x float64) float64    { return s.pdfD(x).Float() }
func (s triangStd) logpdf(x float64) float64 { return logD(s.pdfD(x)).Float() }

// lower returns x²/c and upper (1-x)²/(1-c), the two tail areas.
func (s triangStd) lower(x float64) dd.DD { return dd.MulF(dd.DivF(dd.F(x), s.c), x) }
func (s triangStd) upper(x float64) dd.DD {
	u := oneMinus(x)
	return dd.Div(dd.Mul(u, u), oneMinus(s.c))
}

// diffPlus returns (a - b) + x(1 - x) for a >= b, a sum of two
// non-negative terms: 1 - x²/c = diffPlus(c, x)/c and
// 1 - (1-x)²/(1-c) = diffPlus(x, c)/(1-c), without the cancellation of
// 1 minus a tail area near 1.
func diffPlus(a, b, x float64) dd.DD {
	h, l := dd.TwoSum(a, -b)
	return dd.Add(dd.DD{Hi: h, Lo: l}, dd.MulF(oneMinus(x), x))
}

func (s triangStd) cdfD(x float64) dd.DD {
	if x < s.c {
		return s.lower(x)
	}
	return dd.Div(diffPlus(x, s.c, x), oneMinus(s.c))
}

func (s triangStd) sfD(x float64) dd.DD {
	if x < s.c {
		return dd.DivF(diffPlus(s.c, x, x), s.c)
	}
	return s.upper(x)
}

func (s triangStd) cdf(x float64) float64 { return s.cdfD(x).Float() }
func (s triangStd) sf(x float64) float64  { return s.sfD(x).Float() }
func (s triangStd) logcdf(x float64) float64 {
	if x < s.c && x < 0.5 {
		// 2 log x - log c: x²/c may underflow where its logarithm does not
		return dd.Sub(dd.MulF(dd.Log(x), 2), dd.Log(s.c)).Float()
	}
	return logPair(s.cdfD(x), s.sfD(x))
}
func (s triangStd) logsf(x float64) float64 { return logPair(s.sfD(x), s.cdfD(x)) }

// quantile returns the point with lower tail area p and upper tail area
// q = 1 - p (both given, one of them exactly as the caller holds it):
// √(c p) below the mode, and above it
// 1 - √((1-c) q) = (1 - (1-c) q)/(1 + √((1-c) q)) with
// 1 - (1-c) q = p + c q, a sum without cancellation.
func (s triangStd) quantile(p, q dd.DD) float64 {
	if p.Hi < s.c {
		return sqrtD(dd.MulF(p, s.c)).Float()
	}
	r := sqrtD(dd.Mul(q, oneMinus(s.c)))
	return dd.Div(dd.Add(p, dd.MulF(q, s.c)), dd.AddF(r, 1)).Float()
}

func (s triangStd) ppf(p float64) float64 { return s.quantile(dd.F(p), oneMinus(p)) }
func (s triangStd) isf(q float64) float64 { return s.quantile(oneMinus(q), dd.F(q)) }

func (s triangStd) mean() float64 { return (s.c + 1) / 3 }
func (s triangStd) variance() float64 {
	c := s.c
	return (1 - c + c*c) / 18
}
func (s triangStd) skew() float64 {
	c := s.c
	return math.Sqrt2 * (2*c - 1) * (c + 1) * (c - 2) / (5 * math.Pow(1-c+c*c, 1.5))
}
func (s triangStd) kurtosis() float64 { return -0.6 }
func (s triangStd) median() float64   { return s.ppf(0.5) }

// entropy is 1/2 - log 2.
func (s triangStd) entropy() float64 { return -0.1931471805599453 }

// munp is 2 Σ_{k=0}^{n} c^k / ((n+1)(n+2)).
func (s triangStd) munp(n int) float64 {
	r, ck := 0.0, 1.0
	for k := 0; k <= n; k++ {
		r += ck
		ck *= s.c
	}
	return 2 * r / float64((n+1)*(n+2))
}

func (s triangStd) rvs(g *random.Generator, size []int) (*ndarray.Array, error) {
	return g.Triangular(0, s.c, 1, size...)
}

// FitTriangular returns the maximum-likelihood triangular distribution for
// x: scipy.stats.triang.fit(x, fc=, floc=, fscale=), maximised
// numerically from SciPy's starting point (C = 1, Loc and Scale by the
// method of moments moved to cover the data), to a tighter optimum than
// SciPy's.
func FitTriangular[D Data](x D, o *FitOptions) (Triangular, error) {
	v := floats(x)
	fx, err := o.fixed("Triangular", "C", "Loc", "Scale")
	if err != nil {
		return Triangular{}, err
	}
	if err := checkFitData(v); err != nil {
		return Triangular{}, err
	}
	// SciPy's start covers the data with a margin: it has a likelihood
	sh, loc, scale, _ := fitContinuous(v, fx, []string{"C"}, []bool{false}, []float64{1},
		func(s []float64) contStd { return triangStd{c: s[0]} })
	d := Triangular{C: sh[0], Loc: loc, Scale: scale}
	return d, d.Validate()
}
