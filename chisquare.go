package stats

import (
	"fmt"
	"math"

	"github.com/go-ndarray/ndarray"
	"github.com/go-numerics/special"
	"github.com/go-statistics/stats/internal/dd"
)

// The Cressie-Read power λ of PowerDivergence: SciPy's named lambda_
// values.
const (
	LambdaPearson          = 1.0       // "pearson": Pearson's chi-squared
	LambdaLogLikelihood    = 0.0       // "log-likelihood": the G-test
	LambdaFreemanTukey     = -0.5      // "freeman-tukey"
	LambdaModLogLikelihood = -1.0      // "mod-log-likelihood"
	LambdaNeyman           = -2.0      // "neyman"
	LambdaCressieRead      = 2.0 / 3.0 // "cressie-read"
)

// ChiSquareOptions are the options of ChiSquare and PowerDivergence. The
// zero value (or nil) is SciPy's default: uniform expected frequencies
// and no extra degrees of freedom removed.
type ChiSquareOptions struct {
	// Expected are the expected frequencies (SciPy's f_exp), one per
	// observed frequency; nil means every category equally likely. Their
	// sum must agree with that of the observed frequencies to a relative
	// tolerance of sqrt(eps), as SciPy requires.
	Expected []float64
	// DDOF is the adjustment to the degrees of freedom: the p-value uses
	// k - 1 - DDOF (SciPy's ddof).
	DDOF int
}

// ChiSquare is Pearson's chi-squared goodness-of-fit test of the observed
// frequencies against the expected ones: scipy.stats.chisquare(f_obs,
// f_exp, ddof).
func ChiSquare[D Data](observed D, o *ChiSquareOptions) (TestResult, error) {
	return PowerDivergence(observed, LambdaPearson, o)
}

// PowerDivergence is the Cressie-Read power divergence goodness-of-fit
// test with power lambda (one of the Lambda constants, or any real):
// scipy.stats.power_divergence(f_obs, f_exp, ddof, lambda_=lambda).
func PowerDivergence[D Data](observed D, lambda float64, o *ChiSquareOptions) (TestResult, error) {
	if o == nil {
		o = &ChiSquareOptions{}
	}
	obs := floats(observed)
	exp := o.Expected
	if exp != nil {
		if len(exp) != len(obs) {
			return TestResult{qnan, qnan}, fmt.Errorf("%w: %d expected frequencies for %d observed", ErrData, len(exp), len(obs))
		}
		if anyNaN(obs, exp) {
			return TestResult{qnan, qnan}, nil
		}
		so, se := sum(obs), sum(exp)
		if rel := math.Abs(so-se) / math.Min(so, se); rel > 1.4901161193847656e-08 {
			return TestResult{qnan, qnan}, fmt.Errorf("%w: the observed frequencies sum to %v and the expected ones to %v (relative difference %v > sqrt(eps))", ErrData, so, se, rel)
		}
	} else if anyNaN(obs) {
		return TestResult{qnan, qnan}, nil
	}
	if math.IsNaN(lambda) {
		return TestResult{qnan, qnan}, fmt.Errorf("%w: lambda is NaN", ErrParam)
	}
	return powerDivergence(obs, exp, float64(o.DDOF), lambda), nil
}

// powerDivergence is SciPy's _power_divergence on validated data; exp nil
// means the mean of obs in every cell.
func powerDivergence(obs, exp []float64, ddof, lambda float64) TestResult {
	terms := make([]float64, len(obs))
	e := 0.0
	if exp == nil {
		e = mean(obs)
	}
	for i, f := range obs {
		if exp != nil {
			e = exp[i]
		}
		switch lambda {
		case 1:
			d := f - e
			terms[i] = d * d / e
		case 0:
			terms[i] = 2 * special.Xlogy(f, f/e)
		case -1:
			terms[i] = 2 * special.Xlogy(e, e/f)
		default:
			terms[i] = f * (crPow(f/e, lambda) - 1) / (0.5 * lambda * (lambda + 1))
		}
	}
	stat := sumDD(terms)
	df := float64(len(obs)) - 1 - ddof
	return TestResult{stat, chi2SF(df, stat)}
}

// sumDD is Σ a accumulated in double-double and rounded once: the
// p-value of a large statistic is sensitive to its last bits (the
// relative error of the chi-squared tail grows like x/2 times that of x).
func sumDD(a []float64) float64 {
	if y, ok := sumFast(a); ok {
		return y
	}
	return sumDDExact(a)
}

// sumDDExact is sumDD's double-double sum (Σ through sum where a holds an
// infinity or a NaN).
func sumDDExact(a []float64) float64 {
	var s dd.DD
	for _, v := range a {
		if math.IsInf(v, 0) || v != v {
			return sum(a)
		}
		s = dd.AddF(s, v)
	}
	return s.Float()
}

// Chi2ContingencyOptions are the options of Chi2Contingency. The zero
// value (or nil) is SciPy's default: Yates' correction for 1 degree of
// freedom and Pearson's statistic.
type Chi2ContingencyOptions struct {
	// NoCorrection disables Yates' continuity correction (SciPy's
	// correction=False).
	NoCorrection bool
	// Lambda, when not nil, selects the Cressie-Read power divergence
	// statistic instead of Pearson's (SciPy's lambda_).
	Lambda *float64
}

// Chi2ContingencyResult is the outcome of Chi2Contingency.
type Chi2ContingencyResult struct {
	Statistic, PValue float64
	// DOF is the number of degrees of freedom.
	DOF int
	// Expected are the expected frequencies under independence, an array
	// of the table's shape.
	Expected *ndarray.Array
}

// Chi2Contingency is the chi-squared test of independence of the
// variables of an N-dimensional contingency table of observed counts:
// scipy.stats.chi2_contingency(observed, correction, lambda_).
func Chi2Contingency(observed *ndarray.Array, o *Chi2ContingencyOptions) (Chi2ContingencyResult, error) {
	if o == nil {
		o = &Chi2ContingencyOptions{}
	}
	bad := Chi2ContingencyResult{Statistic: qnan, PValue: qnan}
	if observed == nil || observed.Size() == 0 {
		return bad, fmt.Errorf("%w: chi2_contingency needs a non-empty table", ErrData)
	}
	shape := observed.Shape()
	obs := append([]float64(nil), arrayFloats(observed)...)
	for _, v := range obs {
		if v < 0 {
			return bad, fmt.Errorf("%w: negative count %v in the table", ErrData, v)
		}
	}
	exp := expectedFreq(obs, shape)
	for i, v := range exp {
		if v == 0 {
			return bad, fmt.Errorf("%w: the expected frequency at flat index %d is zero", ErrData, i)
		}
	}
	expected, _ := ndarray.FromData(exp, shape...)
	dof := len(exp) + len(shape) - 1
	for _, s := range shape {
		dof -= s
	}
	res := Chi2ContingencyResult{DOF: dof, Expected: expected}
	if dof == 0 {
		res.Statistic, res.PValue = 0, 1
		return res, nil
	}
	if dof == 1 && !o.NoCorrection {
		for i := range obs {
			diff := exp[i] - obs[i]
			mag := math.Min(0.5, math.Abs(diff))
			switch {
			case diff > 0:
				obs[i] += mag
			case diff < 0:
				obs[i] -= mag
			}
		}
	}
	lambda := LambdaPearson
	if o.Lambda != nil {
		lambda = *o.Lambda
	}
	if anyNaN(obs) {
		res.Statistic, res.PValue = qnan, qnan
		return res, nil
	}
	r := balancedDivergence(obs, exp, float64(len(obs)-1-dof), lambda)
	res.Statistic, res.PValue = r.Statistic, r.PValue
	return res, nil
}

// balancedDivergence is the power divergence statistic for observed and
// expected frequencies with equal totals (as in a contingency table, where
// the expected frequencies come from the margins). Each term is written
// with its part linear in f - e removed (Σ (f - e) = 0), which leaves the
// sum unchanged but makes every term non-negative: SciPy sums terms of
// both signs that cancel, and loses all accuracy when the table is close
// to independence (a G statistic of 2.5e-7 off by 9e-8 relative).
func balancedDivergence(obs, exp []float64, ddof, lambda float64) TestResult {
	if lambda == 1 {
		return powerDivergence(obs, exp, ddof, lambda)
	}
	terms := make([]float64, len(obs))
	for i, f := range obs {
		e := exp[i]
		t := (f - e) / e
		switch lambda {
		case 0:
			// 2e[(1+t)·log(1+t) - t]
			if f == 0 {
				terms[i] = 2 * e
			} else {
				terms[i] = 2 * e * xlog1pxMinusX(t)
			}
		case -1:
			// 2e[t - log(1+t)]
			terms[i] = 2 * e * xMinusLog1p(t)
		default:
			// e[(1+t)^(λ+1) - 1 - (λ+1)t] / (λ(λ+1)/2); for f = 0 and
			// λ < 0 SciPy's term is 0·∞ = NaN, kept for parity.
			if f == 0 && lambda < 0 {
				terms[i] = qnan
				continue
			}
			terms[i] = e * powExcess(t, lambda+1) / (0.5 * lambda * (lambda + 1))
		}
	}
	stat := sumDD(terms)
	df := float64(len(obs)) - 1 - ddof
	return TestResult{stat, chi2SF(df, stat)}
}

// xlog1pxMinusX is (1+t)·log(1+t) - t, by its series Σ_{k>=2}
// (-1)^k t^k / (k(k-1)) for small |t|.
func xlog1pxMinusX(t float64) float64 {
	if math.Abs(t) < 0.1 {
		s, p := 0.0, t
		for k := 2; k < 60; k++ {
			p *= -t
			s1 := s - p/float64(k*(k-1))
			if s1 == s {
				break
			}
			s = s1
		}
		return s
	}
	return (1+t)*math.Log1p(t) - t
}

// xMinusLog1p is t - log(1+t), by its series Σ_{k>=2} (-1)^k t^k / k for
// small |t|.
func xMinusLog1p(t float64) float64 {
	if math.Abs(t) < 0.1 {
		s, p := 0.0, t
		for k := 2; k < 60; k++ {
			p *= -t
			s1 := s - p/float64(k)
			if s1 == s {
				break
			}
			s = s1
		}
		return s
	}
	return t - math.Log1p(t)
}

// powExcess is (1+t)^a - 1 - a·t, by its binomial series Σ_{k>=2}
// C(a,k) t^k for small |t|.
func powExcess(t, a float64) float64 {
	if math.Abs(t) < 0.1 {
		s, p := 0.0, a*t
		for k := 2; k < 60; k++ {
			p *= (a - float64(k-1)) / float64(k) * t
			s1 := s + p
			if s1 == s {
				break
			}
			s = s1
		}
		return s
	}
	return math.Expm1(a*math.Log1p(t)) - a*t
}

// expectedFreq is scipy.stats.contingency.expected_freq: the product of
// the marginal sums divided by total^(ndim-1), row-major.
func expectedFreq(obs []float64, shape []int) []float64 {
	d := len(shape)
	strides := make([]int, d)
	st := 1
	for k := d - 1; k >= 0; k-- {
		strides[k] = st
		st *= shape[k]
	}
	margins := make([][]float64, d)
	for k := range margins {
		margins[k] = make([]float64, shape[k])
	}
	for i, v := range obs {
		for k := 0; k < d; k++ {
			margins[k][(i/strides[k])%shape[k]] += v
		}
	}
	total := sum(obs)
	den := math.Pow(total, float64(d-1))
	exp := make([]float64, len(obs))
	for i := range exp {
		p := margins[0][(i/strides[0])%shape[0]]
		for k := 1; k < d; k++ {
			p *= margins[k][(i/strides[k])%shape[k]]
		}
		exp[i] = p / den
	}
	return exp
}

// FisherExactOptions are the options of FisherExact. The zero value (or
// nil) is a two-sided test.
type FisherExactOptions struct {
	Alternative Alternative
}

// FisherExact is Fisher's exact test of independence for a 2×2 table of
// counts: scipy.stats.fisher_exact(table, alternative). The statistic is
// the sample odds ratio table[0][0]·table[1][1] / (table[0][1]·table[1][0])
// (+Inf when the denominator is zero, NaN with a p-value of 1 when a row
// or a column sums to zero); the p-value is computed from the
// hypergeometric distribution of table[0][0] given the margins, the
// two-sided one summing the probabilities of the tables no more likely
// than the observed one (relative tolerance 1e-14), as SciPy does.
func FisherExact(table [2][2]int64, o *FisherExactOptions) (TestResult, error) {
	if o == nil {
		o = &FisherExactOptions{}
	}
	if err := checkAlternative(o.Alternative); err != nil {
		return TestResult{qnan, qnan}, err
	}
	c00, c01, c10, c11 := table[0][0], table[0][1], table[1][0], table[1][1]
	if c00 < 0 || c01 < 0 || c10 < 0 || c11 < 0 {
		return TestResult{qnan, qnan}, fmt.Errorf("%w: negative count in the table", ErrData)
	}
	if c00+c10 == 0 || c01+c11 == 0 || c00+c01 == 0 || c10+c11 == 0 {
		return TestResult{qnan, 1}, nil
	}
	odds := math.Inf(1)
	if c10 > 0 && c01 > 0 {
		odds = float64(c00*c11) / float64(c10*c01)
	}
	n1 := float64(c00 + c01)
	n2 := float64(c10 + c11)
	n := float64(c00 + c10)
	x := float64(c00)
	h := hypergeom{total: n1 + n2, marked: n1, nDraw: n}
	var p float64
	switch o.Alternative {
	case Less:
		p = h.cdf(x)
	case Greater:
		p = hypergeom{total: n1 + n2, marked: n1, nDraw: float64(c01 + c11)}.cdf(float64(c01))
	default:
		p = fisherTwoSided(h, x, n)
	}
	return TestResult{odds, math.Min(p, 1)}, nil
}

// fisherTwoSided is the two-sided p-value of fisher_exact: the tail of
// the observed count plus the far tail of the counts whose probability
// does not exceed the observed one by more than a relative 1e-14.
func fisherTwoSided(h hypergeom, x, n float64) float64 {
	mode := math.Floor((n + 1) * (h.marked + 1) / (h.total + 2))
	pexact := h.pmf(x)
	pmode := h.pmf(mode)
	const eps = 1e-14
	gamma := 1 + eps
	if math.Abs(pexact-pmode)/math.Max(pexact, pmode) <= eps {
		return 1
	}
	if x < mode {
		plower := h.cdf(x)
		if h.pmf(n) > pexact*gamma {
			return plower
		}
		guess := binarySearch(func(k float64) float64 { return -h.pmf(k) }, -pexact*gamma, mode, n)
		return plower + h.sf(guess)
	}
	pupper := h.sf(x - 1)
	if h.pmf(0) > pexact*gamma {
		return pupper
	}
	guess := binarySearch(h.pmf, pexact*gamma, 0, mode)
	return pupper + h.cdf(guess)
}

// binarySearch is SciPy's _binary_search_for_binom_tst: for a function
// ascending on [lo, hi], the integer i with a(i) <= d < a(i+1).
func binarySearch(a func(float64) float64, d, lo, hi float64) float64 {
	for lo < hi {
		mid := lo + math.Floor((hi-lo)/2)
		v := a(mid)
		switch {
		case v < d:
			lo = mid + 1
		case v > d:
			hi = mid - 1
		default:
			lo, hi = mid, mid
		}
	}
	if a(lo) <= d {
		return lo
	}
	return lo - 1
}
