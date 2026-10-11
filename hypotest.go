package stats

import (
	"fmt"
	"math"

	"github.com/go-numerics/special"
)

// This file holds the helpers shared by the hypothesis tests.
//
// Conventions of the tests (ttest.go, chisquare.go, correlation.go, ks.go,
// ranktests.go, normality.go, anova.go):
//
//   - The options argument may be nil; the zero value of every options
//     struct selects SciPy's defaults.
//   - Where SciPy raises ValueError (mismatched lengths, an invalid option,
//     a negative count), the function returns an error wrapping ErrData or
//     ErrParam.
//   - Where SciPy returns NaN, with or without a warning (a NaN in the
//     input under SciPy's default nan_policy="propagate", a sample too
//     small for the test, constant data), the function returns a result
//     whose fields are NaN and a nil error, as SciPy does.

// TestResult is the outcome of a test that has a statistic and a p-value
// only (SciPy's two-field result tuples and SignificanceResult).
type TestResult struct {
	Statistic, PValue float64
}

// Method selects how a test computes its p-value: SciPy's method= argument
// of the rank, correlation and Kolmogorov-Smirnov tests. The zero value is
// MethodAuto, SciPy's default.
type Method int

const (
	// MethodAuto is method="auto": the exact distribution when it is cheap
	// and valid, the asymptotic one otherwise (each test documents its
	// rule).
	MethodAuto Method = iota
	// MethodExact is method="exact": the exact null distribution.
	MethodExact
	// MethodAsymptotic is method="asymptotic" (method="asymp" for the
	// Kolmogorov-Smirnov tests): the large-sample approximation.
	MethodAsymptotic
	// MethodApprox is the Kolmogorov-Smirnov one-sample test's
	// method="approx": twice the one-sided exact p-value.
	MethodApprox
)

// String returns SciPy's name for the method.
func (m Method) String() string {
	switch m {
	case MethodAuto:
		return "auto"
	case MethodExact:
		return "exact"
	case MethodAsymptotic:
		return "asymptotic"
	case MethodApprox:
		return "approx"
	}
	return "invalid"
}

var nan = math.NaN()

// checkAlternative returns an error wrapping ErrParam for an Alternative
// outside {TwoSided, Less, Greater}.
func checkAlternative(a Alternative) error {
	if a < TwoSided || a > Greater {
		return fmt.Errorf("%w: alternative %d is not TwoSided, Less or Greater", ErrParam, int(a))
	}
	return nil
}

// hasNaN reports whether any value is NaN.
func hasNaN(v ...[]float64) bool {
	for _, s := range v {
		for _, x := range s {
			if x != x {
				return true
			}
		}
	}
	return false
}

// pvalueSym is SciPy's _get_pvalue for a distribution symmetric about 0
// given by its survival function: less → sf(-t), greater → sf(t),
// two-sided → 2·sf(|t|).
func pvalueSym(t float64, sf func(float64) float64, alt Alternative) float64 {
	switch alt {
	case Less:
		return sf(-t)
	case Greater:
		return sf(t)
	}
	return 2 * sf(math.Abs(t))
}

// pvalueCDF is SciPy's _get_pvalue with explicit cdf and sf.
func pvalueCDF(t float64, cdf, sf func(float64) float64, alt Alternative) float64 {
	switch alt {
	case Less:
		return cdf(t)
	case Greater:
		return sf(t)
	}
	return 2 * math.Min(cdf(t), sf(t))
}

// normSF is special.ndtr(-z), SciPy's _SimpleNormal().sf.
func normSF(z float64) float64 { return special.Ndtr(-z) }

// studentSF is the survival function of Student's t with df degrees of
// freedom (SciPy's _SimpleStudentT(df).sf = stdtr(df, -t)).
func studentSF(df float64) func(float64) float64 {
	d := StudentT{Df: df, Scale: 1}
	return d.SF
}

// chi2SF is the chi-squared survival function (SciPy's chdtrc(df, x)).
func chi2SF(df, x float64) float64 {
	return ChiSquared{Df: df, Scale: 1}.SF(x)
}

// clip01 clips a probability to [0, 1], propagating NaN as np.clip does.
func clip01(p float64) float64 {
	switch {
	case p < 0:
		return 0
	case p > 1:
		return 1
	}
	return p
}

// clip clips x to [lo, hi], propagating NaN.
func clip(x, lo, hi float64) float64 {
	switch {
	case x < lo:
		return lo
	case x > hi:
		return hi
	}
	return x
}

// centralMoment is SciPy's _moment(a, order) about the given center:
// mean((a - center)**order), the powers formed by repeated products.
func centralMoment(a []float64, center float64, order int) float64 {
	d := make([]float64, len(a))
	for i, v := range a {
		e := v - center
		p := e
		for k := 1; k < order; k++ {
			p = float64(p * e)
		}
		d[i] = p
	}
	return mean(d)
}

// sampleSkew is scipy.stats.skew(a) (biased): m3/m2^1.5, NaN where the
// variance is zero to working precision (m2 <= (eps·mean)²).
func sampleSkew(a []float64) float64 {
	m := mean(a)
	m2 := centralMoment(a, m, 2)
	m3 := centralMoment(a, m, 3)
	if m2 <= (0x1p-52*m)*(0x1p-52*m) {
		return nan
	}
	return m3 / math.Pow(m2, 1.5)
}

// sampleKurtosis is scipy.stats.kurtosis(a, fisher=False) (biased,
// Pearson's definition): m4/m2², NaN where the variance is zero to working
// precision.
func sampleKurtosis(a []float64) float64 {
	m := mean(a)
	m2 := centralMoment(a, m, 2)
	m4 := centralMoment(a, m, 4)
	if m2 <= (0x1p-52*m)*(0x1p-52*m) {
		return nan
	}
	return m4 / (m2 * m2)
}

// varDdof is SciPy's _var(a, ddof): mean of the squared deviations, then
// scaled by n/(n-ddof).
func varDdof(a []float64, ddof float64) float64 {
	n := float64(len(a))
	v := centralMoment(a, mean(a), 2)
	return v * (n / (n - ddof))
}
