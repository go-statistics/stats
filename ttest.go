package stats

import (
	"fmt"
	"math"
	"slices"
)

// TTestResult is the outcome of a t-test (SciPy's TtestResult).
type TTestResult struct {
	// Statistic is the t statistic.
	Statistic float64
	// PValue is the p-value for the alternative hypothesis.
	PValue float64
	// DF is the number of degrees of freedom of the t distribution.
	DF float64

	alt      Alternative
	isNaN    bool    // the result is NaN (NaN input, too few observations)
	se       float64 // the denominator of the statistic
	estimate float64 // the point estimate (mean, or difference of means)
}

// ConfidenceInterval returns the confidence interval at the given level
// (in [0, 1]) for the population mean (one-sample test), the difference of
// the population means (two-sample tests) or the mean difference (paired
// test), on the side of the test's alternative (one-sided intervals have
// an infinite bound): TtestResult.confidence_interval(confidence_level).
func (r TTestResult) ConfidenceInterval(level float64) (ConfidenceInterval, error) {
	if !(level >= 0 && level <= 1) {
		return ConfidenceInterval{qnan, qnan}, fmt.Errorf("%w: confidence level %v outside [0, 1]", ErrParam, level)
	}
	if r.isNaN {
		return ConfidenceInterval{qnan, qnan}, nil
	}
	t := StudentT{Df: r.DF, Scale: 1}
	lo, hi := math.Inf(-1), math.Inf(1)
	switch r.alt {
	case Less:
		hi = t.PPF(level)
	case Greater:
		lo = t.PPF(1 - level)
	default:
		tail := (1 - level) / 2
		lo, hi = t.PPF(tail), t.PPF(1-tail)
	}
	return ConfidenceInterval{lo*r.se + r.estimate, hi*r.se + r.estimate}, nil
}

// TTestOptions are the options of TTest1Samp and TTestRel. The zero
// value (or nil) is SciPy's default: a two-sided test.
type TTestOptions struct {
	Alternative Alternative
}

// TTestIndOptions are the options of TTestInd. The zero value (or nil)
// is SciPy's default: Student's test with pooled variance, two-sided, no
// trimming.
type TTestIndOptions struct {
	// Welch selects Welch's test, which does not assume equal variances
	// (SciPy's equal_var=False).
	Welch bool
	// Trim is the proportion in [0, 0.5) cut from each end of each sample
	// for Yuen's trimmed t-test (SciPy's trim=).
	Trim        float64
	Alternative Alternative
}

func nanTTest() TTestResult {
	return TTestResult{Statistic: qnan, PValue: qnan, DF: qnan, isNaN: true, se: qnan, estimate: qnan}
}

// TTest1Samp is the one-sample t-test of the null hypothesis that the
// mean of the population x is drawn from is popmean:
// scipy.stats.ttest_1samp(x, popmean, alternative=...).
func TTest1Samp[D Data](x D, popmean float64, o *TTestOptions) (TTestResult, error) {
	if o == nil {
		o = &TTestOptions{}
	}
	if err := checkAlternative(o.Alternative); err != nil {
		return nanTTest(), err
	}
	return ttest1(floats(x), popmean, o.Alternative), nil
}

func ttest1(a []float64, popmean float64, alt Alternative) TTestResult {
	if len(a) == 0 || anyNaN(a) {
		return nanTTest()
	}
	n := float64(len(a))
	df := n - 1
	m := mean(a)
	d := m - popmean
	v := varDdof(a, 1)
	denom := math.Sqrt(v / n)
	t := d / denom
	return TTestResult{
		Statistic: t,
		PValue:    pvalueSym(t, studentSF(df), alt),
		DF:        df,
		alt:       alt,
		se:        denom,
		estimate:  m,
	}
}

// TTestRel is the paired t-test of the null hypothesis that the mean of
// the differences a - b is zero: scipy.stats.ttest_rel(a, b,
// alternative=...). a and b must have the same length.
func TTestRel[D Data](a, b D, o *TTestOptions) (TTestResult, error) {
	if o == nil {
		o = &TTestOptions{}
	}
	if err := checkAlternative(o.Alternative); err != nil {
		return nanTTest(), err
	}
	x, y := floats(a), floats(b)
	if len(x) != len(y) {
		return nanTTest(), fmt.Errorf("%w: ttest_rel needs samples of equal length, got %d and %d", ErrData, len(x), len(y))
	}
	d := make([]float64, len(x))
	for i := range x {
		d[i] = x[i] - y[i]
	}
	return ttest1(d, 0, o.Alternative), nil
}

// TTestInd is the two-sample t-test of the null hypothesis that two
// independent samples have identical means: scipy.stats.ttest_ind(a, b,
// equal_var=!Welch, trim=Trim, alternative=...). With Trim > 0 it is
// Yuen's test on the trimmed means and winsorized variances.
func TTestInd[D Data](a, b D, o *TTestIndOptions) (TTestResult, error) {
	if o == nil {
		o = &TTestIndOptions{}
	}
	if err := checkAlternative(o.Alternative); err != nil {
		return nanTTest(), err
	}
	if !(o.Trim >= 0 && o.Trim < 0.5) {
		return nanTTest(), fmt.Errorf("%w: trim %v outside [0, 0.5)", ErrParam, o.Trim)
	}
	x, y := floats(a), floats(b)
	if len(x) == 0 || len(y) == 0 || anyNaN(x, y) {
		return nanTTest(), nil
	}
	var v1, v2, m1, m2, n1, n2 float64
	if o.Trim == 0 {
		n1, n2 = float64(len(x)), float64(len(y))
		v1, v2 = varDdof(x, 1), varDdof(y, 1)
		m1, m2 = mean(x), mean(y)
	} else {
		v1, m1, n1 = trimmedStats(x, o.Trim)
		v2, m2, n2 = trimmedStats(y, o.Trim)
	}
	var df, denom float64
	if o.Welch {
		vn1, vn2 := v1/n1, v2/n2
		df = (vn1 + vn2) * (vn1 + vn2) / (vn1*vn1/(n1-1) + vn2*vn2/(n2-1))
		if df != df {
			df = 1
		}
		denom = math.Sqrt(vn1 + vn2)
	} else {
		if n1 == 1 {
			v1 = 0
		}
		if n2 == 1 {
			v2 = 0
		}
		df = n1 + n2 - 2
		svar := ((n1-1)*v1 + (n2-1)*v2) / df
		denom = math.Sqrt(svar * (1/n1 + 1/n2))
	}
	t := (m1 - m2) / denom
	return TTestResult{
		Statistic: t,
		PValue:    pvalueSym(t, studentSF(df), o.Alternative),
		DF:        df,
		alt:       o.Alternative,
		se:        denom,
		estimate:  m1 - m2,
	}, nil
}

// trimmedStats is SciPy's _ttest_trim_var_mean_len: the g-times
// winsorized variance (ddof 2g+1), the trimmed mean and the trimmed
// length, g = floor(n·trim).
func trimmedStats(a []float64, trim float64) (v, m, n float64) {
	s := slices.Clone(a)
	sortFloats(s)
	g := int(float64(len(s)) * trim)
	if g == 0 {
		v = varDdof(s, 1)
	} else {
		w := slices.Clone(s)
		for i := 0; i < g; i++ {
			w[i] = s[g]
			w[len(w)-1-i] = s[len(s)-1-g]
		}
		v = varDdof(w, float64(2*g+1))
	}
	// trim_mean: g values cut from each end, the mean of the middle
	// (SciPy sums the middle in np.partition's order, which differs from
	// the sorted one only in rounding).
	m = mean(s[g : len(s)-g])
	return v, m, float64(len(s) - 2*g)
}
