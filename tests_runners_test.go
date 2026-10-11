package stats

import (
	"fmt"
	"math"

	"github.com/go-ndarray/ndarray"
)

// The Go counterparts of the fixture functions of scipy_tests.py.

func ttOut(r TTestResult) (outputs, error) {
	o := outputs{"statistic": r.Statistic, "pvalue": r.PValue, "df": r.DF}
	for _, lv := range []float64{0.95, 0.9, 0.5, 0.999} {
		ci, err := r.ConfidenceInterval(lv)
		if err != nil {
			return nil, err
		}
		o[fmt.Sprintf("ci_lo_%v", lv)] = ci.Low
		o[fmt.Sprintf("ci_hi_%v", lv)] = ci.High
	}
	return o, nil
}

func tr(r TestResult, err error) (outputs, error) {
	return outputs{"statistic": r.Statistic, "pvalue": r.PValue}, err
}

func init() {
	registerTestRunner("ttest_1samp", func(c testCase) (outputs, error) {
		r, err := TTest1Samp(c.data[0], c.num("popmean"), &TTestOptions{Alternative: c.alt()})
		if err != nil {
			return nil, err
		}
		return ttOut(r)
	})
	registerTestRunner("ttest_ind", func(c testCase) (outputs, error) {
		o := &TTestIndOptions{Welch: !c.flag("equal_var"), Trim: c.num("trim"), Alternative: c.alt()}
		var r TTestResult
		var err error
		if len(c.data[0]) > 0 {
			r, err = TTestInd(arr(c.data[0]), arr(c.data[1]), o)
		} else {
			r, err = TTestInd(c.data[0], c.data[1], o)
		}
		if err != nil {
			return nil, err
		}
		return ttOut(r)
	})
	registerTestRunner("ttest_rel", func(c testCase) (outputs, error) {
		r, err := TTestRel(c.data[0], c.data[1], &TTestOptions{Alternative: c.alt()})
		if err != nil {
			return nil, err
		}
		return ttOut(r)
	})
	registerTestRunner("chisquare", func(c testCase) (outputs, error) {
		o := &ChiSquareOptions{DDOF: int(c.num("ddof"))}
		if len(c.data) > 1 {
			o.Expected = c.data[1]
		}
		return tr(ChiSquare(c.data[0], o))
	})
	registerTestRunner("power_divergence", func(c testCase) (outputs, error) {
		o := &ChiSquareOptions{DDOF: int(c.num("ddof"))}
		if len(c.data) > 1 {
			o.Expected = c.data[1]
		}
		return tr(PowerDivergence(c.data[0], c.num("lambda_"), o))
	})
	registerTestRunner("chi2_contingency", func(c testCase) (outputs, error) {
		shape := make([]int, len(c.data[1]))
		for i, s := range c.data[1] {
			shape[i] = int(s)
		}
		t, err := ndarray.FromData(append([]float64(nil), c.data[0]...), shape...)
		if err != nil {
			return nil, err
		}
		o := &Chi2ContingencyOptions{NoCorrection: !c.flag("correction")}
		if c.str("lambda_") != "none" {
			l := c.num("lambda_")
			o.Lambda = &l
		}
		r, err := Chi2Contingency(t, o)
		if err != nil {
			return nil, err
		}
		e, _ := ndarray.Data[float64](r.Expected)
		return outputs{"statistic": r.Statistic, "pvalue": r.PValue, "dof": float64(r.DOF), "expected": e}, nil
	})
	registerTestRunner("fisher_exact", func(c testCase) (outputs, error) {
		d := c.data[0]
		t := [2][2]int64{{int64(d[0]), int64(d[1])}, {int64(d[2]), int64(d[3])}}
		return tr(FisherExact(t, &FisherExactOptions{Alternative: c.alt()}))
	})
	registerTestRunner("pearsonr", func(c testCase) (outputs, error) {
		r, err := Pearsonr(c.data[0], c.data[1], &CorrelationOptions{Alternative: c.alt()})
		if err != nil {
			return nil, err
		}
		o := outputs{"statistic": r.Statistic, "pvalue": r.PValue}
		for _, lv := range []float64{0.95, 0.9, 0.5, 0.999} {
			ci, err := r.ConfidenceInterval(lv)
			if err != nil {
				return nil, err
			}
			o[fmt.Sprintf("ci_lo_%v", lv)] = ci.Low
			o[fmt.Sprintf("ci_hi_%v", lv)] = ci.High
		}
		return o, nil
	})
	registerTestRunner("spearmanr", func(c testCase) (outputs, error) {
		return tr(Spearmanr(c.data[0], c.data[1], &CorrelationOptions{Alternative: c.alt()}))
	})
	registerTestRunner("kendalltau", func(c testCase) (outputs, error) {
		o := &KendallOptions{Alternative: c.alt(), Method: c.method()}
		if c.str("variant") == "c" {
			o.Variant = TauC
		}
		return tr(Kendalltau(c.data[0], c.data[1], o))
	})
	registerTestRunner("linregress", func(c testCase) (outputs, error) {
		r, err := Linregress(c.data[0], c.data[1], &CorrelationOptions{Alternative: c.alt()})
		return outputs{"slope": r.Slope, "intercept": r.Intercept, "rvalue": r.RValue,
			"pvalue": r.PValue, "stderr": r.StdErr, "intercept_stderr": r.InterceptStdErr}, err
	})
	registerTestRunner("rankdata", func(c testCase) (outputs, error) {
		m := map[string]RankMethod{"average": RankAverage, "min": RankMin, "max": RankMax,
			"dense": RankDense, "ordinal": RankOrdinal}[c.str("method")]
		r, err := RankData(c.data[0], m)
		return outputs{"ranks": r}, err
	})
	registerTestRunner("kstwo_sf", func(c testCase) (outputs, error) {
		n, d := int(c.data[0][0]), c.data[0][1]
		return outputs{"sf": kstwoSF(n, d), "cdf": kstwoCDF(n, d)}, nil
	})
	registerTestRunner("smirnov", func(c testCase) (outputs, error) {
		return outputs{"sf": smirnovSF(int(c.data[0][0]), c.data[0][1])}, nil
	})
	registerTestRunner("kolmogorov", func(c testCase) (outputs, error) {
		return outputs{"sf": kolmogorovSF(c.data[0][0])}, nil
	})
	ksOut := func(r KSResult, err error) (outputs, error) {
		sign := float64(r.StatisticSign)
		if r.Statistic != r.Statistic {
			sign = math.NaN()
		}
		return outputs{"statistic": r.Statistic, "pvalue": r.PValue,
			"statistic_location": r.StatisticLocation, "statistic_sign": sign}, err
	}
	registerTestRunner("ks_1samp", func(c testCase) (outputs, error) {
		cdf := Normal{Scale: 1}.CDF
		if c.str("cdf") == "t4" {
			cdf = StudentT{Df: 4, Loc: 0.1, Scale: 1.3}.CDF
		}
		return ksOut(KSTest1Samp(c.data[0], cdf, &KSOptions{Alternative: c.alt(), Method: c.method()}))
	})
	registerTestRunner("ks_2samp", func(c testCase) (outputs, error) {
		return ksOut(KSTest2Samp(c.data[0], c.data[1], &KSOptions{Alternative: c.alt(), Method: c.method()}))
	})
}

// kstwoCDF is P(D_n <= d), for the fixtures (scipy.stats.kstwo.cdf).
func kstwoCDF(n int, d float64) float64 {
	switch {
	case d <= 0.5/float64(n):
		return 0
	case d >= 1:
		return 1
	}
	return kolmogn(n, d, true)
}

func rankOut(r RankTestResult, err error) (outputs, error) {
	return outputs{"statistic": r.Statistic, "pvalue": r.PValue, "zstatistic": r.ZStatistic}, err
}

func init() {
	registerTestRunner("mannwhitneyu", func(c testCase) (outputs, error) {
		return rankOut(MannWhitneyU(c.data[0], c.data[1], &MannWhitneyUOptions{
			Alternative: c.alt(), Method: c.method(), NoContinuity: !c.flag("use_continuity")}))
	})
	zm := map[string]ZeroMethod{"wilcox": ZeroWilcox, "pratt": ZeroPratt, "zsplit": ZeroZsplit}
	registerTestRunner("wilcoxon", func(c testCase) (outputs, error) {
		return rankOut(Wilcoxon(c.data[0], c.data[1], &WilcoxonOptions{
			ZeroMethod: zm[c.str("zero_method")], Correction: c.flag("correction"),
			Alternative: c.alt(), Method: c.method()}))
	})
	registerTestRunner("wilcoxon1", func(c testCase) (outputs, error) {
		return rankOut(Wilcoxon(arr(c.data[0]), nil, nil))
	})
	registerTestRunner("kruskal", func(c testCase) (outputs, error) {
		return tr(KruskalWallis(c.data...))
	})
	registerTestRunner("shapiro", func(c testCase) (outputs, error) { return tr(ShapiroWilk(c.data[0])) })
	registerTestRunner("skewtest", func(c testCase) (outputs, error) {
		return tr(SkewTest(c.data[0], &SkewTestOptions{Alternative: c.alt()}))
	})
	registerTestRunner("kurtosistest", func(c testCase) (outputs, error) {
		return tr(KurtosisTest(c.data[0], &SkewTestOptions{Alternative: c.alt()}))
	})
	registerTestRunner("normaltest", func(c testCase) (outputs, error) { return tr(Normaltest(c.data[0])) })
	registerTestRunner("jarque_bera", func(c testCase) (outputs, error) { return tr(JarqueBera(c.data[0])) })
	ads := map[string]ADDist{"norm": ADNorm, "expon": ADExpon, "logistic": ADLogistic,
		"gumbel_l": ADGumbelL, "gumbel_r": ADGumbelR}
	registerTestRunner("anderson", func(c testCase) (outputs, error) {
		r, err := AndersonDarling(c.data[0], ads[c.str("dist")])
		return outputs{"statistic": r.Statistic, "pvalue": r.PValue, "critical_values": r.CriticalValues,
			"significance_level": r.SignificanceLevels, "loc": r.Loc, "scale": r.Scale}, err
	})
	registerTestRunner("f_oneway", func(c testCase) (outputs, error) {
		return tr(FOneway(&FOnewayOptions{Welch: !c.flag("equal_var")}, c.data...))
	})
	registerTestRunner("bartlett", func(c testCase) (outputs, error) { return tr(Bartlett(c.data...)) })
	centers := map[string]LeveneCenter{"median": CenterMedian, "mean": CenterMean, "trimmed": CenterTrimmed}
	registerTestRunner("levene", func(c testCase) (outputs, error) {
		return tr(Levene(&LeveneOptions{Center: centers[c.str("center")], ProportionToCut: c.num("proportiontocut")}, c.data...))
	})
}
