package stats

func init() {
	const m = 1_000_000
	perfCases["ttest_ind"] = perfCase{n: m, setup: func(n int) (func(), func()) {
		a, b := normalDraws(n, 1), normalDraws(n, 2)
		for i := range b {
			b[i] += 0.01
		}
		return nil, func() { TTestInd(a, b, nil) }
	}}
	perfCases["ttest_1samp"] = perfCase{n: m, setup: func(n int) (func(), func()) {
		a := normalDraws(n, 1)
		return nil, func() { TTest1Samp(a, 0, nil) }
	}}
	perfCases["spearmanr"] = perfCase{n: m, setup: func(n int) (func(), func()) {
		a, b := normalDraws(n, 1), normalDraws(n, 2)
		return nil, func() { Spearmanr(a, b, nil) }
	}}
	perfCases["pearsonr"] = perfCase{n: m, setup: func(n int) (func(), func()) {
		a, b := normalDraws(n, 1), normalDraws(n, 2)
		return nil, func() { Pearsonr(a, b, nil) }
	}}
}
