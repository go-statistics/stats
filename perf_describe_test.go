package stats

func init() {
	perfCases["describe"] = perfCase{n: 1_000_000, setup: func(n int) (func(), func()) {
		x := normalDraws(n, 1)
		return nil, func() { Describe(x, nil) }
	}}
	perfCases["percentile"] = perfCase{n: 1_000_000, setup: func(n int) (func(), func()) {
		x := normalDraws(n, 1)
		return nil, func() { Percentiles(x, []float64{5, 25, 50, 75, 95}, nil) }
	}}
}
