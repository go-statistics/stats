package stats

func init() {
	// scipy.stats.gaussian_kde(d)(x) with d 10^4 normal draws and x 10^4
	// points: construction included on both sides.
	perfCases["kde.evaluate"] = perfCase{n: 10_000, setup: func(n int) (func(), func()) {
		d := normalDraws(n, 1)
		x := points(n, -4, 4)
		return nil, func() {
			k, _ := NewGaussianKDE(d, nil)
			k.EvaluateSlice(x)
		}
	}}
}
