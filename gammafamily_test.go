package stats

func init() {
	// The draws come from go-ndarray's samplers, which reproduce NumPy's
	// x86-64 build except where the C library's pow/log/exp round
	// differently from Go's (measured: a handful of draws per thousand, by
	// 1 or 2 ulp; see docs/rvs-table.md).
	rvsULP["Gamma"] = 1
	rvsULP["ChiSquared"] = 1
	rvsULP["Chi"] = 1
	rvsULP["Beta"] = 1
	rvsULP["F"] = 2
	// SciPy draws inverse-gamma variates as 1/gammainccinv(a, U): the
	// quantile here is special.GammaQInv, more accurate than SciPy's, so
	// most draws differ in the last bits (by up to 83 ulp near a = 0.8,
	// where the inversion is ill-conditioned).
	rvsULP["InverseGamma"] = 100
}
