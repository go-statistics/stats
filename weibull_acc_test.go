package stats

func init() {
	d := func(a []float64) Weibull { return Weibull{C: a[1], Scale: 1} }
	register("weibull_min_pdf", func(a []float64) float64 { return d(a).PDF(a[0]) }, nil)
	register("weibull_min_logpdf", func(a []float64) float64 { return d(a).LogPDF(a[0]) }, nil)
	register("weibull_min_cdf", func(a []float64) float64 { return d(a).CDF(a[0]) }, nil)
	register("weibull_min_sf", func(a []float64) float64 { return d(a).SF(a[0]) }, nil)
	register("weibull_min_logcdf", func(a []float64) float64 { return d(a).LogCDF(a[0]) }, nil)
	register("weibull_min_logsf", func(a []float64) float64 { return d(a).LogSF(a[0]) }, nil)
	register("weibull_min_ppf", func(a []float64) float64 { return d(a).PPF(a[0]) }, nil)
	register("weibull_min_isf", func(a []float64) float64 { return d(a).ISF(a[0]) }, nil)
}
