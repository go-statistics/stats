package stats

func init() {
	d := func(a []float64) LogNormal { return LogNormal{S: a[1], Scale: 1} }
	register("lognorm_pdf", func(a []float64) float64 { return d(a).PDF(a[0]) }, nil)
	register("lognorm_logpdf", func(a []float64) float64 { return d(a).LogPDF(a[0]) }, nil)
	register("lognorm_cdf", func(a []float64) float64 { return d(a).CDF(a[0]) }, nil)
	register("lognorm_sf", func(a []float64) float64 { return d(a).SF(a[0]) }, nil)
	register("lognorm_logcdf", func(a []float64) float64 { return d(a).LogCDF(a[0]) }, nil)
	register("lognorm_logsf", func(a []float64) float64 { return d(a).LogSF(a[0]) }, nil)
	register("lognorm_ppf", func(a []float64) float64 { return d(a).PPF(a[0]) }, nil)
	register("lognorm_isf", func(a []float64) float64 { return d(a).ISF(a[0]) }, nil)
}
