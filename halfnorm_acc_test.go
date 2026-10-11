package stats

func init() {
	d := HalfNormal{Scale: 1}
	register("halfnorm_pdf", func(a []float64) float64 { return d.PDF(a[0]) }, nil)
	register("halfnorm_logpdf", func(a []float64) float64 { return d.LogPDF(a[0]) }, nil)
	register("halfnorm_cdf", func(a []float64) float64 { return d.CDF(a[0]) }, nil)
	register("halfnorm_sf", func(a []float64) float64 { return d.SF(a[0]) }, nil)
	register("halfnorm_logcdf", func(a []float64) float64 { return d.LogCDF(a[0]) }, nil)
	register("halfnorm_logsf", func(a []float64) float64 { return d.LogSF(a[0]) }, nil)
	register("halfnorm_ppf", func(a []float64) float64 { return d.PPF(a[0]) }, nil)
	register("halfnorm_isf", func(a []float64) float64 { return d.ISF(a[0]) }, nil)
}
