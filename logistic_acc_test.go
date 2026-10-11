package stats

func init() {
	d := Logistic{Scale: 1}
	register("logistic_pdf", func(a []float64) float64 { return d.PDF(a[0]) }, nil)
	register("logistic_logpdf", func(a []float64) float64 { return d.LogPDF(a[0]) }, nil)
	register("logistic_cdf", func(a []float64) float64 { return d.CDF(a[0]) }, nil)
	register("logistic_sf", func(a []float64) float64 { return d.SF(a[0]) }, nil)
	register("logistic_logcdf", func(a []float64) float64 { return d.LogCDF(a[0]) }, nil)
	register("logistic_logsf", func(a []float64) float64 { return d.LogSF(a[0]) }, nil)
	register("logistic_ppf", func(a []float64) float64 { return d.PPF(a[0]) }, nil)
	register("logistic_isf", func(a []float64) float64 { return d.ISF(a[0]) }, nil)
}
