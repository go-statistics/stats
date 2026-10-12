package stats

func init() {
	d := Laplace{Scale: 1}
	register("laplace_pdf", func(a []float64) float64 { return d.PDF(a[0]) }, nil)
	register("laplace_logpdf", func(a []float64) float64 { return d.LogPDF(a[0]) }, nil)
	register("laplace_cdf", func(a []float64) float64 { return d.CDF(a[0]) }, nil)
	register("laplace_sf", func(a []float64) float64 { return d.SF(a[0]) }, nil)
	register("laplace_logcdf", func(a []float64) float64 { return d.LogCDF(a[0]) }, nil)
	register("laplace_logsf", func(a []float64) float64 { return d.LogSF(a[0]) }, nil)
	register("laplace_ppf", func(a []float64) float64 { return d.PPF(a[0]) }, nil)
	register("laplace_isf", func(a []float64) float64 { return d.ISF(a[0]) }, nil)
}
