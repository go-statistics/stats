package stats

func init() {
	d := Exponential{Scale: 1}
	register("expon_pdf", func(a []float64) float64 { return d.PDF(a[0]) }, nil)
	register("expon_logpdf", func(a []float64) float64 { return d.LogPDF(a[0]) }, nil)
	register("expon_cdf", func(a []float64) float64 { return d.CDF(a[0]) }, nil)
	register("expon_sf", func(a []float64) float64 { return d.SF(a[0]) }, nil)
	register("expon_logcdf", func(a []float64) float64 { return d.LogCDF(a[0]) }, nil)
	register("expon_logsf", func(a []float64) float64 { return d.LogSF(a[0]) }, nil)
	register("expon_ppf", func(a []float64) float64 { return d.PPF(a[0]) }, nil)
	register("expon_isf", func(a []float64) float64 { return d.ISF(a[0]) }, nil)
}
