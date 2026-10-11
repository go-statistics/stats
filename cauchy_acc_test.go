package stats

func init() {
	d := Cauchy{Scale: 1}
	register("cauchy_pdf", func(a []float64) float64 { return d.PDF(a[0]) }, nil)
	register("cauchy_logpdf", func(a []float64) float64 { return d.LogPDF(a[0]) }, nil)
	register("cauchy_cdf", func(a []float64) float64 { return d.CDF(a[0]) }, nil)
	register("cauchy_sf", func(a []float64) float64 { return d.SF(a[0]) }, nil)
	register("cauchy_logcdf", func(a []float64) float64 { return d.LogCDF(a[0]) }, nil)
	register("cauchy_logsf", func(a []float64) float64 { return d.LogSF(a[0]) }, nil)
	register("cauchy_ppf", func(a []float64) float64 { return d.PPF(a[0]) }, nil)
	register("cauchy_isf", func(a []float64) float64 { return d.ISF(a[0]) }, nil)
}
