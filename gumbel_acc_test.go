package stats

func init() {
	d := Gumbel{Scale: 1}
	register("gumbel_r_pdf", func(a []float64) float64 { return d.PDF(a[0]) }, nil)
	register("gumbel_r_logpdf", func(a []float64) float64 { return d.LogPDF(a[0]) }, nil)
	register("gumbel_r_cdf", func(a []float64) float64 { return d.CDF(a[0]) }, nil)
	register("gumbel_r_sf", func(a []float64) float64 { return d.SF(a[0]) }, nil)
	register("gumbel_r_logcdf", func(a []float64) float64 { return d.LogCDF(a[0]) }, nil)
	register("gumbel_r_logsf", func(a []float64) float64 { return d.LogSF(a[0]) }, nil)
	register("gumbel_r_ppf", func(a []float64) float64 { return d.PPF(a[0]) }, nil)
	register("gumbel_r_isf", func(a []float64) float64 { return d.ISF(a[0]) }, nil)
}
