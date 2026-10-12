package stats

func init() {
	d := Rayleigh{Scale: 1}
	register("rayleigh_pdf", func(a []float64) float64 { return d.PDF(a[0]) }, nil)
	register("rayleigh_logpdf", func(a []float64) float64 { return d.LogPDF(a[0]) }, nil)
	register("rayleigh_cdf", func(a []float64) float64 { return d.CDF(a[0]) }, nil)
	register("rayleigh_sf", func(a []float64) float64 { return d.SF(a[0]) }, nil)
	register("rayleigh_logcdf", func(a []float64) float64 { return d.LogCDF(a[0]) }, nil)
	register("rayleigh_logsf", func(a []float64) float64 { return d.LogSF(a[0]) }, nil)
	register("rayleigh_ppf", func(a []float64) float64 { return d.PPF(a[0]) }, nil)
	register("rayleigh_isf", func(a []float64) float64 { return d.ISF(a[0]) }, nil)
}
