package stats

func init() {
	d := func(a []float64) Reciprocal { return Reciprocal{A: a[1], B: a[2], Scale: 1} }
	register("reciprocal_pdf", func(a []float64) float64 { return d(a).PDF(a[0]) }, nil)
	register("reciprocal_logpdf", func(a []float64) float64 { return d(a).LogPDF(a[0]) }, nil)
	register("reciprocal_cdf", func(a []float64) float64 { return d(a).CDF(a[0]) }, nil)
	register("reciprocal_sf", func(a []float64) float64 { return d(a).SF(a[0]) }, nil)
	register("reciprocal_logcdf", func(a []float64) float64 { return d(a).LogCDF(a[0]) }, nil)
	register("reciprocal_logsf", func(a []float64) float64 { return d(a).LogSF(a[0]) }, nil)
	register("reciprocal_ppf", func(a []float64) float64 { return d(a).PPF(a[0]) }, nil)
	register("reciprocal_isf", func(a []float64) float64 { return d(a).ISF(a[0]) }, nil)
}
