package stats

func init() {
	n := Normal{Loc: 0, Scale: 1}
	register("norm_pdf", func(a []float64) float64 { return n.PDF(a[0]) }, nil)
	register("norm_logpdf", func(a []float64) float64 { return n.LogPDF(a[0]) }, nil)
	register("norm_cdf", func(a []float64) float64 { return n.CDF(a[0]) }, nil)
	register("norm_sf", func(a []float64) float64 { return n.SF(a[0]) }, nil)
	register("norm_logcdf", func(a []float64) float64 { return n.LogCDF(a[0]) }, nil)
	register("norm_logsf", func(a []float64) float64 { return n.LogSF(a[0]) }, nil)
	register("norm_ppf", func(a []float64) float64 { return n.PPF(a[0]) }, nil)
	register("norm_isf", func(a []float64) float64 { return n.ISF(a[0]) }, nil)
}
