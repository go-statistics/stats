package stats

func init() {
	d := func(a []float64) Pareto { return Pareto{B: a[1], Scale: 1} }
	register("pareto_pdf", func(a []float64) float64 { return d(a).PDF(a[0]) }, nil)
	register("pareto_logpdf", func(a []float64) float64 { return d(a).LogPDF(a[0]) }, nil)
	register("pareto_cdf", func(a []float64) float64 { return d(a).CDF(a[0]) }, nil)
	register("pareto_sf", func(a []float64) float64 { return d(a).SF(a[0]) }, nil)
	register("pareto_logcdf", func(a []float64) float64 { return d(a).LogCDF(a[0]) }, nil)
	register("pareto_logsf", func(a []float64) float64 { return d(a).LogSF(a[0]) }, nil)
	register("pareto_ppf", func(a []float64) float64 { return d(a).PPF(a[0]) }, nil)
	register("pareto_isf", func(a []float64) float64 { return d(a).ISF(a[0]) }, nil)
}
