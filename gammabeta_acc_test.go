package stats

// The ULP fixtures of testdata/gen/fam_gammabeta.py: arguments are the
// shape parameters then x (or p).
func init() {
	type fam struct {
		name string
		mk   func(a []float64) Continuous
	}
	fams := []fam{
		{"gamma", func(a []float64) Continuous { return Gamma{A: a[0], Scale: 1} }},
		{"chi2", func(a []float64) Continuous { return ChiSquared{Df: a[0], Scale: 1} }},
		{"chi", func(a []float64) Continuous { return Chi{Df: a[0], Scale: 1} }},
		{"invgamma", func(a []float64) Continuous { return InverseGamma{A: a[0], Scale: 1} }},
		{"beta", func(a []float64) Continuous { return Beta{A: a[0], B: a[1], Scale: 1} }},
		{"t", func(a []float64) Continuous { return StudentT{Df: a[0], Scale: 1} }},
		{"f", func(a []float64) Continuous { return F{Dfn: a[0], Dfd: a[1], Scale: 1} }},
		{"ncx2", func(a []float64) Continuous { return NoncentralChiSquared{Df: a[0], Nc: a[1], Scale: 1} }},
	}
	for _, f := range fams {
		registerContinuous(f.name, f.mk)
	}
}

// registerContinuous registers the eight fixtures <name>_<fn> of a
// continuous family whose rows hold the shapes then the argument.
func registerContinuous(name string, mk func(a []float64) Continuous) {
	fns := map[string]func(d Continuous, x float64) float64{
		"pdf":    func(d Continuous, x float64) float64 { return d.PDF(x) },
		"logpdf": func(d Continuous, x float64) float64 { return d.LogPDF(x) },
		"cdf":    func(d Continuous, x float64) float64 { return d.CDF(x) },
		"sf":     func(d Continuous, x float64) float64 { return d.SF(x) },
		"logcdf": func(d Continuous, x float64) float64 { return d.LogCDF(x) },
		"logsf":  func(d Continuous, x float64) float64 { return d.LogSF(x) },
		"ppf":    func(d Continuous, x float64) float64 { return d.PPF(x) },
		"isf":    func(d Continuous, x float64) float64 { return d.ISF(x) },
	}
	for fn, call := range fns {
		register(name+"_"+fn, func(a []float64) float64 {
			n := len(a) - 1
			return call(mk(a[:n]), a[n])
		}, nil)
	}
}
