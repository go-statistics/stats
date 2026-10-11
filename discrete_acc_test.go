package stats

// The accuracy fixtures of the discrete distributions (testdata/gen/
// fam_binom.py, fam_poisson.py, ...): rows (k, shapes...) for the masses and
// distribution functions, (probability, shapes...) for the quantiles.

func registerDiscrete(prefix string, mk func(shape []float64) Discrete) {
	fns := map[string]func(d Discrete, x float64) float64{
		"pmf": Discrete.PMF, "logpmf": Discrete.LogPMF, "cdf": Discrete.CDF, "sf": Discrete.SF,
		"logcdf": Discrete.LogCDF, "logsf": Discrete.LogSF, "ppf": Discrete.PPF, "isf": Discrete.ISF,
	}
	for name, fn := range fns {
		register(prefix+"_"+name, func(a []float64) float64 { return fn(mk(a[1:]), a[0]) }, nil)
	}
}

func init() {
	registerDiscrete("binom", func(s []float64) Discrete { return Binomial{N: int(s[0]), P: s[1]} })
	registerDiscrete("bernoulli", func(s []float64) Discrete { return Bernoulli{P: s[0]} })
	registerDiscrete("poisson", func(s []float64) Discrete { return Poisson{Mu: s[0]} })
	registerDiscrete("geom", func(s []float64) Discrete { return Geometric{P: s[0]} })
	registerDiscrete("nbinom", func(s []float64) Discrete { return NegativeBinomial{N: s[0], P: s[1]} })
	registerDiscrete("hypergeom", func(s []float64) Discrete {
		return Hypergeometric{Total: int(s[0]), Good: int(s[1]), Draws: int(s[2])}
	})
	registerDiscrete("randint", func(s []float64) Discrete { return DiscreteUniform{Low: int(s[0]), High: int(s[1])} })
	registerDiscrete("zipf", func(s []float64) Discrete { return Zipf{A: s[0]} })
}
