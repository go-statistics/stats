package stats

import (
	"github.com/go-numerics/special"
	"github.com/go-statistics/stats/internal/dd"
)

// Double-double log-gamma and digamma for the Dirichlet distribution, whose
// log-density and entropy are small differences of large log-gamma and
// digamma terms when the concentrations are large.

// lgammaDD is log Γ(a) for a > 0: Stirling's series in double-double for
// a ≥ 30, where the value is large and float64 would lose its ULPs in the
// differences; special.LogGamma below (value at most about 71).
func lgammaDD(a float64) dd.DD {
	if a < stirlingMin {
		return dd.F(special.LogGamma(a))
	}
	h, l := dd.TwoSum(a, -0.5)
	t := dd.Mul(dd.DD{Hi: h, Lo: l}, dd.LogAccurate(a))
	t = dd.AddF(t, -a)
	t = dd.Add(t, halfLog2Pi)
	return dd.AddF(t, stirlingCorr(a))
}

// lgammaDDAt is log Γ(a) at the double-double a (its low part a first-order
// correction ψ(a)·a.Lo).
func lgammaDDAt(a dd.DD) dd.DD {
	return dd.AddF(lgammaDD(a.Hi), special.Digamma(a.Hi)*a.Lo)
}

// digammaDD is ψ(a) for a > 0: log a - 1/(2a) - Σ B₂ₖ/(2k·a²ᵏ) with the
// logarithm in double-double for a ≥ 30, special.Digamma below.
func digammaDD(a float64) dd.DD {
	if a < stirlingMin {
		return dd.F(special.Digamma(a))
	}
	r := 1 / a
	r2 := r * r
	series := r2 * (1.0/12 - r2*(1.0/120-r2*(1.0/252-r2*(1.0/240-r2*(1.0/132)))))
	t := dd.Sub(dd.LogAccurate(a), dd.DivF(dd.F(0.5), a))
	return dd.AddF(t, -series)
}

// digammaDDAt is ψ at the double-double a (first-order correction
// ψ'(a)·a.Lo ≈ a.Lo/a).
func digammaDDAt(a dd.DD) dd.DD {
	return dd.AddF(digammaDD(a.Hi), a.Lo/a.Hi)
}

// sumDD is the double-double sum of v.
func sumDD(v []float64) dd.DD {
	var s dd.DD
	for _, x := range v {
		s = dd.AddF(s, x)
	}
	return s
}
