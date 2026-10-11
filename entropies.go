package stats

import (
	"math"

	"github.com/go-numerics/special"
)

// The entropies of the gamma and beta families. SciPy's formulas,
// log Γ(a) - (a-1)ψ(a) + a and log B(a,b) - (a-1)ψ(a) - (b-1)ψ(b) +
// (a+b-2)ψ(a+b), cancel terms of size a·log a down to a result of size
// log a; SciPy switches to truncated expansions beyond fixed thresholds
// (and for the beta distribution leaves a gap where neither applies and
// returns 0: scipy.stats.beta(3, 2e7).entropy() is 0.0, the true value
// -14.96). Here the large terms are cancelled analytically: with
// Stirling's series log Γ(x) = (x-1/2) log x - x + log(2π)/2 + corr(x) and
// ψ(x) = log x - 1/(2x) - R(x), every x·log x term goes away and what
// remains is a sum of small terms.

// digammaTail is R(x) = log x - 1/(2x) - ψ(x) = Σ B_2k/(2k x^2k), for
// x >= 30 (error below 1e-20 there).
func digammaTail(x float64) float64 {
	r := 1 / (x * x)
	return r * (1.0/12 - r*(1.0/120-r*(1.0/252-r*(1.0/240-r*(1.0/132)))))
}

// gammaEntropyPart is log Γ(a) - (a-1)ψ(a) + a, the entropy of the
// standard gamma distribution.
func gammaEntropyPart(a float64) float64 {
	if a < stirlingMin {
		return special.LogGamma(a) - (a-1)*special.Digamma(a) + a
	}
	// = log(2πa)/2 + 1/2 - 1/(2a) + (a-1)R(a) + corr(a)
	return 0.5*math.Log(2*math.Pi*a) + 0.5 - 0.5/a + (a-1)*digammaTail(a) + stirlingCorr(a)
}

// betaEntropy is the entropy of the standard beta distribution.
func betaEntropy(a, b float64) float64 {
	if a > b {
		a, b = b, a // symmetric in (a, b)
	}
	if b < stirlingMin {
		return special.LogBeta(a, b) - (a-1)*special.Digamma(a) - (b-1)*special.Digamma(b) + (a+b-2)*special.Digamma(a+b)
	}
	s := a + b
	// b (and so s) large: log Γ(b) - log Γ(s) - (b-1)ψ(b) + (s-2)ψ(s)
	// = a + log(b)/2 - 3 log(s)/2 - 1/(2b) + 1/s + (b-1)R(b) - (s-2)R(s)
	//   + corr(b) - corr(s)
	tail := 0.5*math.Log(b) - 1.5*math.Log(s) - 0.5/b + 1/s + (b-1)*digammaTail(b) - (s-2)*digammaTail(s) + stirlingCorr(b) - stirlingCorr(s)
	return gammaEntropyPart(a) + tail
}
