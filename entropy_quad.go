package stats

import "math"

// entropyPositive is the differential entropy -∫₀^∞ f log f of a standard
// form on (0, ∞) without a closed form, by exp-sinh quadrature (Takahasi
// and Mori 1974): x = m·exp(π/2·sinh t), which integrates the endpoint
// behaviour at 0 (where f may be infinite) and the tail at ∞ with double
// exponential decay of the error. m is a typical value (the mean). The
// step is halved until two estimates agree to 1e-14.
func entropyPositive(pdf, logpdf func(float64) float64, m float64) float64 {
	eval := func(t float64) float64 {
		x := m * math.Exp(math.Pi/2*math.Sinh(t))
		if x == 0 || math.IsInf(x, 1) {
			return 0
		}
		f := pdf(x)
		if f == 0 {
			return 0
		}
		return -f * logpdf(x) * x * math.Pi / 2 * math.Cosh(t)
	}
	h := 0.5
	sum := eval(0)
	for k := 1; k <= 12; k++ {
		sum += eval(-float64(k)*h) + eval(float64(k)*h)
	}
	est := sum * h
	for level := 0; level < 10; level++ {
		h /= 2
		for k := 1; float64(k)*h <= 6; k += 2 {
			sum += eval(-float64(k)*h) + eval(float64(k)*h)
		}
		next := sum * h
		if math.Abs(next-est) <= 1e-14*math.Abs(next) {
			return next
		}
		est = next
	}
	return est
}
