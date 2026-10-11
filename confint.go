package stats

// ConfidenceInterval is a confidence interval [Low, High] (SciPy's
// ConfidenceInterval(low, high) named tuple). A one-sided interval has an
// infinite bound.
type ConfidenceInterval struct {
	Low, High float64
}
