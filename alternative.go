package stats

// Alternative is the alternative hypothesis of a test or the side of a
// confidence interval, SciPy's alternative= argument. The zero value is
// TwoSided, SciPy's default.
type Alternative int

const (
	// TwoSided is alternative="two-sided".
	TwoSided Alternative = iota
	// Less is alternative="less": the statistic (or the location of the
	// first sample) is less than under the null hypothesis.
	Less
	// Greater is alternative="greater".
	Greater
)

// String returns SciPy's name for the alternative.
func (a Alternative) String() string {
	switch a {
	case TwoSided:
		return "two-sided"
	case Less:
		return "less"
	case Greater:
		return "greater"
	}
	return "invalid"
}
