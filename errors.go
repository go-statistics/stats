package stats

import "errors"

// The errors returned by this package wrap one of these, so that a caller
// can test the kind with errors.Is.
var (
	// ErrParam reports a distribution parameter or an option outside its
	// domain (a non-positive scale, a probability outside [0, 1], ...).
	ErrParam = errors.New("stats: parameter out of domain")
	// ErrData reports data a computation cannot use: too few observations,
	// mismatched lengths, a negative count, a NaN where it has no meaning.
	ErrData = errors.New("stats: invalid data")
	// ErrFit reports a fit that did not converge to a finite optimum.
	ErrFit = errors.New("stats: fit did not converge")
)
