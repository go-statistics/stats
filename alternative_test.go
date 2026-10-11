package stats

import "testing"

func TestAlternativeString(t *testing.T) {
	for a, want := range map[Alternative]string{TwoSided: "two-sided", Less: "less", Greater: "greater", Alternative(7): "invalid"} {
		if got := a.String(); got != want {
			t.Errorf("%d: %q, want %q", int(a), got, want)
		}
	}
}
