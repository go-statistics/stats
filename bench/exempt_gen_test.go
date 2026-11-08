package bench

import (
	"flag"
	"fmt"
	"os"
	"sort"
	"strings"
	"testing"

	"github.com/go-scikit/refbench"
)

var exemptFlag = flag.Bool("perf.exempt", false, "TestGenExempt: write ../docs/perf-exempt.md from the public functions without a row")

// section is a group of public functions without a row and the reason of the
// group. A function belongs to the first section whose match accepts it.
type section struct {
	title, why string
	match      func(name string) bool
}

func method(name string) (typ, m string) {
	typ, m, _ = strings.Cut(name, ".")
	return typ, m
}

func in(set ...string) func(string) bool {
	return func(name string) bool {
		for _, s := range set {
			if s == name {
				return true
			}
		}
		return false
	}
}

func methodIn(set ...string) func(string) bool {
	return func(name string) bool {
		if !strings.Contains(name, ".") {
			return false
		}
		_, m := method(name)
		for _, s := range set {
			if s == m {
				return true
			}
		}
		return false
	}
}

var distTypes = func() map[string]bool {
	m := map[string]bool{}
	for _, pd := range perfDists {
		m[typeName(pd.d)] = true
	}
	return m
}()

func distMethod(set ...string) func(string) bool {
	in := methodIn(set...)
	return func(name string) bool { typ, _ := method(name); return distTypes[typ] && in(name) }
}

var sections = []section{
	{"Distribution attributes (one scalar per call)",
		"closed-form scalar of the frozen distribution; SciPy's counterpart is a Python method call (`rv.mean()`, `rv.support()`), not an array operation, so there is no ufunc to time against and the measurement would be Python call overhead",
		distMethod("Mean", "Var", "Std", "Skew", "Kurtosis", "Moment", "Support")},
	{"Distribution quantile helpers (one scalar per call)",
		"a scalar call of PPF, which is measured as the `ppf` row of the distribution (`rv.median()` and `rv.interval()` are Python method calls)",
		distMethod("Median", "Interval")},
	{"Parameter validation",
		"checks the parameters of the value; no SciPy counterpart (SciPy checks inside each call)",
		distMethod("Validate")},
	{"Constructors",
		"validates and stores the parameters in constant time, no array work; SciPy freezes a distribution with a Python call",
		func(n string) bool { return strings.HasPrefix(n, "New") && !strings.Contains(n, ".") }},
	{"Kernel density estimate: attributes and factors",
		"attribute or bandwidth factor of the fitted estimate, computed at construction (the `kde.*` rows time the construction)",
		func(n string) bool {
			return strings.HasPrefix(n, "GaussianKDE.") && in("GaussianKDE.ChoCov", "GaussianKDE.Covariance", "GaussianKDE.D",
				"GaussianKDE.Dataset", "GaussianKDE.Factor", "GaussianKDE.InvCov", "GaussianKDE.LogDet", "GaussianKDE.N",
				"GaussianKDE.Neff", "GaussianKDE.Weights", "GaussianKDE.ScottsFactor", "GaussianKDE.SilvermanFactor",
				"GaussianKDE.SetBandwidth")(n)
		}},
	{"Kernel density estimate: variants of a measured operation",
		"a variant of `kde.evaluate`, `kde.logpdf` or `kde.integrate_box_1d` (same kernel sums); follow-up row",
		func(n string) bool {
			return in("GaussianKDE.LogPDF", "GaussianKDE.IntegrateBox", "GaussianKDE.IntegrateGaussian",
				"GaussianKDE.IntegrateKDE", "GaussianKDE.Marginal")(n)
		}},
	{"Multivariate families: attributes and single-point calls",
		"an attribute of the frozen family or one point per call; the batch methods (`LogPDFPoints`, `LogPMFPoints`, `Rvs`) are the rows `mvn.*`, `dirichlet.*`, `multinomial.*`",
		func(n string) bool {
			typ, _ := method(n)
			return (typ == "MultivariateNormal" || typ == "Dirichlet" || typ == "Multinomial") && in(
				"Dirichlet.Alpha", "Dirichlet.Cov", "Dirichlet.Entropy", "Dirichlet.LogPDF", "Dirichlet.Mean", "Dirichlet.PDF",
				"Dirichlet.PDFPoints", "Dirichlet.Var", "Multinomial.Cov", "Multinomial.Entropy", "Multinomial.LogPMF",
				"Multinomial.Mean", "Multinomial.N", "Multinomial.P", "Multinomial.PMF", "Multinomial.PMFPoints",
				"MultivariateNormal.Cov", "MultivariateNormal.Dim", "MultivariateNormal.Entropy", "MultivariateNormal.LogPDF",
				"MultivariateNormal.LogPDet", "MultivariateNormal.Mean", "MultivariateNormal.PDF",
				"MultivariateNormal.PDFPoints", "MultivariateNormal.Rank")(n)
		}},
	{"Multivariate normal: variants of the cdf",
		"a variant of `mvn.cdf` (the randomised lattice rule); follow-up row",
		in("MultivariateNormal.CDFEstimate", "MultivariateNormal.CDFPoints", "MultivariateNormal.LogCDF")},
	{"Variants of a measured function",
		"a variant of a function that has a row (`percentile`, `quantile`, `ttest_*`, `pearsonr`, `Map/*`)",
		in("MapSlice", "Percentile", "Quantiles", "PearsonResult.ConfidenceInterval", "TTestResult.ConfidenceInterval")},
	{"Multiple testing (follow-up rows)",
		"scipy.stats.false_discovery_control and statsmodels' multipletests are the references; no row yet",
		in("FalseDiscoveryControl", "MultipleTests")},
}

func TestGenExempt(t *testing.T) {
	if !*exemptFlag {
		t.Skip("set -perf.exempt to write ../docs/perf-exempt.md")
	}
	public, err := refbench.PublicFuncs("..")
	if err != nil {
		t.Fatal(err)
	}
	cov := refbench.CheckCoverage(public, cases(nil), nil)
	by := make([][]string, len(sections)+1)
	for _, name := range cov.Missing {
		placed := false
		for i, s := range sections {
			if s.match(name) {
				by[i] = append(by[i], name)
				placed = true
				break
			}
		}
		if !placed {
			by[len(sections)] = append(by[len(sections)], name)
		}
	}
	var b strings.Builder
	b.WriteString("# Public functions without a performance case\n\n")
	b.WriteString("Written by `cd bench && go test -run TestGenExempt -perf.exempt .` and read by `refbench` in the CI mode\n")
	b.WriteString("(`go test -run TestPerf .`): every public function of the module has a case in `bench/perf_test.go` or is\n")
	b.WriteString("named here with its reason. The method and the table are in [perf.md](perf.md). A row here is a debt or a\n")
	b.WriteString("deliberate absence, not a verdict.\n\n")
	fmt.Fprintf(&b, "Counted from the module's own registry: %d public functions and methods, %d with a row, %d named below.\n\n",
		len(public), len(public)-len(cov.Missing), len(cov.Missing))
	for i, s := range sections {
		if len(by[i]) == 0 {
			continue
		}
		sort.Strings(by[i])
		fmt.Fprintf(&b, "## %s\n\n", s.title)
		for _, n := range by[i] {
			fmt.Fprintf(&b, "- `%s`: %s.\n", n, s.why)
		}
		b.WriteString("\n")
	}
	if rest := by[len(sections)]; len(rest) > 0 {
		sort.Strings(rest)
		t.Errorf("%d functions have no section:\n%s", len(rest), strings.Join(rest, "\n"))
	}
	if err := os.WriteFile("../docs/perf-exempt.md", []byte(b.String()), 0o644); err != nil {
		t.Fatal(err)
	}
}
