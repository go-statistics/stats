package stats

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"os/exec"
	"sort"
	"strconv"
	"strings"
	"testing"
)

// The SciPy side of the speed comparison. Every case is defined here, in
// Go: a Python expression building the arguments (a tuple; n is bound) and
// a Python expression for the timed callable (n is bound there too). One `uv run python -c`
// process evaluates them, reading one JSON request per line and answering
// one JSON line: the elapsed wall time of the call, allocation of its
// result included as on the Go side, or null where SciPy raises; with
// "hash", the SHA-256 of the arguments' bytes instead (to check that both
// sides time the same data).

// pyCase is a case's SciPy side.
type pyCase struct {
	Setup string `json:"setup"`
	Call  string `json:"call"`
}

// pyLoop is the whole Python program: arguments are built once per
// (setup, n) and cached across rounds.
const pyLoop = `
import sys, json, time, hashlib, warnings
import numpy as np, scipy.stats as st
warnings.simplefilter("ignore")
cache = {}
for line in sys.stdin:
    c = json.loads(line)
    key = (c["setup"], c["n"])
    if key not in cache:
        cache[key] = eval(c["setup"], {"np": np, "st": st, "n": c["n"]})
    args = cache[key]
    if c.get("hash"):
        h = hashlib.sha256()
        for a in args:
            for d in ([a] if isinstance(a, np.ndarray) else [a.dataset] if hasattr(a, "dataset") else [a.mean, a.cov] if hasattr(a, "cov") else []):
                h.update(np.ascontiguousarray(d).tobytes())
            if not isinstance(a, np.ndarray) and not hasattr(a, "dataset") and not hasattr(a, "cov"):
                h.update(repr(a).encode())
        print(json.dumps({"hash": h.hexdigest()}), flush=True)
        continue
    f = eval(c["call"], {"np": np, "st": st, "n": c["n"]})
    with np.errstate(all="ignore"):
        t0 = time.perf_counter()
        try:
            f(*args)
            t = time.perf_counter() - t0
        except Exception:
            t = None
    print(json.dumps({"t": t}), flush=True)
`

// sciPy is the running Python process.
type sciPy struct {
	cmd *exec.Cmd
	in  *json.Encoder
	out *bufio.Scanner
}

func startSciPy(t *testing.T) *sciPy {
	uv := os.ExpandEnv("$HOME/.local/bin/uv")
	cmd := exec.Command(uv, "run", "--with", "scipy==1.18.1", "--with", "numpy==2.5.4", "python", "-c", pyLoop)
	// One thread on the SciPy side too: the comparison is core for core.
	cmd.Env = append(os.Environ(), "OMP_NUM_THREADS=1", "OPENBLAS_NUM_THREADS=1", "MKL_NUM_THREADS=1", "VECLIB_MAXIMUM_THREADS=1")
	cmd.Stderr = os.Stderr
	stdin, err := cmd.StdinPipe()
	if err != nil {
		t.Fatal(err)
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		t.Fatal(err)
	}
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	s := &sciPy{cmd: cmd, in: json.NewEncoder(stdin), out: bufio.NewScanner(stdout)}
	t.Cleanup(func() { stdin.Close(); cmd.Wait() })
	return s
}

type pyRequest struct {
	pyCase
	N    int  `json:"n"`
	Hash bool `json:"hash,omitempty"`
}

func (s *sciPy) ask(t *testing.T, c pyCase, n int, hash bool) map[string]any {
	if err := s.in.Encode(pyRequest{c, n, hash}); err != nil {
		t.Fatal(err)
	}
	if !s.out.Scan() {
		t.Fatalf("SciPy process: %v", s.out.Err())
	}
	var r map[string]any
	if err := json.Unmarshal(s.out.Bytes(), &r); err != nil {
		t.Fatal(err)
	}
	return r
}

// time returns the seconds SciPy takes, NaN where it raises.
func (s *sciPy) time(t *testing.T, c pyCase, n int) float64 {
	v, ok := s.ask(t, c, n, false)["t"].(float64)
	if !ok {
		return math.NaN()
	}
	return v
}

// Python expressions for the data, the same IEEE operations as points and
// normalDraws (go-ndarray's random reproduces NumPy's generator).
func pyPoints(lo, hi string) string {
	return fmt.Sprintf("((%s) + ((%s) - (%s)) * np.mod(np.arange(n, dtype=np.float64) * 0.6180339887498949, 1.0))", lo, hi, lo)
}

func pyNormal(seed int) string {
	return fmt.Sprintf("np.random.default_rng(%d).standard_normal(n)", seed)
}

func pyPositive(seed int) string { return "(np.abs(" + pyNormal(seed) + ") + 0.5)" }

var (
	pyCounts = "np.floor(" + pyPoints("50", "150") + ")"
	pyN1     = "(" + pyNormal(1) + ",)"
	pyN2     = "(" + pyNormal(1) + ", " + pyNormal(2) + ")"
	pyN3     = "(" + pyNormal(1) + ", " + pyNormal(2) + ", " + pyNormal(3) + ")"
	pyMVN    = "st.multivariate_normal(np.array([0, 1, -1, 0.5, 2.0]), np.array([[2, 0.5, 0.3, 0.1, 0], [0.5, 1.5, 0.2, 0, 0.1], [0.3, 0.2, 1, 0.4, 0.2], [0.1, 0, 0.4, 2.5, 0.3], [0, 0.1, 0.2, 0.3, 1.2]]))"
	pyMVNX   = pyMVN + ".rvs(size=n, random_state=np.random.default_rng(8))"
	pyAlpha  = "np.array([1.5, 2, 3.5, 0.8, 4])"
	pyMNP    = "np.array([0.1, 0.2, 0.3, 0.15, 0.25])"
)

// pyNum formats a float64 so that Python reads the same double (an int
// where it is integral: hypergeom's shapes must be integers).
func pyNum(v float64) string {
	if v == math.Trunc(v) && math.Abs(v) < 1<<53 {
		return strconv.FormatInt(int64(v), 10)
	}
	return strconv.FormatFloat(v, 'g', -1, 64)
}

func pyShapes(a []float64) string {
	s := make([]string, len(a))
	for i, v := range a {
		s[i] = pyNum(v)
	}
	return strings.Join(s, ", ")
}

func pyWithShapes(head string, a []float64) string {
	if len(a) == 0 {
		return head
	}
	return head + ", " + pyShapes(a)
}

// The generic distribution cases.
func pyDist(name, method string, lo, hi float64, disc bool, args []float64) pyCase {
	x := pyPoints(strconv.FormatFloat(lo, 'g', -1, 64), strconv.FormatFloat(hi, 'g', -1, 64))
	if disc {
		x = "np.floor(" + x + ")"
	}
	return pyCase{"(" + x + ",)", "lambda x: st." + name + "." + method + "(" + pyWithShapes("x", args) + ")"}
}

func pyRvs(name string, args []float64) pyCase {
	return pyCase{"()", "lambda: st." + name + ".rvs(" + pyShapes(args) + sep(args) + "size=n, random_state=np.random.default_rng(1))"}
}

func sep(a []float64) string {
	if len(a) == 0 {
		return ""
	}
	return ", "
}

func pyEntropy(name string, args []float64) pyCase {
	return pyCase{"()", "lambda: [st." + name + "(" + pyShapes(args) + ").entropy() for _ in range(100)]"}
}

func pyFit(name string, args []float64) pyCase {
	return pyCase{"(st." + name + ".rvs(" + pyShapes(args) + sep(args) + "size=n, random_state=np.random.default_rng(7)),)", "st." + name + ".fit"}
}

// pyCases are the SciPy sides of the named cases.
var pyCases = map[string]pyCase{
	"norm.pdf":     {"(" + pyPoints("-10", "10") + ",)", "lambda x: st.norm.pdf(x)"},
	"norm.cdf":     {"(" + pyPoints("-10", "10") + ",)", "lambda x: st.norm.cdf(x)"},
	"norm.ppf":     {"(" + pyPoints("1e-12", "1 - 1e-12") + ",)", "lambda p: st.norm.ppf(p)"},
	"beta.pdf":     {"(" + pyPoints("1e-6", "1 - 1e-6") + ",)", "lambda x: st.beta.pdf(x, 2.5, 7.0)"},
	"beta.cdf":     {"(" + pyPoints("1e-6", "1 - 1e-6") + ",)", "lambda x: st.beta.cdf(x, 2.5, 7.0)"},
	"beta.ppf":     {"(" + pyPoints("1e-9", "1 - 1e-9") + ",)", "lambda p: st.beta.ppf(p, 2.5, 7.0)"},
	"gamma.pdf":    {"(" + pyPoints("1e-3", "30") + ",)", "lambda x: st.gamma.pdf(x, 4.5)"},
	"gamma.cdf":    {"(" + pyPoints("1e-3", "30") + ",)", "lambda x: st.gamma.cdf(x, 4.5)"},
	"gamma.ppf":    {"(" + pyPoints("1e-9", "1 - 1e-9") + ",)", "lambda p: st.gamma.ppf(p, 4.5)"},
	"t.cdf":        {"(" + pyPoints("-20", "20") + ",)", "lambda x: st.t.cdf(x, 7.5)"},
	"t.ppf":        {"(" + pyPoints("1e-9", "1 - 1e-9") + ",)", "lambda p: st.t.ppf(p, 7.5)"},
	"chi2.sf":      {"(" + pyPoints("1e-3", "100") + ",)", "lambda x: st.chi2.sf(x, 12.0)"},
	"norm.rvs":     {"(n,)", "lambda n: st.norm.rvs(size=n, random_state=np.random.default_rng(1))"},
	"gamma.rvs":    {"(n,)", "lambda n: st.gamma.rvs(2.5, size=n, random_state=np.random.default_rng(1))"},
	"ttest_ind":    {"(" + pyNormal(1) + ", " + pyNormal(2) + " + 0.01)", "st.ttest_ind"},
	"ttest_1samp":  {pyN1, "lambda a: st.ttest_1samp(a, 0.0)"},
	"spearmanr":    {pyN2, "st.spearmanr"},
	"pearsonr":     {pyN2, "st.pearsonr"},
	"kde.evaluate": {"(" + pyNormal(1) + ", " + pyPoints("-4", "4") + ")", "lambda d, x: st.gaussian_kde(d)(x)"},
	"describe":     {pyN1, "st.describe"},
	"percentile":   {pyN1, "lambda a: np.percentile(a, [5, 25, 50, 75, 95])"},

	"mean":                 {pyN1, "np.mean"},
	"var":                  {pyN1, "np.var"},
	"std":                  {pyN1, "np.std"},
	"median":               {pyN1, "np.median"},
	"quantile":             {pyN1, "lambda a: np.quantile(a, 0.3)"},
	"average":              {"(" + pyNormal(1) + ", " + pyPositive(2) + ")", "lambda a, w: np.average(a, weights=w)"},
	"moment":               {pyN1, "lambda a: st.moment(a, 3)"},
	"skew":                 {pyN1, "st.skew"},
	"kurtosis":             {pyN1, "st.kurtosis"},
	"sem":                  {pyN1, "st.sem"},
	"variation":            {"(" + pyPositive(1) + ",)", "st.variation"},
	"zscore":               {pyN1, "st.zscore"},
	"zmap":                 {pyN2, "st.zmap"},
	"gmean":                {"(" + pyPositive(1) + ",)", "st.gmean"},
	"hmean":                {"(" + pyPositive(1) + ",)", "st.hmean"},
	"pmean":                {"(" + pyPositive(1) + ",)", "lambda a: st.pmean(a, 2.5)"},
	"iqr":                  {pyN1, "st.iqr"},
	"mad":                  {pyN1, "st.median_abs_deviation"},
	"mode":                 {"(np.round(" + pyNormal(1) + " * 10),)", "st.mode"},
	"tmean":                {pyN1, "lambda a: st.tmean(a, (-1, 1))"},
	"tvar":                 {pyN1, "lambda a: st.tvar(a, (-1, 1))"},
	"tstd":                 {pyN1, "lambda a: st.tstd(a, (-1, 1))"},
	"tsem":                 {pyN1, "lambda a: st.tsem(a, (-1, 1))"},
	"tmin":                 {pyN1, "lambda a: st.tmin(a, -1)"},
	"tmax":                 {pyN1, "lambda a: st.tmax(a, 1)"},
	"trim_mean":            {pyN1, "lambda a: st.trim_mean(a, 0.1)"},
	"trimboth":             {pyN1, "lambda a: st.trimboth(a, 0.1)"},
	"trim1":                {pyN1, "lambda a: st.trim1(a, 0.1, 'right')"},
	"rankdata":             {pyN1, "st.rankdata"},
	"histogram":            {pyN1, "lambda a: np.histogram(a, 100)"},
	"histogram_bin_edges":  {pyN1, "lambda a: np.histogram_bin_edges(a, 'auto')"},
	"cumfreq":              {pyN1, "lambda a: st.cumfreq(a, 100)"},
	"relfreq":              {pyN1, "lambda a: st.relfreq(a, 100)"},
	"binned_statistic":     {pyN2, "lambda a, b: st.binned_statistic(a, b, 'mean', 50)"},
	"entropy":              {"(" + pyPositive(1) + ",)", "st.entropy"},
	"differential_entropy": {pyN1, "st.differential_entropy"},
	"ttest_rel":            {pyN2, "st.ttest_rel"},
	"linregress":           {pyN2, "st.linregress"},
	"kendalltau":           {pyN2, "st.kendalltau"},
	"mannwhitneyu":         {pyN2, "st.mannwhitneyu"},
	"wilcoxon":             {pyN2, "st.wilcoxon"},
	"kruskal":              {pyN3, "st.kruskal"},
	"f_oneway":             {pyN3, "st.f_oneway"},
	"bartlett":             {pyN3, "st.bartlett"},
	"levene":               {pyN3, "st.levene"},
	"ks_1samp":             {pyN1, "lambda a: st.ks_1samp(a, st.norm.cdf)"},
	"ks_2samp":             {pyN2, "st.ks_2samp"},
	"shapiro":              {pyN1, "st.shapiro"},
	"anderson":             {pyN1, "lambda a: st.anderson(a, 'norm')"},
	"skewtest":             {pyN1, "st.skewtest"},
	"kurtosistest":         {pyN1, "st.kurtosistest"},
	"normaltest":           {pyN1, "st.normaltest"},
	"jarque_bera":          {pyN1, "st.jarque_bera"},
	"chisquare":            {"(" + pyCounts + ",)", "st.chisquare"},
	"power_divergence":     {"(" + pyCounts + ",)", "lambda a: st.power_divergence(a, lambda_='cressie-read')"},
	"chi2_contingency":     {"(" + pyCounts + ".reshape(100, n // 100),)", "st.chi2_contingency"},
	"fisher_exact":         {"()", "lambda: st.fisher_exact([[1200, 800], [900, 1100]])"},
	"bootstrap":            {pyN1, "lambda a: st.bootstrap((a,), np.mean, n_resamples=9999, rng=np.random.default_rng(3))"},
	"kde.logpdf":           {"(" + pyNormal(1) + ", " + pyPoints("-4", "4") + ")", "lambda d, x: st.gaussian_kde(d).logpdf(x)"},
	"kde.evaluate2d": {"(np.vstack([" + pyNormal(1) + ", " + pyNormal(2) + "]), np.vstack([" + pyPoints("-4", "4") + ", " + pyPoints("-3", "3") + "]))",
		"lambda d, x: st.gaussian_kde(d)(x)"},
	"kde.resample":         {"(st.gaussian_kde(np.random.default_rng(1).standard_normal(10_000)), n)", "lambda k, n: k.resample(n, seed=np.random.default_rng(4))"},
	"kde.integrate_box_1d": {"(st.gaussian_kde(" + pyNormal(1) + "),)", "lambda k: k.integrate_box_1d(-1, 0.5)"},
	"mvn.logpdf":           {"(" + pyMVN + ", " + pyMVNX + ")", "lambda d, x: d.logpdf(x)"},
	"mvn.rvs":              {"(" + pyMVN + ", n)", "lambda d, n: d.rvs(size=n, random_state=np.random.default_rng(5))"},
	"mvn.cdf":              {"(" + pyMVN + ",)", "lambda d: d.cdf([0.5, 0.2, 1, -0.3, 0.8])"},
	"mvn.fit":              {"(" + pyMVNX + ",)", "st.multivariate_normal.fit"},
	"dirichlet.logpdf": {"(st.dirichlet.rvs(" + pyAlpha + ", size=n, random_state=np.random.default_rng(6)).T,)",
		"lambda x: st.dirichlet.logpdf(x, " + pyAlpha + ")"},
	"dirichlet.rvs": {"(n,)", "lambda n: st.dirichlet.rvs(" + pyAlpha + ", size=n, random_state=np.random.default_rng(6))"},
	"multinomial.logpmf": {"(st.multinomial.rvs(20, " + pyMNP + ", size=n, random_state=np.random.default_rng(6)),)",
		"lambda x: st.multinomial.logpmf(x, 20, " + pyMNP + ")"},
	"multinomial.rvs": {"(n,)", "lambda n: st.multinomial.rvs(20, " + pyMNP + ", size=n, random_state=np.random.default_rng(6))"},
}

var perfDataHash = flag.String("perf.datahash", "", "write the SHA-256 of every case's SciPy arguments to this file (no timing)")

// TestPerfSciPyData checks that every case has a SciPy side whose arguments
// build, and with -perf.datahash records their hashes (to compare the data
// two versions of the harness give SciPy).
func TestPerfSciPyData(t *testing.T) {
	if *perfDataHash == "" {
		t.Skip("set -perf.datahash=PATH")
	}
	sp := startSciPy(t)
	var names []string
	for k := range perfCases {
		names = append(names, k)
	}
	sort.Strings(names)
	var b strings.Builder
	for _, name := range names {
		c := perfCases[name]
		py := c.py
		if py == (pyCase{}) {
			var ok bool
			if py, ok = pyCases[name]; !ok {
				t.Fatalf("%s: no SciPy side", name)
			}
		}
		fmt.Fprintf(&b, "%s %d %v\n", name, c.n, sp.ask(t, py, c.n, true)["hash"])
	}
	if err := os.WriteFile(*perfDataHash, []byte(b.String()), 0o644); err != nil {
		t.Fatal(err)
	}
}
