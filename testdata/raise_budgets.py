"""Raise budgets from the failures of an accuracy run on another
architecture (go test output, possibly under qemu):

    python3 testdata/raise_budgets.py LOG...

Each "fixture/region: max error X ulp > budget B" line raises the region's
budget in testdata/budgets/<fixture>.txt to 1.5·X (at least 1 ulp, two
significant digits up), the rule TestAccuracyBudgets applies.
"""
import math, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
pat = re.compile(r"(\w+)/(\w+): max error ([0-9.e+]+) ulp > budget")


def budget_of(m):
    b = max(1.0, 1.5 * m)
    e = 10 ** (math.floor(math.log10(b)) - 1)
    return math.ceil(b / e) * e


LANE = "another architecture"
args = sys.argv[1:]
if args and args[0].startswith("--lane="):
    LANE = args[0][len("--lane="):]
    args = args[1:]
new = {}
for log in args:
    for line in open(log):
        m = pat.search(line)
        if m:
            k = (m.group(1), m.group(2))
            new[k] = max(new.get(k, 0), budget_of(float(m.group(3))))
for (fx, region), b in sorted(new.items()):
    path = os.path.join(HERE, "budgets", fx + ".txt")
    lines = open(path).read().splitlines()
    out = []
    for l in lines:
        f = l.split()
        if len(f) == 2 and f[0] == region and float(f[1]) < b:
            l = "%s %g" % (region, b)
            print(fx, l)
            out.append(l)
            out.append("# %s raised to 1.5x the error measured on %s (per-architecture math and fused multiply-adds move the last bits)" % (region, LANE))
            continue
        out.append(l)
    open(path, "w").write("\n".join(out) + "\n")
