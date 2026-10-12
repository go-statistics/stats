#!/bin/bash
# Negative control: perturb one coefficient, run the accuracy tests, restore.
# usage: testdata/negative_control.sh FILE OLD NEW  (OLD must occur exactly once)
# Example (docs/accuracy.md): testdata/negative_control.sh accurate.go "1, -0.125, 0.0078125" "1, -0.1250001, 0.0078125"
cd "$(dirname "$0")/.."
f=$1; old=$2; new=$3
cp "$f" "$f.orig"
python3 - "$f" "$old" "$new" <<'PY'
import sys
p, old, new = sys.argv[1:]
s = open(p).read()
assert s.count(old) == 1, (old, s.count(old))
open(p, 'w').write(s.replace(old, new))
PY
echo "== perturbed $f: $old -> $new"
GOWORK=off go test -count=1 -run 'TestAccuracy$|TestParity$' . 2>&1 | grep -E "budget|^ok|^FAIL" | head -12
mv "$f.orig" "$f"
