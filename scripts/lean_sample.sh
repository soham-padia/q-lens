#!/bin/bash
# Seeded file sample of a Lean repository (the paper's Lean protocol): list the .lean files at <ref> (no lakefile, nothing
# under .lake/; for Mathlib only Mathlib/), in byte order, shuffle with seed 20261121 and keep the first n (default 600).
# Writes <dir>.commit, <dir>.all_files.txt and <dir>.sample_files.txt next to <dir>; src/lean_qlens.py --stage data
# reads the sample and the commit. Called by fetch_research_library.sh and 12_lean.sh.
# Usage: bash scripts/lean_sample.sh <repo dir> <ref> [n]
dir=${1%/}; ref=${2:-HEAD}; n=${3:-600}; seed=${SEED:-20261121}
[ -d "$dir/.git" ] || { echo "not a git checkout: $dir"; exit 1; }
git -C "$dir" rev-parse "$ref" > "$dir.commit" || exit 1
git -C "$dir" ls-tree -r --name-only "$ref" | grep '\.lean$' | grep -v -e lakefile -e '/\.lake/' | LC_ALL=C sort \
    > "$dir.all_files.txt"
if [ "$(basename "$dir")" = mathlib ]; then
  grep '^Mathlib/' "$dir.all_files.txt" > "$dir.tmp" && mv "$dir.tmp" "$dir.all_files.txt"
fi
python3 -I -c "
import random, sys
fs = open(sys.argv[1]).read().split()
random.Random(int(sys.argv[3])).shuffle(fs)
open(sys.argv[2], 'w').write('\n'.join(sorted(fs[:int(sys.argv[4])])) + '\n')
" "$dir.all_files.txt" "$dir.sample_files.txt" "$seed" "$n" || exit 1
echo "$(basename "$dir"): commit $(cat "$dir.commit"), $(wc -l < "$dir.all_files.txt") .lean files, sampled $(wc -l < "$dir.sample_files.txt")"
