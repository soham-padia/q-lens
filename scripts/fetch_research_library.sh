#!/bin/bash
# Step 5 research library (see the README, "Step 5"). It has no licence, so this repository does not redistribute it or
# quote any of its lines: you fetch it yourself from GitHub, at the commit the paper used:
#
#   git clone https://github.com/tamnd/riemann-lean <dir>
#   git -C <dir> checkout af705e0d9a5904326eb0693310e36d9484df7f8f
#
# This script does exactly that into $QLENS_OUT/lean_v1/data/riemann-lean (default ./outputs/..., git-ignored; the clone
# is about 330 MB) and writes the seeded 600-file sample that src/lean_qlens.py reads (scripts/lean_sample.sh).
# Do not publish $QLENS_OUT/lean_v1/data/: contexts.pt and resp_map_v2/ there hold the library's token ids and text.
# Then run scripts/12_lean.sh.
source "$(dirname "$0")/common.sh"
D="$OUT/lean_v1/data"; R="$D/riemann-lean"
mkdir -p "$D"
[ -d "$R/.git" ] || git clone https://github.com/tamnd/riemann-lean "$R" || exit 1
git -C "$R" -c advice.detachedHead=false checkout af705e0d9a5904326eb0693310e36d9484df7f8f || exit 1
bash scripts/lean_sample.sh "$R" HEAD
