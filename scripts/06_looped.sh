#!/bin/bash
# Looped models (Sec. 5): Huginn-0125, Ouro-1.4B, Ouro-2.6B at pinned revisions. These need transformers 4.55
# (trust_remote_code), so they run in a separate environment: set LOOP_PY to its python (see requirements-loop.txt).
# L0 = exactness gates (exits nonzero on failure), L1a = transport across loops (read by paper/claims.py),
# L1b = Huginn variable depth. Output: $QLENS_OUT/loop_{huginn,ouro14,ouro26}. One GPU each.
source "$(dirname "$0")/common.sh"
export HF_HUB_OFFLINE=1
PY="${LOOP_PY:-python}"
$PY -c "import transformers; assert transformers.__version__.startswith('4.55'), transformers.__version__" || exit 1
for x in "tomg-group-umd/huginn-0125 bb6621b65e90b6a4b9b29ef88dc83866d450470c huginn" \
         "ByteDance/Ouro-1.4B 574fa66cb8bf5abdc979642d01cf2b79b16bfab1 ouro14" \
         "ByteDance/Ouro-2.6B 1ed04250da1a9936042725d302e81c8fa2ab5abd ouro26"; do
  set -- $x
  $PY src/loop_lens.py --model "$1" --revision "$2" --out "$OUT/loop_$3" --stages L0,L1a,L1b || exit 1
done
