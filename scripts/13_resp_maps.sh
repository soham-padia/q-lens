#!/bin/bash
# Step 5 (Section 8; App. M responsibility-map protocol, N tables, O named failures, P cost of the pair readouts):
# which lines of a proof act together, on Goedel-Prover-V2-8B. Needs lean_v1/data/contexts.pt (12_lean.sh, data stage).
#   src/responsibility_map_v2.py  per context, 16 contexts per corpus: exact pair removal (D2) against Q, path-averaged
#                                 Q and the third-order term, at blocks 11, 20, 29; ML, FW, their null controls, RH
#   src/fd_precision.py           finite-difference step-size check of Q on one Mathlib context, fp32 and fp64
#   src/cost_curve.py             cost of exact pair removal against the Q estimators at S = 12, 50, 200 lines
# Output $QLENS_OUT/resp_map_v2/ (RH: $QLENS_OUT/lean_v1/data/resp_map_v2/, which holds the library's text: keep it
# private). One B200 per command. Each script starts with a tiny CPU-sized check (--tiny).
source "$(dirname "$0")/common.sh"
export HF_HUB_OFFLINE=1
python src/responsibility_map_v2.py --tiny --corpora ML:1,MLnull:1 --layers 1,2 --out "$OUT/resp_v2_tiny" || exit 1
for c in ML FW MLnull FWnull; do
  python src/responsibility_map_v2.py --corpora "$c:16" || exit 1
done
(umask 077; python src/responsibility_map_v2.py --corpora RH:16 --out "$OUT/lean_v1/data/resp_map_v2") || exit 1
python src/fd_precision.py --tiny --layers 1,2 --out "$OUT/fd_tiny" || exit 1
python src/fd_precision.py --dtype float32 || exit 1
python src/fd_precision.py --dtype float64 || exit 1
python src/cost_curve.py --tiny || exit 1
python src/cost_curve.py || exit 1
