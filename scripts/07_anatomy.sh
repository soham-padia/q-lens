#!/bin/bash
# Exact split of Q over blocks, sublayers, heads and neurons (A2) and third order (A3) (Sec. 7, appendix on third
# order). TF32 settings as in the paper's setup table: off everywhere except Qwen3-8B third order.
# Output: $QLENS_OUT/qanat_qwen3-8b, $QLENS_OUT/qanat_q38-27b. One B200 per command.
source "$(dirname "$0")/common.sh"
export HF_HUB_OFFLINE=1
Q8="--model Qwen/Qwen3-8B --out $OUT/qanat_qwen3-8b"
python src/q_anatomy.py $Q8 --stages check || exit 1                      # exactness gates
python src/q_anatomy.py $Q8 --stages A2 --name _32 --tf32 0 || exit 1
python src/q_anatomy.py $Q8 --stages A3 --tf32 1 || exit 1
Q27="--model Qwen/Qwen3.8-27B --out $OUT/qanat_q38-27b"
python src/q_anatomy.py $Q27 --stages check || exit 1
python src/q_anatomy.py $Q27 --stages A2 --sources 8,16 --name _a32 --tf32 0 || exit 1
python src/q_anatomy.py $Q27 --stages A2 --sources 24,32,40,48,56 --name _b32 --tf32 0 || exit 1
for s in 8 24 40; do
  python src/q_anatomy.py $Q27 --stages A3 --a3-sources $s --a3-n-ctx 4 --name _s${s}_32 --tf32 0 || exit 1
done
