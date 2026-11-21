#!/bin/bash
# Exact split of Q along the actual two-hop swap pushes (Sec. 7.3, appendix on the swap split): 48 swaps, layers
# 48/40/36, fp32, TF32 off, in 6 independent shards. Needs fit_v1 (01) and second_order_v1/s2 (02).
# Output: $QLENS_OUT/swap_split_v1/swap_split_s{0..5}.json
source "$(dirname "$0")/common.sh"
export HF_HUB_OFFLINE=1
for i in 0 1 2 3 4 5; do
  python src/swap_split.py --model Qwen/Qwen3.8-27B --tf32 0 --n-trials 48 --shard $i/6 --name _s$i --max-min 105 \
      --out "$OUT/swap_split_v1" || exit 1
done
