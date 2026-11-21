#!/bin/bash
# First-order J-lens on Qwen3.8-27B: 96 prompts in 6 shards (one GPU each; shards are independent and can run in
# parallel). Used for the lens directions of the concept swaps (02, 08). Output: $QLENS_OUT/fit_v1/shardXXof06.pt
source "$(dirname "$0")/common.sh"
export HF_HUB_OFFLINE=1
for i in 0 1 2 3 4 5; do
  python src/fit_shard.py --shard $i --n-shards 6 --n-prompts 96 --dim-batch 8 --ckpt-every 4 || exit 1
done
