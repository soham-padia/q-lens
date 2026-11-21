#!/bin/bash
# Token readout of the swap jets: which vocabulary tokens J and Q/2 of the logits raise and lower along the
# France->China S2 swap (capital, language, continent) at layers 24/40/56 of Qwen3.8-27B; exactness gates against
# finite differences and the stored S2 jets. fp32 weights, TF32 off, one B200 (about 1 minute).
# Needs fit_v1 (01_fit_jlens.sh) and layer_pick_v1 (02_qwen38_27b.sh). Output: $QLENS_OUT/swap_tokens_v1/swap_tokens.json
# (table: scripts/20_summaries.sh -> results/tables/swap_token_tables_full.md).
source "$(dirname "$0")/common.sh"
export HF_HUB_OFFLINE=1
python src/swap_tokens.py --model Qwen/Qwen3.8-27B --tf32 0 --out "$OUT/swap_tokens_v1" || exit 1

# The same readout over all 147 S2 two-hop swaps at the 11 S2 layers (1617 swap-layer cases), primary layers 24/40/56
# first. Full-vocabulary FD gate (h = 0.05) on every 15th swap plus the France->China trials (12 swaps, 132 cases); the
# S2 / r0 / z(0) gates on every case. One B200, about 25 minutes; resumable (a rerun skips cases already in the JSONL).
# Output: $QLENS_OUT/swap_tokens_v2/swap_tokens.jsonl (one line per case, 14 MB) and swap_tokens.json
# (rates: scripts/20_summaries.sh -> results/swap_tokens_rates.json, results/tables/swap_token_rates_tables.md).
# Exit code 3 means gate failures: the paper's run had 12, all Q vs FD at h = 0.05 on cases with small |Q| (FD round-off),
# re-checked with larger steps below.
python src/swap_tokens.py --model Qwen/Qwen3.8-27B --tf32 0 --out "$OUT/swap_tokens_v2" --trials all \
    --layers "24,40,56;16,20,28,32,36,44,48,52" --fd-every 15
echo "swap_tokens_v2 exit=$?"

# FD re-check of the 4 swaps with those failures, layers 16-32 (20 cases), FD gate on every case at h = 0.2 and 0.5.
# Output: $QLENS_OUT/swap_tokens_v2_fdcheck/h{0.2,0.5}/. The exit code is 3 here too: at the larger steps the control
# cases with large |Q| show the O(h^2) truncation error of the FD (J vs FD at h = 0.2; J and Q at 0.5), the other side
# of the trade-off; the 12 small-|Q| cases pass at h = 0.2.
for H in 0.2 0.5; do
  python src/swap_tokens.py --model Qwen/Qwen3.8-27B --tf32 0 --out "$OUT/swap_tokens_v2_fdcheck/h$H" \
      --trials language:Germany:Spain,continent:Japan:Canada,language:China:Spain,continent:Egypt:Japan \
      --layers 16,20,24,28,32 --h "$H"
  echo "swap_tokens_v2_fdcheck h=$H exit=$?"
done
