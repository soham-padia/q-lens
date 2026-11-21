#!/bin/bash
# Token readout of the swap jets: which vocabulary tokens J and Q/2 of the logits raise and lower along the
# France->China S2 swap (capital, language, continent) at layers 24/40/56 of Qwen3.8-27B; exactness gates against
# finite differences and the stored S2 jets. fp32 weights, TF32 off, one B200 (about 1 minute).
# Needs fit_v1 (01_fit_jlens.sh) and layer_pick_v1 (02_qwen38_27b.sh). Output: $QLENS_OUT/swap_tokens_v1/swap_tokens.json
# (table: scripts/20_summaries.sh -> results/tables/swap_token_tables_full.md).
source "$(dirname "$0")/common.sh"
export HF_HUB_OFFLINE=1
python src/swap_tokens.py --model Qwen/Qwen3.8-27B --tf32 0 --out "$OUT/swap_tokens_v1" || exit 1
