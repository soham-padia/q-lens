#!/bin/bash
# Other families (Sec. 5): Llama-3.1-8B-Instruct and nine OLMo-2-1B training checkpoints (check, vecs, D1, fp32).
# Gemma-4-31B fails the finite-difference gate and is not used for any number; it is optional (GEMMA=1).
# Output: $QLENS_OUT/so_llama3_1-8b, $QLENS_OUT/so_olmo2-1b_ckpt/<revision>.
source "$(dirname "$0")/common.sh"
run() {  # model, out dir, [revision]
  local spec="$1" ra=""; [ -n "$3" ] && { spec="$1@$3"; ra="--revision $3"; }
  [ -s "$2/D1.json" ] && { echo "skip $spec (D1.json exists)"; return 0; }
  HF_HUB_OFFLINE=0 python src/fetch.py "$spec" || return 1
  mkdir -p "$2"
  HF_HUB_OFFLINE=1 python src/second_order.py --model "$1" $ra --dtype float32 --out "$2" --stages check,vecs,D1
}
run meta-llama/Llama-3.1-8B-Instruct "$OUT/so_llama3_1-8b" || exit 1
for R in stage1-step0-tokens0B stage1-step300-tokens1B stage1-step10000-tokens21B stage1-step30000-tokens63B \
         stage1-step100000-tokens210B stage1-step300000-tokens630B stage1-step1000000-tokens2098B \
         stage1-step1907359-tokens4001B main; do
  run allenai/OLMo-2-0425-1B "$OUT/so_olmo2-1b_ckpt/$R" "$R" || exit 1
done
[ "${GEMMA:-0}" = 1 ] && { run google/gemma-4-31B-it "$OUT/so_gemma4-31b" || exit 1; }
exit 0
