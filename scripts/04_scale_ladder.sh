#!/bin/bash
# Scale ladder (Qwen3.5 dense 0.8B-27B) and the 122B mixture-of-experts model (timing in Sec. 3; appendix setup).
# The 122B runs keep weights in bf16 and compute in fp32 (so_core.enable_upcast), with layers split over 2 GPUs and
# expert routing frozen for the finite-difference check (so_core.patch_routers). Output: $QLENS_OUT/so_qwen3_5-*.
source "$(dirname "$0")/common.sh"
for M in Qwen/Qwen3.5-0.8B Qwen/Qwen3.5-2B Qwen/Qwen3.5-4B Qwen/Qwen3.5-9B Qwen/Qwen3.5-27B; do
  tag=$(echo "${M#*/}" | tr 'A-Z.' 'a-z_')
  HF_HUB_OFFLINE=0 python src/fetch.py "$M" || exit 1
  HF_HUB_OFFLINE=1 python src/second_order.py --model "$M" --dtype float32 --out "$OUT/so_$tag" \
      --stages check,vecs,B1,B2x,B4f,B4c || exit 1
done
export HF_HUB_OFFLINE=1
# end-to-end test of the mixed bf16/fp32 multi-GPU MoE path on a tiny random model of the same architecture:
# mixed mode on 2 GPUs and plain fp32 on 1 GPU must agree (compare check.json / B1.json of the two dirs)
TINY="$OUT/tiny_qwen35moe"
[ -f "$TINY/config.json" ] || python src/tiny_moe.py "$TINY" || exit 1
A="--model $TINY --stages check,vecs,B1 --n-ctx 1 --n-vec-ctx 1 --b2x-n-ctx 1 --b2x-n-pos 2 --tf32 0"
python src/second_order.py $A --dtype float32 --weights-dtype bfloat16 --out "$OUT/test_tiny_mixed2gpu" || exit 1
CUDA_VISIBLE_DEVICES=0 python src/second_order.py $A --dtype float32 --out "$OUT/test_tiny_fp32" || exit 1
# 122B (2 x B200)
B122="--model Qwen/Qwen3.5-122B-A10B --dtype float32 --weights-dtype bfloat16 --layers 56,48,40,32,24,16 \
  --b2x-layers 56,48,40,32 --b4c-layers 56,40,24"
python src/second_order.py $B122 --out "$OUT/so_qwen3_5-122b-a10b" --stages check,vecs,B1,B2x || exit 1
python src/second_order.py $B122 --out "$OUT/so_qwen3_5-122b-a10b_b4c" --stages vecs,B4c || exit 1
