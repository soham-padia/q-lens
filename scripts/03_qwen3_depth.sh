#!/bin/bash
# Qwen3-8B / 14B / 32B (Sec. 4-5): exactness check, concept vectors, Taylor error, then the depth test D1 and
# B2x/B4c at 8 contexts. fp32 weights. Output: $QLENS_OUT/so_qwen3-{8b,14b,32b}[_ctx8]. One B200 (32B: fp32 needs ~130 GB).
source "$(dirname "$0")/common.sh"
for M in Qwen/Qwen3-8B Qwen/Qwen3-14B Qwen/Qwen3-32B; do
  tag=$(echo "${M#*/}" | tr 'A-Z.' 'a-z_'); D="$OUT/so_$tag"
  HF_HUB_OFFLINE=0 python src/fetch.py "$M" || exit 1
  export HF_HUB_OFFLINE=1
  python src/second_order.py --model "$M" --dtype float32 --out "$D" --stages check,vecs,B1,B2x,B4f,B4c || exit 1
  python src/second_order.py --model "$M" --dtype float32 --out "$D" --stages D1 || exit 1
  mkdir -p "${D}_ctx8" && ln -sfn "$(realpath "$D/concept_vecs.pt")" "${D}_ctx8/concept_vecs.pt"
  python src/second_order.py --model "$M" --dtype float32 --out "${D}_ctx8" --stages B2x,B4c --b2x-n-ctx 8 --b4c-n-ctx 8 || exit 1
  unset HF_HUB_OFFLINE
done
