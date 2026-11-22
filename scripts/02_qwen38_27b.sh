#!/bin/bash
# Qwen3.8-27B (the default model of src/second_order.py), output $QLENS_OUT/second_order_v1. One B200 per command.
# Sections: exactness check and Taylor error (Sec. 3-4), depth D1 (Sec. 4-5), Q as a readout lens (B3x, Sec. 6),
# concept swaps S1/S2 and the fine dose grid (Sec. 6). Commands are those of the paper's runs; S2 shards are
# independent (the paper ran them in parallel).
source "$(dirname "$0")/common.sh"
export HF_HUB_OFFLINE=1
SO="$OUT/second_order_v1"
python src/second_order.py --stages check,vecs,B1,B2 || exit 1          # writes concept_vecs.pt, used below
python src/second_order.py --stages B2x || exit 1
python src/second_order.py --stages B4f,B4c --b4c-layers 24,40,56 || exit 1
# depth test (D1), and B2x/B4c at 8 contexts into a separate dir
python src/second_order.py --dtype float32 --out "$SO" --stages D1 || exit 1
mkdir -p "${SO}_ctx8" && ln -sfn "$(realpath "$SO/concept_vecs.pt")" "${SO}_ctx8/concept_vecs.pt"
python src/second_order.py --dtype float32 --out "${SO}_ctx8" --stages B2x,B4c --b2x-n-ctx 8 --b4c-n-ctx 8 || exit 1
# Q as a readout lens on the J-lens evaluation sets (needs $JLENS_DIR/data/evaluations). Reconstructed command:
# the run ledger records the sets and --b3-exact-pos 3; the output names and layer list are read off the outputs.
for x in "a multihop,order-ops" "b multilingual,typo" "c poetry,association"; do
  set -- $x
  python src/second_order.py --stages B3 --b3-sets "$2" --b3-exact-pos 3 --b3-name "B3x_$1" \
      --b3-layers 24,32,40,44,48,52,56 || exit 1
done
# B3x_a again after the space-token fix (a number is matched by its first non-space token), with the reading part
# R of S2: the paper's multi-hop and order-ops lens numbers (Sec. 6) and the App. Q reading numbers are from this run
python src/second_order.py --stages B3,S2 --weights-dtype bfloat16 --dtype float32 --b3-sets multihop,order-ops \
    --b3-exact-pos 3 --b3-layers 24,32,40,44,48,52,56 --b3-name B3x_a_wsfix --s2-parts R --tf32 0 \
    --out "$SO/wsfix" || exit 1
# concept swaps: S1 one-hop (needs the J-lens of 01_fit_jlens.sh), S2 two-hop in 8 shards (bf16-stored weights,
# fp32 compute, TF32 off)
python src/second_order.py --stages S1 || exit 1
MIX="--tf32 0 --weights-dtype bfloat16 --dtype float32"
# reading part R (J-lens rank of the hidden intermediate, all 141 items) was first run in this short run, which
# also ran 3 one-hop and 3 two-hop trials (the paper reads part R from $SO/wsfix, above); the full shards below ran
# only O and T
python src/second_order.py --stages S2 $MIX --s2-limit 3 --out "$SO/s2_smoke_rtx" || exit 1
for i in 0 1; do
  python src/second_order.py --stages S2 $MIX --s2-parts O --s2-shard $i/2 --s2-max-min 25 --out "$SO/s2/o$i" || exit 1
done
for i in 0 1 2 3 4 5; do
  python src/second_order.py --stages S2 $MIX --s2-parts T --s2-shard $i/6 --s2-max-min 44 --out "$SO/s2/t$i" || exit 1
done
# fine dose grid for the same two-hop swaps (26 single-layer and 20 band strengths), 6 shards
LAMS=0.1,0.25,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.8,2.0,2.5,3.0
BLAMS=0.25,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0,1.1,1.2,1.4,1.6,2.0
for i in 0 1 2 3 4 5; do
  python src/second_order.py --stages S2 $MIX --s2-parts T --s2-shard $i/6 --s2-max-min 100 --s2-lams $LAMS \
      --s2-band-lams $BLAMS --out "$OUT/layer_pick_v1/t$i" || exit 1
done
