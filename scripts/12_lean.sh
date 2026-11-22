#!/bin/bash
# Step 5 (Section 8; App. K Lean protocol, App. L results by layer): the Lean pilot, src/lean_qlens.py. Models: prover
# P = Goedel-LM/Goedel-Prover-V2-8B, base B = Qwen/Qwen3-8B-Base, chat C = Qwen/Qwen3-8B (E1 only). Corpora: RH = the
# research library (fetch it first: bash scripts/fetch_research_library.sh), ML = Mathlib at 0df444a (Apache-2.0;
# fetched here, the 600 sampled files only), FW = FineWeb-Edu docs 1000-1063 of the corpus of 00_data.sh.
# Output $QLENS_OUT/lean_v1/<model>/. One B200 per model command (each 1-2.5 h); the data stage is CPU.
# The defaults of src/lean_qlens.py are the protocol's final values (amendment 1: 504 contexts per Lean corpus;
# amendment 2: the gate2 stage).
source "$(dirname "$0")/common.sh"
D="$OUT/lean_v1/data"; L="$OUT/lean_v1"
[ -f "$D/riemann-lean.sample_files.txt" ] || { echo "run scripts/fetch_research_library.sh first"; exit 1; }
M="$D/mathlib"
if [ ! -d "$M/.git" ]; then
  git init -q "$M" && git -C "$M" remote add origin https://github.com/leanprover-community/mathlib4.git
fi
git -C "$M" -c protocol.version=2 fetch -q --depth 1 --filter=blob:none origin 0df444a360eaa60ab8c11dca51a86af692955474 \
  || exit 1
bash scripts/lean_sample.sh "$M" FETCH_HEAD || exit 1
git -C "$M" config core.sparseCheckout true
git -C "$M" sparse-checkout set --no-cone --stdin < "$M.sample_files.txt"
git -C "$M" -c advice.detachedHead=false checkout -q FETCH_HEAD || exit 1
python src/fetch.py Goedel-LM/Goedel-Prover-V2-8B Qwen/Qwen3-8B-Base Qwen/Qwen3-8B || exit 1
export HF_HUB_OFFLINE=1
python src/lean_qlens.py --stage data || exit 1          # -> lean_v1/data/contexts.pt (holds RH token ids: keep private)
for MODEL in Goedel-LM/Goedel-Prover-V2-8B Qwen/Qwen3-8B-Base; do
  TAG=$(echo "${MODEL#*/}" | tr 'A-Z.' 'a-z_')
  python src/lean_qlens.py --stage check,E1,E23 --model "$MODEL" --out "$L/$TAG" || exit 1
  python src/lean_qlens.py --stage gate2 --model "$MODEL" --out "$L/$TAG" || exit 1
  python src/lean_qlens.py --stage E4 --model "$MODEL" --out "$L/$TAG" || exit 1
done
python src/lean_qlens.py --stage check,E1 --model Qwen/Qwen3-8B --out "$L/qwen3-8b" || exit 1
