#!/bin/bash
# Does |Q|/|J| mark behaviourally fragile directions? Qwen3-8B, source blocks 11, 20, 29 (Sec. 10, limitations).
# The smoke run exits nonzero if a gate fails. Output: $QLENS_OUT/q_robust_qwen3-8b[_smoke]
source "$(dirname "$0")/common.sh"
export HF_HUB_OFFLINE=1
python src/q_robust.py --smoke || exit 1
for l in 11 20 29; do python src/q_robust.py --layers $l || exit 1; done
