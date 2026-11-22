#!/bin/bash
# Attribution patching against J+Q on Qwen3-8B (Sec. 6, App. H). The smoke run exits nonzero if a hard gate fails.
# Output: $QLENS_OUT/atp_q_qwen3-8b[_smoke]
source "$(dirname "$0")/common.sh"
export HF_HUB_OFFLINE=1
python src/atp_q.py --model Qwen/Qwen3-8B --smoke || exit 1
python src/atp_q.py --model Qwen/Qwen3-8B --tasks ioi --pairs 0:25 || exit 1
python src/atp_q.py --model Qwen/Qwen3-8B --tasks ioi --pairs 25:50 || exit 1
python src/atp_q.py --model Qwen/Qwen3-8B --tasks fact || exit 1
