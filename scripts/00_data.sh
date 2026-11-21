#!/bin/bash
# Prompt corpus (FineWeb-Edu sample-10BT, 2000 docs, seed 0) and model weights. Needs internet, no GPU.
# Fetch only the models for the experiments you want to run (the 122B model is about 240 GB).
source "$(dirname "$0")/common.sh"
python src/build_corpus.py || exit 1
python src/fetch.py Qwen/Qwen3.8-27B Qwen/Qwen3-8B Qwen/Qwen3-14B Qwen/Qwen3-32B || exit 1
# other families and the scale ladder (scripts 03 and 04 also fetch on demand):
#   python src/fetch.py meta-llama/Llama-3.1-8B-Instruct Qwen/Qwen3.5-0.8B Qwen/Qwen3.5-2B Qwen/Qwen3.5-4B \
#       Qwen/Qwen3.5-9B Qwen/Qwen3.5-27B Qwen/Qwen3.5-122B-A10B
# looped models (pinned revisions):
#   python src/fetch.py tomg-group-umd/huginn-0125@bb6621b65e90b6a4b9b29ef88dc83866d450470c \
#       ByteDance/Ouro-1.4B@574fa66cb8bf5abdc979642d01cf2b79b16bfab1 ByteDance/Ouro-2.6B@1ed04250da1a9936042725d302e81c8fa2ab5abd
