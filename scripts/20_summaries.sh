#!/bin/bash
# CPU only, no model is loaded (compare_scale.py reads model configs from the HF cache). Reads $QLENS_OUT and
# writes results/ (claims.csv, qj_curves.csv, explorable_data/, a3_by_context.csv, d1_family_fits*.jsonl,
# jq_split_depth.jsonl, jq_split_paths.jsonl, swap_tokens_rates.json, fig_inputs/) and
# results/tables/*.md. Run after the GPU runners whose outputs you have; scripts skip missing models where they can.
source "$(dirname "$0")/common.sh"
T=results/tables; mkdir -p "$T"
python src/compare_depth.py
HF_HUB_OFFLINE=1 python src/compare_scale.py > /dev/null
HF_HUB_OFFLINE=1 python src/compare_scale.py _ctx8 > /dev/null   # 8-context B2x/B4c (binding test numbers)
python src/summarize_s2.py "$OUT"/second_order_v1/s2/*/S2.json --md "$T/s2_results.md" > /dev/null
# App. J reading numbers: the Reading section of this table (its one-/two-hop tables are from 3 smoke trials only)
python src/summarize_s2.py "$OUT"/second_order_v1/s2_smoke_rtx/S2.json --md "$T/s2_reading.md" > /dev/null
python src/layer_pick.py --tag "original 6-point grid" --md "$T/layer_pick_tables_grid6.md" > /dev/null
python src/layer_pick.py "$OUT"/layer_pick_v1/t{0,1,2,3,4,5}/S2.json --tag "dense grid" \
    --old "$OUT/second_order_v1/s2/t*/S2.json" --md "$T/layer_pick_tables_dense.md" > /dev/null
python src/swap_split_summary.py "$OUT"/swap_split_v1/swap_split_s*.json --md "$T/swap_split_tables.md" > /dev/null
python src/swap_tokens_table.py "$OUT/swap_tokens_v1/swap_tokens.json" --md "$T/swap_token_tables_full.md" > /dev/null
python src/swap_tokens_rates.py "$OUT/swap_tokens_v2/swap_tokens.jsonl" --v1 "$OUT/swap_tokens_v1/swap_tokens.json" \
    --md "$T/swap_token_rates_tables.md" --json-out results/swap_tokens_rates.json > /dev/null   # 147-swap rates
python src/summarize_atp_q.py > /dev/null
python src/flag_then_patch.py > /dev/null      # adds the Flag-then-patch section to atp_q_results.md
python src/summarize_q_robust.py > /dev/null
for m in 27b 8b; do python src/curv_circuits.py --model $m --md "$T/curvature_circuits_tables_$m.md" > /dev/null; done
OLMO="stage1-step0-tokens0B stage1-step300-tokens1B stage1-step10000-tokens21B stage1-step30000-tokens63B
      stage1-step100000-tokens210B stage1-step300000-tokens630B stage1-step1000000-tokens2098B
      stage1-step1907359-tokens4001B main"
D1=(so_qwen3-8b so_llama3_1-8b); [ -d "$OUT/so_gemma4-31b" ] && D1+=(so_gemma4-31b)
for r in $OLMO; do D1+=("so_olmo2-1b_ckpt/$r"); done
python src/fit_d1_family.py "${D1[@]}" > results/d1_family_fits.jsonl                # read by paper/claims.py
python src/fit_d1_family.py so_qwen3-8b so_qwen3-14b so_qwen3-32b second_order_v1 so_llama3_1-8b \
    so_olmo2-1b_ckpt/{stage1-step0-tokens0B,stage1-step10000-tokens21B,stage1-step1907359-tokens4001B,main} \
    > results/d1_family_fits_v2.jsonl                                              # exponential vs power-law fits
python src/jq_split_depth.py so_qwen3-8b so_qwen3-14b so_qwen3-32b so_llama3_1-8b \
    so_olmo2-1b_ckpt/main so_olmo2-1b_ckpt/stage1-step0-tokens0B > results/jq_split_depth.jsonl   # |JU| vs |Q| growth
python src/jq_split_paths.py > results/jq_split_paths.jsonl            # path of |JU| and |Q| (Table 4)
python paper/token_table.py "$OUT/swap_tokens_v1/swap_tokens.json" > "$T/token_table.tex"   # Table 2
python paper/fig_inputs.py > /dev/null                                # results/fig_inputs/ (F2b, F6)
python paper/a3_by_context.py > /dev/null
python paper/claims.py
