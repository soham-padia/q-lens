#!/bin/bash
# Step 5 (Section 8): specificity control for "J d + Q(d,d)/2 helps numeric intermediates" on the J-lens order-ops
# evaluation set, Qwen3.8-27B, 15 layers. src/qlens_numbers_ctrl.py imports src/qlens_app.py for the model, the fitted
# J-lens ($QLENS_OUT/fit_v1, from 01_fit_jlens.sh), the corpus means and the order-ops sweep. Needs
# $JLENS_DIR/data/evaluations. Output $QLENS_OUT/orderops_layers/ctrl_order-ops.json (and tensors_order-ops.pt).
# One B200.
source "$(dirname "$0")/common.sh"
export HF_HUB_OFFLINE=1
CTRL_SETS=order-ops python src/qlens_numbers_ctrl.py || exit 1
