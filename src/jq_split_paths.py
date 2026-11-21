"""Path of ||JU|| and ||Q|| with blocks traversed, not only its log-linear slope (v8 figure F2b; follow-up to
src/jq_split_depth.py). For each push block: the median over contexts x directions of ||JU|| and of ||Q(U,U)|| at each
read-out block, divided by its value one block after the push. Per run (random directions):
  J_min       smallest ||JU|| ratio over every (push, read-out) pair
  J_final     range over push blocks of the ||JU|| ratio at the last read-out block
  Q_final     range over push blocks of the ||Q|| ratio at the last read-out block
  r_final_min smallest ratio, over push blocks, of the ||Q|| ratio to the ||JU|| ratio at the last read-out block
  frac_J_half share of (push, read-out) pairs with r grown (k > 1) in which the fall of ||JU|| makes more than half of
              the growth of log r (log r - log r_1 = dlog||Q|| - dlog||JU||)
Usage: QLENS_OUT=<outputs dir> python src/jq_split_paths.py > results/jq_split_paths.jsonl
"""
import json
import os
from pathlib import Path
import numpy as np, torch

W = Path(os.environ.get("QLENS_OUT", "outputs"))
for d in ("so_qwen3-8b", "so_qwen3-14b", "so_qwen3-32b", "so_llama3_1-8b", "so_olmo2-1b_ckpt/stage1-step0-tokens0B",
          "so_olmo2-1b_ckpt/main"):
    raw = torch.load(W / d / "D1_raw.pt", map_location="cpu", weights_only=False)
    jmin, jfin, qfin, half, rfin = [], [], [], [], []
    for s in sorted(raw):
        tg = raw[s]["targets"]
        assert tg[0] - s == 1
        mJ = np.array([float(np.median(raw[s]["iso"]["norm_J"].numpy()[:, :, i])) for i in range(len(tg))])
        mQ = np.array([float(np.median(raw[s]["iso"]["norm_Q"].numpy()[:, :, i])) for i in range(len(tg))])
        dJ, dQ = np.log(mJ / mJ[0]), np.log(mQ / mQ[0])
        if len(tg) < 2:  # a push block one block before the end has no growth to report
            continue
        jmin.append(float((mJ / mJ[0]).min())); jfin.append(float(mJ[-1] / mJ[0])); qfin.append(float(mQ[-1] / mQ[0]))
        rfin.append(float((mQ[-1] / mQ[0]) / (mJ[-1] / mJ[0])))
        half += [bool(-dJ[i] > 0.5 * (dQ[i] - dJ[i])) for i in range(1, len(tg)) if dQ[i] - dJ[i] > 0]
    print(json.dumps({"dir": d, "J_min": round(min(jmin), 4), "J_final": [round(min(jfin), 4), round(max(jfin), 4)],
                      "Q_final": [round(min(qfin), 4), round(max(qfin), 4)], "r_final_min": round(min(rfin), 4), "frac_J_half": round(float(np.mean(half)), 4),
                      "n_pairs": len(half)}))
