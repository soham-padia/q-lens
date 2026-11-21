"""Does |Q|/|J| grow with blocks traversed because |Q| grows or because |JU| shrinks?

Reads D1_raw.pt (norm_J = ||J U||, norm_Q = ||Q(U,U)|| at each readout block; U = rho_l * unit direction at the push
block). Within-push-block fits (push-block fixed effects) of log median ||JU||, log median ||Q||, and log median r on k
(blocks traversed) and on log k. Paired context x direction bootstrap, B=1000, as in fit_d1_family.py.
Usage: python src/jq_split_depth.py so_qwen3-8b so_llama3_1-8b ...  (dirs under $QLENS_OUT, default ./outputs)
"""
import os, sys, json
from pathlib import Path
import numpy as np, torch

W = Path(os.environ.get("QLENS_OUT", "outputs"))
rng = np.random.default_rng(0)

def fe_slope(ks, ps, y, x):
    Dm = np.stack([(ps == u).astype(float) for u in np.unique(ps)], 1)
    return np.linalg.lstsq(np.c_[Dm, x], y, rcond=None)[0][-1]

for d in sys.argv[1:]:
    raw = torch.load(W / d / "D1_raw.pt", map_location="cpu", weights_only=False)
    out = {"dir": d}
    for tn in ("iso", "data"):
        ks, ps, NJ, NQ = [], [], [], []
        for s in sorted(raw):
            R, tg = raw[s][tn], raw[s]["targets"]
            for t_i, t in enumerate(tg):
                ks.append(t - s); ps.append(s)
                NJ.append(R["norm_J"].numpy()[:, :, t_i]); NQ.append(R["norm_Q"].numpy()[:, :, t_i])
        ks, ps = np.array(ks, float), np.array(ps)
        NJ, NQ = np.stack(NJ, -1), np.stack(NQ, -1)  # [C, D, pairs]
        C, D = NJ.shape[:2]
        med = lambda A: np.log(np.median(A.reshape(-1, A.shape[-1]), 0))
        stats = {"J": lambda a, b: med(a), "Q": lambda a, b: med(b), "r": lambda a, b: med(b / a)}
        res = {}
        for nm, f in stats.items():
            for form, x in (("per_block", ks), ("power", np.log(ks))):
                pt = fe_slope(ks, ps, f(NJ, NQ), x)
                bs = []
                for _ in range(1000):
                    ci = rng.integers(0, C, C); di = rng.integers(0, D, D)
                    a, b = NJ[ci][:, di], NQ[ci][:, di]
                    bs.append(fe_slope(ks, ps, f(a, b), x))
                lo, hi = np.percentile(bs, [2.5, 97.5])
                res[f"{nm}_{form}"] = [round(float(pt), 4), round(float(lo), 4), round(float(hi), 4)]
        out[tn] = res
    print(json.dumps(out))
