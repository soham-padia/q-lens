"""Cross-family D1 fit: the same C2 slope (log median |Q|/|J| on blocks traversed k, OLS over every (source,target)
pair, paired context x direction bootstrap) and the C1 near-output gain as paper/claims.py, for any so_* dir.
Usage: python src/fit_d1_family.py so_qwen3-8b so_llama3_1-8b so_olmo2-1b_ckpt/main ...  (dirs under $QLENS_OUT, default ./outputs)
The bootstrap is re-seeded for each (dir, direction type), so a dir's intervals do not depend on the other dirs
in the call."""
import json, os, sys
from pathlib import Path
import numpy as np, torch

W = Path(os.environ.get("QLENS_OUT", "outputs"))
B = 1000


def boot(stat, G, C, D):
    v = stat(G)
    ix = [(rng.integers(0, C, C), rng.integers(0, D, D)) for _ in range(B)]
    bs = np.array([stat(G[ci][:, di]) for ci, di in ix])
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    return float(v), float(lo), float(hi)


def gate(d):
    p = W / d / "check.json"
    if not p.exists():
        p = W / d.split("/")[0] / "check.json"
    if not p.exists():
        return "no check.json"
    c = json.loads(p.read_text())
    best = max(min(x["cos"]) for x in c["fd_vs_jvp"].values())
    return f"fd-vs-jvp best min-cos={best:.4f} ({'PASS' if best > 0.99 else 'FAIL'})"


for d in sys.argv[1:]:
    js = json.loads((W / d / "D1.json").read_text())
    raw = torch.load(W / d / "D1_raw.pt", map_location="cpu", weights_only=False)
    nl, al, C, D = js["n_layers"], js["alphas"], js["n_ctx"], js["n_dir"]
    out = {"dir": d, "n_layers": nl, "gate": gate(d)}
    for tn in ("iso", "data"):
        rng = np.random.default_rng(0)  # per (dir, direction type): independent of the other dirs in the call
        ks, gs, ps = [], [], []
        gains = []
        for s in sorted(raw):
            R, tg = raw[s][tn], raw[s]["targets"]
            qj = (R["norm_Q"] / R["norm_J"]).numpy()
            for t_i, t in enumerate(tg):
                ks.append(t - s); gs.append(qj[:, :, t_i]); ps.append(s)
            fi = tg.index(nl - 1)
            a_i = al.index(0.1) if 0.1 in al else 0
            rl, rq = R["rel_lin"].numpy()[:, :, a_i, fi], R["rel_quad"].numpy()[:, :, a_i, fi]
            gains.append((s, nl - 1 - s, float(np.median(rl)), float(np.median(rl) / np.median(rq))))
        ks = np.array(ks, float); G = np.stack(gs, -1)
        y_of = lambda g: np.log(np.median(g.reshape(-1, g.shape[-1]), 0))
        sl = boot(lambda g: np.polyfit(ks, y_of(g), 1)[0], G, C, D)
        b, a0 = np.polyfit(ks, y_of(G), 1)
        rms = float(np.sqrt(np.mean((y_of(G) - (a0 + b * ks)) ** 2)))
        sv = np.array(ps); Dm = np.stack([(sv == u).astype(float) for u in sorted(set(ps))], 1)
        fe = lambda g: np.linalg.lstsq(np.c_[Dm, ks], y_of(g), rcond=None)[0][-1]
        wb = boot(fe, G, C, D)
        lk = np.log(ks)
        fp = lambda g: np.linalg.lstsq(np.c_[Dm, lk], y_of(g), rcond=None)[0][-1]
        pw = boot(fp, G, C, D)
        def rms_of(x):
            X = np.c_[Dm, x]; yy = y_of(G); b_ = np.linalg.lstsq(X, yy, rcond=None)[0]
            return float(np.sqrt(np.mean((yy - X @ b_) ** 2)))
        # sqrt(k) test with no free exponent: log r = per-push-block intercept + 0.5 log k
        yy = y_of(G); res = yy - 0.5 * lk; icp = np.linalg.lstsq(Dm, res, rcond=None)[0]
        rms_half = float(np.sqrt(np.mean((res - Dm @ icp) ** 2)))
        out[tn] = dict(slope=sl, slope_per_frac_depth=[x * nl for x in sl], within_push_block=wb, power_exponent_within=pw, rms_exp_within=rms_of(ks), rms_pow_within=rms_of(lk), rms_sqrtk_fixed=rms_half, k_range=[float(ks.min()), float(ks.max())],
                       qj_k1=float(np.exp(a0 + b)), rms_resid=rms, n_pairs=len(ks),
                       gain_final_alpha0p1=[dict(src=s, k=k, relerr_J=round(e, 4), gain=round(g, 2)) for s, k, e, g in gains])
    print(json.dumps(out))
