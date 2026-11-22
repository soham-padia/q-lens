#!/usr/bin/env python3
"""Interactive J / Q / J+Q lens on Qwen3.8-27B (Gradio demo on one GPU).
src/qlens_numbers_ctrl.py imports its model, fitted J-lens, mean states and order-ops sweep machinery.

Conventions follow src/so_core.py and the paper (B3x / qdv_lens):
  h_l        output of block l;  L = last block (pre final norm);  readout of a d-vector y: W_U (gamma * y)
  fitted J   the J-lens J_l = E[dh_L/dh_l] from fit_v1 (96 prompts, 6 shards), read like the J-lens repo
  mu_l       mean of h_l over the valid positions of 8 corpus contexts (fineweb-edu docs 1000+, 128 tokens)
  d          h_l,t - mu_l, the state's deviation from the mean (scaled by the dose lambda)
  Q(a, b)    exact second derivative by nested forward-mode AD; the Taylor term is 1/2 Q(d, d)

Tabs
  1 Prompt    fitted J-lens grid (layers x positions, top-1 token, hover for top-5) + tracked-token ranks
  2 Q lens    averaged, like the J-lens: per layer, d is placed at sampled positions of corpus contexts and pushed
              to the final block (F = sum over valid targets); J d, 1/2 Q(d,d), J d + 1/2 Q and the measured average
              change F(h_c + d) - F(h_c) ("truth"), next to the fitted J-lens of h and of d
  3 One cell  exact, inside this prompt: push u at (l, t) from a base point; J u, 1/2 Q(u,u), J+Q and truth, as
              promoted tokens and as next-token predictions, with a dose curve

  python src/qlens_app.py            (on a GPU node; PORT env picks the port; --selftest runs every path first)
"""
from __future__ import annotations

import argparse
import functools
import threading
import glob
import html
import json
import math
import os
import sys
import time
import traceback

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
import transformers  # noqa: E402
from torch.func import jvp  # noqa: E402

JLENS_DIR = os.environ.get("JLENS_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "jacobian-lens"))
sys.path.insert(0, JLENS_DIR)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jlens  # noqa: E402
from jlens.fitting import valid_position_mask  # noqa: E402
from jlens.hooks import ActivationRecorder  # noqa: E402
import so_core  # noqa: E402

W = os.environ.get("QLENS_OUT", "outputs")
MODEL = os.environ.get("QL_MODEL", "Qwen/Qwen3.8-27B")
LENS_DIR = os.environ.get("QL_LENS", f"{W}/fit_v1")
CORPUS = f"{W}/corpus/fineweb_edu_2000.jsonl"
PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]  # dataviz reference categorical, fixed order
STY = ["-", "--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1))]
MRK = ["o", "s", "^", "D", "v", "P"]
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# ---------------------------------------------------------------------------------------------------- model
tok = transformers.AutoTokenizer.from_pretrained(MODEL)
_arch = (transformers.AutoConfig.from_pretrained(MODEL).architectures or [""])[0]
_cls = transformers.AutoModelForImageTextToText if "ConditionalGeneration" in _arch else transformers.AutoModelForCausalLM
BIG = torch.cuda.get_device_properties(0).total_memory > 150e9
if BIG:  # B200: fp32 weights
    hf = _cls.from_pretrained(MODEL, dtype=torch.float32, attn_implementation="sdpa", device_map="cuda")
else:  # RTX PRO 6000 (96 GB): bf16 weights upcast per call (so_core mixed mode; bf16 -> fp32 is exact, same model)
    hf = _cls.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation="sdpa", device_map="cuda")
    so_core.enable_upcast(hf, torch.float32)
hf.eval().requires_grad_(False)
m = jlens.from_hf(hf, tok)
NL = m.n_layers
L = NL - 1
D = m.d_model
DEV = "cuda"
W_U = hf.get_output_embeddings().weight.detach().float()
with torch.no_grad():
    GAMMA = m._final_norm(torch.ones(1, D, device=DEV)).float()[0]
log(f"loaded {MODEL} ({_arch}) {'fp32' if BIG else 'bf16->fp32 mixed'} n_layers={NL} d={D} vocab={W_U.shape[0]} mem={torch.cuda.memory_allocated() / 1e9:.1f} GB")

# ---------------------------------------------------------------------------------------------------- fitted J-lens
JL, NP = {}, 0
for fp in sorted(glob.glob(f"{LENS_DIR}/shard*of*.pt")):
    ck = torch.load(fp, map_location="cpu", weights_only=True, mmap=True)
    n = ck["n_prompts"]; NP += n
    for l, J in ck["J"].items():
        x = J.to(DEV).float() * n
        JL[int(l)] = x if int(l) not in JL else JL[int(l)].add_(x)
    del ck
for l in JL:
    JL[l] /= NP
    if not BIG:
        JL[l] = JL[l].half()  # stored fp16 (as fitted), applied in fp32


def Jm(l):
    return JL[l].float()


log(f"J-lens: {len(JL)} layers, n_prompts={NP}, mem={torch.cuda.memory_allocated() / 1e9:.1f} GB")

# ---------------------------------------------------------------------------------------------------- corpus, mu_l
_docs = [json.loads(s)["text"] for s in open(CORPUS)]
CTX = [m.encode(t, max_length=128) for t in _docs[1000:1064]]
CTX = [c for c in CTX if c.shape[1] == 128]
VALID = valid_position_mask(128).nonzero(as_tuple=True)[0].to(DEV)
MU = torch.zeros(NL, D, device=DEV)
RHO = torch.zeros(NL)
with torch.no_grad(), ActivationRecorder(m.layers, at=list(range(NL))) as rec:
    _n = 0
    for ids in CTX[:8]:
        m.forward(ids)
        for l in range(NL):
            H = rec.activations[l][0, VALID].float()
            MU[l] += H.sum(0); RHO[l] += H.norm(dim=-1).sum().cpu()
        _n += len(VALID)
MU /= _n; RHO /= _n
log(f"mu_l from 8 contexts x {len(VALID)} positions")

# ---------------------------------------------------------------------------------------------------- helpers
def tstr(i):
    s = tok.decode([int(i)])
    return s.replace(" ", "␣").replace("\n", "↵").replace("\t", "⇥") or "∅"


def logits_of(y):
    return (y.float() * GAMMA) @ W_U.T


def next_probs(hL):  # pre-norm final state [d] -> next-token probabilities
    return torch.softmax(m._final_norm(hL[None].float())[0].float() @ W_U.T, -1)


def topk(lg, k):
    v, i = lg.topk(k)
    return [(tstr(j), float(x)) for j, x in zip(i.tolist(), v.tolist())]


def track_ids(text):
    """'9, 72, multiplication' -> [(label, [first-token ids of variants])]."""
    out = []
    for w in [s.strip() for s in text.split(",") if s.strip()][:6]:
        ids = set()
        for v in {w, w.lower(), w.capitalize()}:
            for p in ("", " "):
                e = [i for i in tok.encode(p + v, add_special_tokens=False) if tok.decode([i]).strip()]  # skip bare-space first token: ' 9' -> ' ', '9'
                if e:
                    ids.add(e[0])
        out.append((w, sorted(ids)))
    return out


def rank(lg, ids):  # 1-based best rank among ids
    sub = lg[ids]
    return int((lg[None, :] > sub[:, None]).sum(-1).min()) + 1


def esc(s):
    return html.escape(s, quote=True)


def run_F(ids, l, delta, target):
    """Run with h_l += delta ([B, T, d]); return the block-L output at `target`: 'last' -> [B, d], or an index
    tensor of positions -> sum over them [B, d]."""
    box = {}

    def add(_m, _i, out):
        if torch.is_tensor(out):
            return out + delta
        return (out[0] + delta,) + tuple(out[1:])

    def recf(_m, _i, out):
        box["h"] = out if torch.is_tensor(out) else out[0]

    B = delta.shape[0]
    h1 = m.layers[l].register_forward_hook(add)
    h2 = m.layers[L].register_forward_hook(recf)
    try:
        m.forward(ids.expand(B, -1))
    finally:
        h1.remove(); h2.remove()
    hL = box["h"]
    return hL[:, -1] if isinstance(target, str) else hL[:, target].sum(1)


def nested(f, X0, A):
    """(J(X0) A, Q(A, A)) by one nested forward-mode call."""
    with so_core.math_sdpa():
        return jvp(lambda x: jvp(f, (x,), (A,))[1], (X0,), (A,))


def relerr(a, b):
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


def cos(a, b):
    return float(torch.nn.functional.cosine_similarity(a.flatten()[None], b.flatten()[None])[0])


# ---------------------------------------------------------------------------------------------------- state
CUR = {}
GPU = threading.RLock()  # one model, hooks are global: every forward/jvp holds this (the sweep thread and the UI)


def locked(fn):
    @functools.wraps(fn)
    def w(*a, **k):
        with GPU:
            return fn(*a, **k)
    return w
CSS = """
.ql table{border-collapse:collapse;font-family:ui-monospace,Menlo,monospace;font-size:12px}
.ql td,.ql th{border:1px solid rgba(128,128,128,.25);padding:2px 5px;white-space:nowrap;text-align:left}
.ql th{position:sticky;top:0;background:var(--block-background-fill,#fff);z-index:1}
.ql .sm{opacity:.7;font-size:11px}
.ql .key span{display:inline-block;padding:1px 6px;margin-right:6px;border-radius:3px}
"""


def tint(i, a=0.28):
    h = PAL[i % len(PAL)].lstrip("#")
    r, g, b = (int(h[k:k + 2], 16) for k in (0, 2, 4))
    return f"rgba({r},{g},{b},{a})"


def key_html(tracks):
    if not tracks:
        return ""
    s = "".join(f'<span style="background:{tint(i)}">{esc(w)} → {" ".join(esc(tstr(j)) for j in ids)}</span>'
                for i, (w, ids) in enumerate(tracks))
    return f'<div class="key">tracked (first token of each variant): {s}</div>'


def pos_choices():
    return [f"{t}: {tstr(i)}" for t, i in enumerate(CUR["ids"][0].tolist())]


# ---------------------------------------------------------------------------------------------------- tab 1
@locked
@torch.no_grad()
def run_prompt(prompt, track_text, window):
    plt.close("all")
    ids = m.encode(prompt, max_length=256)
    T = ids.shape[1]
    with ActivationRecorder(m.layers, at=list(range(NL))) as rec:
        m.forward(ids)
    H = torch.stack([rec.activations[l][0].float() for l in range(NL)])  # [NL, T, d]
    tracks = track_ids(track_text)
    CUR.update(ids=ids, H=H, T=T, prompt=prompt)
    p = next_probs(H[L, -1])
    nxt = ", ".join(f"`{esc(s)}` {v:.2f}" for s, v in topk(p, 5))
    w = max(1, min(int(window), T))
    cols = list(range(T - w, T))
    rows = []
    for l in list(range(NL)):
        Y = H[l, cols] @ Jm(l).T if l in JL else H[l, cols]
        lg = logits_of(Y)  # [w, V]
        cells = []
        for k, t in enumerate(cols):
            top = topk(lg[k], 5)
            rk = [rank(lg[k], ids_) for _, ids_ in tracks]
            hit = next((i for i, r in enumerate(rk) if r == 1), None)
            bg = f' style="background:{tint(hit)}"' if hit is not None else ""
            tip = f"L{l} pos {t}: " + " | ".join(f"{s}" for s, _ in top) + \
                  ("  —  ranks: " + ", ".join(f"{w_}={r}" for (w_, _), r in zip(tracks, rk)) if tracks else "")
            cells.append(f'<td{bg} title="{esc(tip)}">{esc(top[0][0][:10])}</td>')
        lab = f"L{l}" + (" (output)" if l == L else "")
        rows.append(f"<tr><th>{lab}</th>{''.join(cells)}</tr>")
    head = "<tr><th>layer \\ pos</th>" + "".join(f"<th>{t}<br>{esc(tstr(ids[0, t]))}</th>" for t in cols) + "</tr>"
    grid = (f'<div class="ql">{key_html(tracks)}<div style="max-height:720px;overflow:auto">'
            f"<table>{head}{''.join(rows)}</table></div>"
            '<div class="sm">Fitted J-lens top-1 per cell (hover for top-5 and tracked ranks). Last row: the '
            "model's own output. A cell is tinted when a tracked token is top-1.</div></div>")
    CUR["tracks"] = tracks
    fig = jlens_rank_fig(T - 1, tracks)
    ch = pos_choices()
    info = f"**{T} tokens.** Next token: {nxt}"
    import gradio as gr
    return info, grid, fig, gr.Dropdown(choices=ch, value=ch[-1]), gr.Dropdown(choices=ch, value=ch[-1])


@torch.no_grad()
def jlens_rank_fig(t, tracks):
    H = CUR["H"]
    fig, ax = plt.subplots(figsize=(8, 3.4), dpi=110)
    if not tracks:
        ax.text(0.5, 0.5, "add tracked tokens to see ranks", ha="center", transform=ax.transAxes)
        return fig
    ls = sorted(JL)
    lg = logits_of(torch.stack([H[l, t] @ Jm(l).T for l in ls]))
    for i, (w, ids_) in enumerate(tracks):
        r = [rank(lg[k], ids_) for k in range(len(ls))]
        ax.plot(ls, r, color=PAL[i], ls=STY[i], marker=MRK[i], ms=3, lw=2, label=w)
    style_rank_ax(ax, f"fitted J-lens rank at pos {t} ({tstr(CUR['ids'][0, t])})")
    fig.tight_layout()
    return fig


def style_rank_ax(ax, title):
    ax.set_yscale("log"); ax.invert_yaxis()
    ax.set_xlabel("layer"); ax.set_ylabel("rank (1 = top)")
    ax.set_title(title, fontsize=10, loc="left")
    ax.grid(alpha=.25, lw=.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=8, frameon=False, loc="best")


def parse_pos(s):
    return int(str(s).split(":")[0])


def parse_layers(s):
    out = []
    for part in str(s).replace(" ", "").split(","):
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")[:2]
            step = 1
            if ":" in b:
                b, step = b.split(":")
            out += list(range(int(a), int(b) + 1, int(step)))
        else:
            out.append(int(part))
    return [l for l in dict.fromkeys(out) if 0 <= l < L]


# ---------------------------------------------------------------------------------------------------- tab 2
def qbar(l, d, n_ctx, n_pos):
    """Averaged over corpus contexts: d placed at n_pos sampled valid positions (one per batch row), F = sum over
    valid targets. Returns (J d, Q(d,d), truth = F(h_c + d@tp) - F(h_c)), each [d] averaged over rows and contexts."""
    acc = [torch.zeros(D, device=DEV) for _ in range(3)]
    for c in range(n_ctx):
        ids = CTX[8 + c]  # contexts disjoint from the 8 used for mu
        g = torch.Generator().manual_seed(1234 + l + 7919 * c)
        tp = VALID[torch.randperm(len(VALID), generator=g)[:n_pos].to(DEV)]
        A = torch.zeros(n_pos, 128, D, device=DEV)
        A[torch.arange(n_pos, device=DEV), tp] = d
        f = lambda dl: run_F(ids, l, dl, VALID)  # noqa: E731
        Jd, Q = nested(f, torch.zeros_like(A), A)
        with torch.no_grad(), so_core.math_sdpa():
            tr = f(A) - f(torch.zeros_like(A))
        for a, x in zip(acc, (Jd, Q, tr)):
            a += x.mean(0)
    return [a / n_ctx for a in acc]


ARMS = ["J-lens of h (fitted)", "J̄ d (fitted)", "J d (exact)", "½Q(d,d)", "J d + ½Q", "truth (avg)"]


def lens_column(pos, layers_txt, lam, n_ctx, n_pos, track_text):
    if "H" not in CUR:
        yield "Run a prompt first.", None
        return
    plt.close("all")
    t = parse_pos(pos); layers = parse_layers(layers_txt); tracks = track_ids(track_text)
    n_ctx, n_pos, lam = int(n_ctx), int(n_pos), float(lam)
    H = CUR["H"]
    res = []
    t0 = time.time()
    for l in layers:
        h = H[l, t]
        d = (h - MU[l]) * lam
        with GPU:
            Jd, Q, tr = qbar(l, d, n_ctx, n_pos)
        with torch.no_grad():
            ys = [h @ Jm(l).T, d @ Jm(l).T, Jd, 0.5 * Q, Jd + 0.5 * Q, tr]
            lgs = [logits_of(y) for y in ys]
            res.append(dict(l=l, top=[topk(lg, 3) for lg in lgs],
                            rk=[[rank(lg, ids_) for _, ids_ in tracks] for lg in lgs],
                            r=float((0.5 * Q).norm() / Jd.norm()), eJ=relerr(Jd, tr), eJQ=relerr(Jd + 0.5 * Q, tr),
                            cJ=cos(Jd, tr), cJQ=cos(Jd + 0.5 * Q, tr), dn=float(d.norm() / h.norm())))
        yield column_html(res, tracks, t, lam, time.time() - t0), column_fig(res, tracks, t)


def column_html(res, tracks, t, lam, el):
    head = "<tr><th>layer</th><th>|d|/|h|</th>" + "".join(f"<th>{esc(a)}</th>" for a in ARMS) + \
           "<th>|½Q|/|Jd|</th><th>err J</th><th>err J+Q</th><th>cos J</th><th>cos J+Q</th></tr>"
    rows = []
    for r in res:
        cells = []
        for k in range(len(ARMS)):
            hit = next((i for i, x in enumerate(r["rk"][k]) if x == 1), None)
            bg = f' style="background:{tint(hit)}"' if hit is not None else ""
            top = " ".join(esc(s[:10]) for s, _ in r["top"][k])
            rk = " ".join(f"{esc(w)}:{x}" for (w, _), x in zip(tracks, r["rk"][k]))
            cells.append(f'<td{bg}>{top}<div class="sm">{rk}</div></td>')
        good = "font-weight:600" if r["eJQ"] < r["eJ"] else ""
        rows.append(f"<tr><th>L{r['l']}</th><td>{r['dn']:.2f}</td>{''.join(cells)}<td>{r['r']:.2f}</td>"
                    f"<td>{r['eJ']:.2f}</td><td style='{good}'>{r['eJQ']:.2f}</td><td>{r['cJ']:.2f}</td>"
                    f"<td>{r['cJQ']:.2f}</td></tr>")
    note = (f"pos {t} ({esc(tstr(CUR['ids'][0, t]))}), dose λ = {lam:g}, {len(res)} layers in {el:.0f}s. "
            "Each cell: top-3 tokens, then tracked ranks. err = |pred − truth| / |truth| against the measured "
            "average change; bold when adding ½Q lowers it.")
    return f'<div class="ql">{key_html(tracks)}<div style="overflow:auto"><table>{head}{"".join(rows)}</table></div>' \
           f'<div class="sm">{note}</div></div>'


def column_fig(res, tracks, t):
    if not tracks or not res:
        return None
    n = len(tracks)
    fig, axs = plt.subplots(1, n, figsize=(min(4.2 * n, 16), 3.4), dpi=110, squeeze=False)
    ls = [r["l"] for r in res]
    for j, (w, _) in enumerate(tracks):
        ax = axs[0, j]
        for k, a in enumerate(ARMS):
            ax.plot(ls, [r["rk"][k][j] for r in res], color=PAL[k], ls=STY[k], marker=MRK[k], ms=3, lw=2, label=a)
        style_rank_ax(ax, f"rank of '{w}'")
        if j:
            ax.get_legend().remove()
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------------------------------- tab 3
DOSES = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.25, 1.5, 2.0]


@locked
def cell_exact(pos, l, push, word, alpha, prompt_b, target, lam, track_text):
    if "H" not in CUR:
        return "Run a prompt first.", "", None
    plt.close("all")
    t, l, lam, alpha = parse_pos(pos), int(l), float(lam), float(alpha)
    l = min(max(l, 0), L - 1)
    ids, H, T = CUR["ids"], CUR["H"], CUR["T"]
    h = H[l, t]
    tracks = track_ids(track_text)
    x0 = torch.zeros(D, device=DEV)
    if push.startswith("deviation"):
        u = h - MU[l]; x0 = MU[l] - h
        what = f"d = h − μ (base: state replaced by the corpus mean; λ = 1 is the real state); |d|/|h| = {float(u.norm() / h.norm()):.2f}"
    elif push.startswith("token"):
        e = [i for i in tok.encode(word, add_special_tokens=False) if tok.decode([i]).strip()]  # no bare-space token
        if not e:
            return "Give a token for the push.", "", None
        wid = e[0]
        v = Jm(l).T @ (GAMMA * W_U[wid])
        u = alpha * h.norm() * v / v.norm()
        what = f"toward '{tstr(wid)}' along its J-lens direction, |u| = {alpha:g}·|h| (base: the real state)"
    else:
        with torch.no_grad(), ActivationRecorder(m.layers, at=[l]) as rec:
            idb = m.encode(prompt_b, max_length=256)
            m.forward(idb)
        tb = t - T + idb.shape[1]
        if not 0 <= tb < idb.shape[1]:
            return "Counterfactual prompt too short for this position.", "", None
        u = rec.activations[l][0, tb].float() - h
        what = (f"u = h_B − h at pos {t} (B pos {tb}: '{tstr(idb[0, tb])}'; aligned from the end); λ = 1 is the "
                f"exact patch; |u|/|h| = {float(u.norm() / h.norm()):.2f}")
    X0 = torch.zeros(1, T, D, device=DEV); X0[0, t] = x0
    U = torch.zeros(1, T, D, device=DEV); U[0, t] = u
    is_last = target.startswith("last")
    tgt = "last" if is_last else torch.arange(t, T, device=DEV)
    f = lambda dl: run_F(ids, l, dl, tgt)  # noqa: E731
    Ju, Q = nested(f, X0, U)
    Ju, Q = Ju[0], Q[0]
    with torch.no_grad(), so_core.math_sdpa():
        F0 = f(X0)[0]
        lams = torch.tensor(DOSES + [lam], device=DEV)
        Ft = f(X0 + lams[:, None, None] * U)  # [n, d]
    tr_all = Ft - F0
    truth = tr_all[-1]
    Jt, Qt = lam * Ju, 0.5 * lam ** 2 * Q
    arms = [("J u", Jt), ("½Q(u,u)", Qt), ("J+½Q", Jt + Qt), ("truth", truth)]
    eJ, eJQ = relerr(Jt, truth), relerr(Jt + Qt, truth)
    # promoted tokens
    lgs = [logits_of(y) for _, y in arms]
    head = "<tr><th>#</th>" + "".join(f"<th>{esc(a)}</th>" for a, _ in arms) + "</tr>"
    tops = [topk(lg, 10) for lg in lgs]
    body = "".join("<tr><td>" + str(i + 1) + "</td>" + "".join(f"<td>{esc(tp[i][0][:14])}</td>" for tp in tops) + "</tr>"
                   for i in range(10))
    if tracks:
        body += "".join(f'<tr><th style="background:{tint(j)}">{esc(w)}</th>' +
                        "".join(f"<td>rank {rank(lg, ids_)}</td>" for lg in lgs) + "</tr>"
                        for j, (w, ids_) in enumerate(tracks))
    promo = f'<div class="ql"><b>What each term pushes up</b> (readout W_U(γ·Δ))<table>{head}{body}</table></div>'
    # next-token predictions (only meaningful for the last-position target)
    pred = ""
    if is_last:
        cols = [("base (λ = 0)", F0), (f"J (λ = {lam:g})", F0 + Jt), (f"J+½Q (λ = {lam:g})", F0 + Jt + Qt),
                (f"truth (λ = {lam:g})", F0 + truth)]
        ps = [next_probs(y) for _, y in cols]
        tp_ = [topk(p, 6) for p in ps]
        b2 = "".join("<tr>" + "".join(f"<td>{esc(tp[i][0][:12])} <span class='sm'>{tp[i][1]:.2f}</span></td>"
                                      for tp in tp_) + "</tr>" for i in range(6))
        if tracks:
            b2 += "".join("<tr>" + "".join(f"<td style='background:{tint(j)}'>{esc(w)}: "
                                           f"{float(p[ids_].max()):.3f}</td>" for p in ps) + "</tr>"
                          for j, (w, ids_) in enumerate(tracks))
        pred = ('<div class="ql"><b>Next-token prediction</b> (final norm + unembedding of base + term)<table><tr>' +
                "".join(f"<th>{esc(a)}</th>" for a, _ in cols) + f"</tr>{b2}</table></div>")
    summ = (f"**L{l}, pos {t} ('{tstr(ids[0, t])}'), target: {target}.** Push: {what}.  \n"
            f"|J u| = {float(Jt.norm()):.3g}, |½Q| = {float(Qt.norm()):.3g}, ratio {float(Qt.norm() / Jt.norm()):.2f}; "
            f"|truth| = {float(truth.norm()):.3g}.  \n"
            f"error vs truth: **J {eJ:.2f}**, **J+½Q {eJQ:.2f}**; cos to truth: J {cos(Jt, truth):.2f}, "
            f"J+½Q {cos(Jt + Qt, truth):.2f}.")
    # dose curve
    fig, ax = plt.subplots(figsize=(7, 3.2), dpi=110)
    eJs = [relerr(a * Ju, tr_all[i]) for i, a in enumerate(DOSES)]
    eQs = [relerr(a * Ju + 0.5 * a * a * Q, tr_all[i]) for i, a in enumerate(DOSES)]
    ax.plot(DOSES, eJs, color=PAL[0], marker="o", ms=4, lw=2, label="J")
    ax.plot(DOSES, eQs, color=PAL[1], marker="s", ms=4, lw=2, ls="--", label="J + ½Q")
    ax.axvline(lam, color="#888", lw=1, ls=":")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("dose λ (push = λ u)"); ax.set_ylabel("|pred − truth| / |truth|")
    ax.set_title("how far each order reaches", fontsize=10, loc="left")
    ax.grid(alpha=.25, lw=.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    return summ, pred + promo, fig



# ---------------------------------------------------------------------------------------------------- tab 4: sweep
# The paper's lens comparison (src/second_order.py stage B3 with --b3-exact-pos 3, the "B3x" run) on the 55
# order-of-operations prompts, at every 4th layer instead of 7, keeping each layer's ranks. Same mu, contexts, seeds:
# J d over all valid source positions (mean), Q exact same-position block at 3 sampled positions x 2 contexts.
OO = json.load(open(os.path.join(JLENS_DIR, "data", "evaluations", "lens-eval-order-ops.json")))["items"]
NUMW = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
        "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty"]
OPSW = {"addition": ["+", "plus", "add", "addition", "sum"], "subtraction": ["-", "minus", "subtract", "subtraction",
        "difference"], "multiplication": ["*", "×", "times", "multiply", "multiplication", "product"],
        "division": ["/", "÷", "divided", "divide", "division", "quotient"]}


def variants_oo(word):  # = second_order.variants(word, order_ops=True)
    words = {word}
    if word.isdigit() and int(word) < len(NUMW):
        words.add(NUMW[int(word)])
    for k, syn in OPSW.items():
        if word.lower() in syn or word.lower() == k:
            words |= set(syn)
    ids = set()
    for w_ in words:
        for v in {w_, w_.lower(), w_.capitalize()}:
            for p in ("", " "):
                e = [i for i in tok.encode(p + v, add_special_tokens=False) if tok.decode([i]).strip()]  # skip bare-space first token: ' 9' -> ' ', '9'
                if e:
                    ids.add(e[0])
    return sorted(ids)


SW_LAYERS = [32, 48, 16, 56, 24, 40, 8, 60, 4, 12, 20, 28, 36, 44, 52]  # coarse first, so partial results are useful
SW_ARMS = ["logit lens", "J-lens (fitted)", "J h", "J d", "½Q(d,d)", "J d + ½Q"]
SW_WORDS = [list(it["intermediates"]) + [it["target"]] for it in OO]  # last word = the answer
SW_VAR = [[variants_oo(w_) for w_ in ws] for ws in SW_WORDS]
SW = {"done": [], "res": {}, "status": "waiting to start", "t0": None, "H": None}
SW_OUT = f"{W}/orderops_layers"


def sweep_states():
    hs = []
    for it in OO:
        ids = m.encode(it["prompt"], max_length=512)
        with GPU, torch.no_grad(), ActivationRecorder(m.layers, at=SW_LAYERS) as rec:
            m.forward(ids)
            hs.append(torch.stack([rec.activations[l][0, -1].float() for l in SW_LAYERS]))
    return torch.stack(hs)  # [N, n_layers_swept, d]


def sweep_layer(l, Hl, n_ctx=2, exact_pos=3, batch=8):
    N = Hl.shape[0]
    d = Hl - MU[l]
    Jd = torch.zeros(N, D, device=DEV); Q = torch.zeros_like(Jd); Jmu = torch.zeros(D, device=DEV)
    pg = torch.Generator().manual_seed(1234 + l)
    nV = len(VALID)
    for ci in range(n_ctx):
        ids = CTX[ci]
        tps = VALID[torch.randperm(nV, generator=pg)[:exact_pos].to(DEV)].tolist()
        f = lambda dl: run_F(ids, l, dl, VALID)  # noqa: E731
        with GPU, so_core.math_sdpa():
            A1 = torch.zeros(1, 128, D, device=DEV); A1[:, VALID] = MU[l]
            Jmu += jvp(f, (torch.zeros_like(A1),), (A1,))[1][0] / nV
        for s in range(0, N, batch):
            aa = d[s: s + batch]; B = aa.shape[0]
            with GPU:
                A1 = torch.zeros(B, 128, D, device=DEV); A1[:, VALID] = aa[:, None, :]
                with so_core.math_sdpa():
                    Jd[s: s + B] += jvp(f, (torch.zeros_like(A1),), (A1,))[1] / nV
                for tp in tps:
                    A = torch.zeros(B, 128, D, device=DEV); A[:, tp] = aa
                    Q[s: s + B] += nested(f, torch.zeros_like(A), A)[1] / len(tps)
    return Jd / n_ctx, Q / n_ctx, Jmu / n_ctx


@torch.no_grad()
def sweep_readouts(l, Hl, Jd, Q, Jmu, items):
    outs = [Hl, Hl @ Jm(l).T, Jd + Jmu, Jd, Q, Jd + 0.5 * Q]
    res = {"ranks": {}, "top": {}}
    for k, y in zip(SW_ARMS, outs):
        lg = logits_of(y)
        res["ranks"][k] = [[rank(lg[i], v) for v in SW_VAR[it]] for i, it in enumerate(items)]
        res["top"][k] = [[tstr(j) for j in lg[i].topk(3).indices.tolist()] for i in range(len(items))]
    return res


def sweep_run():
    try:
        SW["status"] = "running"; SW["t0"] = time.time()
        os.makedirs(SW_OUT, exist_ok=True)
        Hs = sweep_states()
        for li, l in enumerate(SW_LAYERS):
            Jd, Q, Jmu = sweep_layer(l, Hs[:, li])
            with GPU:
                SW["res"][l] = sweep_readouts(l, Hs[:, li], Jd, Q, Jmu, list(range(len(OO))))
            SW["done"].append(l)
            log(f"sweep L{l} done ({len(SW['done'])}/{len(SW_LAYERS)}, {time.time() - SW['t0']:.0f}s)")
            with open(f"{SW_OUT}/sweep_wsfix.json", "w") as fh:  # sweep.json = pre-fix (space-token bug)
                json.dump({"model": MODEL, "layers_done": SW["done"], "arms": SW_ARMS, "items": OO,
                           "words": SW_WORDS, "res": {str(k): v for k, v in SW["res"].items()},
                           "method": "B3x per layer: last position; d = h - mu_l (8 ctx); J d all valid sources mean; "
                                     "Q exact at 3 positions x 2 ctx; ranks 1-based, min over first-token variants"},
                          fh, ensure_ascii=False)
        SW["status"] = "done"
    except Exception:
        SW["status"] = "FAILED:\n" + traceback.format_exc()
        log(SW["status"])


def sweep_view(item):
    plt.close("all")
    i = int(str(item).split(":")[0]) if item else 2
    done = sorted(SW["done"])
    el = f", {time.time() - SW['t0']:.0f}s" if SW["t0"] else ""
    st = (f"**Sweep: {len(done)}/{len(SW_LAYERS)} layers done ({SW['status'].splitlines()[0]}{el}).** "
          "Press Refresh to update. Last position; rank 1 = top; numbers also match as words, operations by synonyms.")
    if SW["status"].startswith("FAILED"):
        st += f"\n```\n{SW['status'][-1500:]}\n```"
    if not done:
        return st, None, "", None
    words = SW_WORDS[i]
    nw = len(words)
    fig, axs = plt.subplots(1, nw, figsize=(min(4.4 * nw, 16), 3.5), dpi=110, squeeze=False)
    for j, w_ in enumerate(words):
        ax = axs[0, j]
        for k, a in enumerate(SW_ARMS):
            ax.plot(done, [SW["res"][l]["ranks"][a][i][j] for l in done], color=PAL[k], ls=STY[k], marker=MRK[k],
                    ms=3, lw=2, label=a)
        style_rank_ax(ax, ("answer " if j == nw - 1 else "intermediate ") + f"'{w_}'")
        if j:
            ax.get_legend().remove()
    fig.suptitle(OO[i]["prompt"].strip() + "  →  " + OO[i]["target"], fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    head = "<tr><th>layer</th>" + "".join(f"<th>{esc(a)}</th>" for a in SW_ARMS) + "</tr>"
    rows = []
    for l in done:
        cells = []
        for a in SW_ARMS:
            rk = SW["res"][l]["ranks"][a][i]
            hit = next((j for j, r in enumerate(rk) if r == 1), None)
            bg = f' style="background:{tint(hit)}"' if hit is not None else ""
            top = " ".join(esc(t_[:10]) for t_ in SW["res"][l]["top"][a][i])
            rs = " ".join(f"{esc(w_)}:{r}" for w_, r in zip(words, rk))
            cells.append(f'<td{bg}>{top}<div class="sm">{rs}</div></td>')
        rows.append(f"<tr><th>L{l}</th>{''.join(cells)}</tr>")
    key = "".join(f'<span style="background:{tint(j)}">{esc(w_)}{" (answer)" if j == nw - 1 else ""}</span>'
                  for j, w_ in enumerate(words))
    tab = (f'<div class="ql"><div class="key">top-1 tint: {key}</div><div style="overflow:auto"><table>{head}'
           f'{"".join(rows)}</table></div><div class="sm">Each cell: top-3 tokens, then ranks.</div></div>')
    # aggregate: median rank over the numeric intermediates of all 55 prompts
    agg, ax = plt.subplots(1, 2, figsize=(11, 3.4), dpi=110)
    import statistics
    for k, a in enumerate(SW_ARMS):
        mi = [statistics.median(SW["res"][l]["ranks"][a][ii][j] for ii in range(len(OO))
                                for j, w_ in enumerate(SW_WORDS[ii][:-1]) if w_.isdigit()) for l in done]
        ma = [statistics.median(SW["res"][l]["ranks"][a][ii][-1] for ii in range(len(OO))) for l in done]
        ax[0].plot(done, mi, color=PAL[k], ls=STY[k], marker=MRK[k], ms=3, lw=2, label=a)
        ax[1].plot(done, ma, color=PAL[k], ls=STY[k], marker=MRK[k], ms=3, lw=2, label=a)
    style_rank_ax(ax[0], "all 55 prompts: median rank of the numeric intermediate")
    style_rank_ax(ax[1], "all 55 prompts: median rank of the answer")
    ax[1].get_legend().remove()
    agg.tight_layout()
    return st, fig, tab, agg


# ---------------------------------------------------------------------------------------------------- UI
INTRO = """### J / Q / J+Q lens on Qwen3.8-27B  ·  **4 · Order-ops sweep** fills in by itself in the background
**1 · Prompt** runs the fitted J-lens over every layer and position (the view where `8 * (7 + 2) = ` shows `9` before `72`).
**2 · Q lens (averaged)** reads a state's deviation from the mean, d = h − μ, per layer the way the J-lens does
(averaged over corpus contexts): J d, ½Q(d,d), their sum, and the measured average change. **3 · One cell (exact)** pushes
one state inside *this* prompt and compares J, ½Q, J+½Q with what the model actually does, across push sizes.
Qwen splits numbers into digits, so `72` is tracked by its first token `7`. All maths in fp32, TF32 off; one request at a time."""


def build():
    import gradio as gr
    with gr.Blocks(title="Q-lens explorer") as demo:
        gr.Markdown(INTRO)
        with gr.Row():
            prompt = gr.Textbox("8 * (7 + 2) = ", label="prompt", scale=4)
            track = gr.Textbox("9, 72, 63", label="tracked tokens (comma-separated, up to 6)", scale=2)
            window = gr.Number(16, label="positions shown", precision=0, scale=1)
            go = gr.Button("Run", variant="primary", scale=1)
        info = gr.Markdown()
        with gr.Tabs():
            with gr.Tab("1 · Prompt (fitted J-lens)"):
                grid = gr.HTML()
                rfig = gr.Plot(label="tracked ranks, last position")
            with gr.Tab("2 · Q lens (averaged)"):
                with gr.Row():
                    pos2 = gr.Dropdown([], label="position")
                    layers = gr.Textbox("4-60:4", label="layers (e.g. 4-60:4 or 20,30,40)")
                    lam2 = gr.Slider(0.05, 1.5, value=1.0, step=0.05, label="dose λ (d → λd)")
                    nctx = gr.Slider(1, 4, value=1, step=1, label="corpus contexts")
                    npos = gr.Slider(1, 8, value=4, step=1, label="positions per context")
                go2 = gr.Button("Compute column", variant="primary")
                col = gr.HTML()
                cfig = gr.Plot(label="tracked ranks by readout")
            with gr.Tab("3 · One cell (exact, this prompt)"):
                with gr.Row():
                    pos3 = gr.Dropdown([], label="position")
                    lay3 = gr.Slider(1, L - 1, value=32, step=1, label="layer")
                    target = gr.Radio(["last position (next token)", "this and later positions"],
                                      value="last position (next token)", label="read at")
                with gr.Row():
                    push = gr.Radio(["deviation from mean (d = h − μ)", "token direction", "counterfactual prompt"],
                                    value="deviation from mean (d = h − μ)", label="push")
                    word = gr.Textbox("72", label="token (token direction)")
                    alpha = gr.Slider(0.02, 1.0, value=0.2, step=0.02, label="size |u|/|h| (token direction)")
                    pb = gr.Textbox("8 * (7 + 3) = ", label="prompt B (counterfactual)")
                lam3 = gr.Slider(0.05, 2.0, value=1.0, step=0.05, label="dose λ")
                go3 = gr.Button("Compute cell", variant="primary")
                summ = gr.Markdown()
                tables = gr.HTML()
                dfig = gr.Plot(label="dose curve")
            with gr.Tab("4 · Order-ops sweep (55 prompts × layers)"):
                sw_st = gr.Markdown()
                with gr.Row():
                    item = gr.Dropdown([f"{i}: {it['prompt'].strip()}  → {it['target']}" for i, it in enumerate(OO)],
                                       value=f"2: {OO[2]['prompt'].strip()}  → {OO[2]['target']}", label="prompt",
                                       scale=4)
                    refresh = gr.Button("Refresh", scale=1)
                sw_fig = gr.Plot(label="rank vs layer, this prompt")
                sw_tab = gr.HTML()
                sw_agg = gr.Plot(label="all prompts")
        go.click(run_prompt, [prompt, track, window], [info, grid, rfig, pos2, pos3])
        prompt.submit(run_prompt, [prompt, track, window], [info, grid, rfig, pos2, pos3])
        go2.click(lens_column, [pos2, layers, lam2, nctx, npos, track], [col, cfig])
        refresh.click(sweep_view, [item], [sw_st, sw_fig, sw_tab, sw_agg], concurrency_limit=None)
        item.change(sweep_view, [item], [sw_st, sw_fig, sw_tab, sw_agg], concurrency_limit=None)
        go3.click(cell_exact, [pos3, lay3, push, word, alpha, pb, target, lam3, track], [summ, tables, dfig])
    demo.queue(default_concurrency_limit=1)
    return demo


def selftest():
    for name, fn in [
        ("run_prompt", lambda: run_prompt("8 * (7 + 2) = ", "9, 72", 16)),
        ("lens_column", lambda: list(lens_column(f"{CUR['T'] - 1}: x", "40", 1.0, 1, 4, "9, 72"))),
        ("cell dev", lambda: cell_exact(f"{CUR['T'] - 1}: x", 32, "deviation from mean (d = h − μ)", "72", 0.2, "", "last", 1.0, "9")),
        ("cell tok", lambda: cell_exact(f"{CUR['T'] - 4}: x", 32, "token direction", "72", 0.2, "", "this", 1.0, "9")),
        ("cell cf", lambda: cell_exact(f"{CUR['T'] - 3}: x", 32, "counterfactual prompt", "72", 0.2, "8 * (7 + 3) = ", "last", 1.0,
                                       "9")),
    ]:
        t0 = time.time()
        try:
            out = fn()
            msg = out[0] if name.startswith("cell") else ""
            log(f"selftest {name}: ok {time.time() - t0:.1f}s {str(msg)[:300]}")
        except Exception:
            log(f"selftest {name}: FAILED\n{traceback.format_exc()}")
    try:  # sweep code path on 2 prompts, 1 layer, 1 context, 1 position (not saved)
        t0 = time.time()
        with GPU, torch.no_grad(), ActivationRecorder(m.layers, at=[40]) as rec:
            hs = []
            for it in OO[:2]:
                m.forward(m.encode(it["prompt"], max_length=512)); hs.append(rec.activations[40][0, -1].float())
        Hl = torch.stack(hs)
        Jd, Q, Jmu = sweep_layer(40, Hl, n_ctx=1, exact_pos=1, batch=2)
        r = sweep_readouts(40, Hl, Jd, Q, Jmu, [0, 1])
        log(f"selftest sweep: ok {time.time() - t0:.1f}s ranks {json.dumps({k: v for k, v in r['ranks'].items()})}")
    except Exception:
        log(f"selftest sweep: FAILED\n{traceback.format_exc()}")
    # finite-difference check of the nested jvp at one cell (central second difference, Richardson)
    try:
        ids, H, T = CUR["ids"], CUR["H"], CUR["T"]
        l, t = 32, T - 1
        U = torch.zeros(1, T, D, device=DEV); U[0, t] = (H[l, t] - MU[l]) * 0.1
        f = lambda dl: run_F(ids, l, dl, "last")  # noqa: E731
        Ju, Q = nested(f, torch.zeros_like(U), U)
        with torch.no_grad(), so_core.math_sdpa():
            def sd(e):
                return (f(e * U) - 2 * f(0 * U) + f(-e * U)) / e ** 2
            q1, q2 = sd(0.5), sd(0.25)
            qr = (4 * q2 - q1) / 3
        log(f"selftest FD: rel|Q - FD_rich| = {relerr(Q, qr):.2e}, rel|Q - FD(0.25)| = {relerr(Q, q2):.2e}, "
            f"|½Q|/|Ju| = {float((0.5 * Q).norm() / Ju.norm()):.3f}")
    except Exception:
        log(f"selftest FD: FAILED\n{traceback.format_exc()}")
    plt.close("all")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--sweep-only", action="store_true", help="run the tab-4 order-ops sweep as a batch job, no UI")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    if a.sweep_only:
        sweep_run()
        log(f"sweep-only finished: {SW['status'][:200]}")
        sys.exit(0 if SW["status"] == "done" else 1)
    threading.Thread(target=sweep_run, daemon=True).start()
    import gradio as gr
    pw = open(os.environ["DEMO_AUTH_FILE"]).read().strip()
    log(f"serving on port {os.environ['PORT']}")
    build().launch(server_name="0.0.0.0", server_port=int(os.environ["PORT"]), auth=(os.environ["USER"], pw),
                   ssr_mode=False, css=CSS, theme=gr.themes.Soft())
