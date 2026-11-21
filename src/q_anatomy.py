"""Q-anatomy: is the second-order term local to one block / head / neuron, or a multi-layer object?

Setup (as D1 in second_order.py): push U (rho_l * unit direction) at every valid position of block l's output;
F = sum over valid positions of block L's output (pre-norm). Every block k > l has four nonlinear SITES:
    n1  input RMSNorm                     (function of the block input h_{k-1})
    mix attention / linear attention      (function of n1's output)
    n2  post-attention RMSNorm            (function of h_mid = h_{k-1} + mix)
    mlp MLP                               (function of n2's output)
Residual adds are linear. The second-order chain rule is exact and has no cross-site terms:
    Q_F(U,U) = sum_X J_{X->L} s_X,      s_X = D^2 f_X [tau_X, tau_X],
tau_X = first-order tangent at X's input; c_X = J_{X->L} s_X is site X's contribution (stage A2). Inside mix and
mlp the source splits further: per head (o_proj input) and per neuron (down_proj input), read out along
q_hat = Q_F/|Q_F| with one reverse pass (cotangents at every site).
Third order is where sites first interact (stage A3):
    T_F(U,U,U) = sum_X J_{X->L} ( D^3 f_X[tau,tau,tau]  +  3 D^2 f_X[tau_X, q_X] ),
q_X = second-order tangent at X's input = curvature made upstream. "self" = the D^3 term, "cross" = the 3 D^2 term;
--a3-pair splits the cross term by the upstream block j that made the curvature: M[j,k] along t_hat = T_F/|T_F|.

Stages: check (exactness gates + timing), A2, A3. Results: <out>/{check,A2,A3}<name>.json (+ *_raw.pt).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import transformers
from torch.func import jvp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
JLENS_DIR = os.environ.get("JLENS_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "jacobian-lens"))
sys.path.insert(0, JLENS_DIR)
import jlens  # noqa: E402
from jlens.fitting import valid_position_mask  # noqa: E402
from jlens.hooks import ActivationRecorder  # noqa: E402
from so_core import Perturbed, first_and_second, math_sdpa, no_tf32  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen3-8B")
ap.add_argument("--corpus", default=os.path.join(os.environ.get("QLENS_OUT", "outputs"), "corpus/fineweb_edu_2000.jsonl"))
ap.add_argument("--out", required=True)
ap.add_argument("--name", default="", help="suffix for output files (shards)")
ap.add_argument("--stages", default="check,A2,A3")
ap.add_argument("--sources", default="8,16,24,32,40,48,56", help="A2 sources, nominal 64-layer indices (rescaled as D1)")
ap.add_argument("--types", default="iso,data")
ap.add_argument("--n-ctx", type=int, default=8)
ap.add_argument("--n-dir", type=int, default=8)
ap.add_argument("--a3-sources", default="8,24,40")
ap.add_argument("--a3-types", default="iso,data")
ap.add_argument("--a3-n-ctx", type=int, default=4)
ap.add_argument("--a3-pair", type=int, default=1)
ap.add_argument("--alphas", default="0.05,0.1,0.2,0.4")
ap.add_argument("--check-source", default="48")
ap.add_argument("--check-n-dir", type=int, default=4)
ap.add_argument("--tf32", type=int, default=1, help="TF32 in bulk stages (check measures its effect on the gates)")
ap.add_argument("--max-len", type=int, default=128)
ap.add_argument("--ctx-start", type=int, default=1000)
ap.add_argument("--save-raw", type=int, default=1)
args = ap.parse_args()
stages = args.stages.split(",")
out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
SITES = ("n1", "mix", "n2", "mlp")
INJ = {"n1": "in", "mix": "n1", "n2": "mix", "mlp": "n2"}  # perturbing here moves only site X's input
RANK = {"in": 0, "n1": 1, "oin": 2, "mix": 3, "n2": 4, "din": 5, "mlp": 6, "out": 7}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def dump(stem, obj):
    p = out / f"{stem}{args.name}.json"
    p.write_text(json.dumps(obj, indent=1, default=str))
    log(f"wrote {p}")


# ---------------------------------------------------------------- model
dtype = torch.float32
DEV = os.environ.get("QANAT_DEVICE", "cuda")  # "cpu" only for the tiny-model unit test


def mem_gb(peak=False):
    if DEV != "cuda":
        return 0.0
    return (torch.cuda.max_memory_allocated() if peak else torch.cuda.memory_allocated()) / 1e9


tok = transformers.AutoTokenizer.from_pretrained(args.model)
_cfg = transformers.AutoConfig.from_pretrained(args.model)
_arch = (_cfg.architectures or [""])[0]
_cls = transformers.AutoModelForImageTextToText if "ConditionalGeneration" in _arch else transformers.AutoModelForCausalLM
hf = _cls.from_pretrained(args.model, dtype=dtype, attn_implementation="sdpa", device_map=DEV)
m = jlens.from_hf(hf, tok)
L = m.n_layers - 1
D = m.d_model
W_U = hf.get_output_embeddings().weight.detach()
with torch.no_grad():
    gamma = m._final_norm(torch.ones(1, D, device=DEV, dtype=dtype)).float()[0]
RU = (W_U.float() * gamma).T.contiguous()  # [d, V]: readout(y) = y @ RU
_sc = lambda xs: [min(L - 1, max(1, round(int(x) * m.n_layers / 64))) for x in xs.split(",")]  # noqa: E731
SRC_A2, SRC_A3, SRC_CHK = sorted(set(_sc(args.sources))), sorted(set(_sc(args.a3_sources))), _sc(args.check_source)[0]
torch.backends.cuda.matmul.allow_tf32 = bool(args.tf32); torch.backends.cudnn.allow_tf32 = bool(args.tf32)
log(f"{args.model} arch={_arch} n_layers={m.n_layers} d={D} A2={SRC_A2} A3={SRC_A3} check={SRC_CHK} tf32={args.tf32} "
    f"mem={mem_gb():.1f} GB")

docs = [json.loads(x)["text"] for x in open(args.corpus)]
ctx_ids = [m.encode(t, max_length=args.max_len) for t in docs[args.ctx_start: args.ctx_start + 64]]
ctx_ids = [c for c in ctx_ids if c.shape[1] == args.max_len]


def residual_stats(layers, n=8):  # identical to second_order.py (D1's rho and mu)
    acc = {l: [] for l in layers}
    with ActivationRecorder(m.layers, at=layers) as rec:
        for ids in ctx_ids[:n]:
            m.forward(ids)
            v = Perturbed(m, ids, 0, L).valid
            for l in layers:
                acc[l].append(rec.activations[l][0, v].float())
    return {l: {"mu": torch.cat(acc[l]).mean(0), "rho": float(torch.cat(acc[l]).norm(dim=-1).mean())} for l in layers}


def directions(srcs, nD):
    """D1's iso and data directions (same seeds and contexts), unit norm, per source layer."""
    st = residual_stats(srcs)
    dgen = torch.Generator().manual_seed(4321)
    data = {l: [] for l in srcs}
    with ActivationRecorder(m.layers, at=srcs) as rec, torch.no_grad():
        for ids in ctx_ids[-nD:]:
            m.forward(ids)
            v = Perturbed(m, ids, 0, L).valid
            tp = int(v[torch.randint(len(v), (1,), generator=dgen)])
            for l in srcs:
                data[l].append(rec.activations[l][0, tp].float() - st[l]["mu"])
    dirs = {}
    for l in srcs:
        iso = torch.randn(nD, D, generator=torch.Generator().manual_seed(77 + l)).to(DEV)
        dirs[l] = {"iso": iso, "data": torch.stack(data[l])}
        dirs[l] = {k: v / v.norm(dim=-1, keepdim=True) for k, v in dirs[l].items()}
    return st, dirs


# ---------------------------------------------------------------- runner: inject / record at sites, skip prefix, stop early
class Stop(Exception):
    pass


def _first(o):
    return o if torch.is_tensor(o) else o[0]


def _add(o, d):
    return o + d if torch.is_tensor(o) else (o[0] + d,) + tuple(o[1:])


def _hidden(args_, kwargs):
    return args_[0] if args_ else kwargs["hidden_states"]


def _with_hidden(args_, kwargs, h):
    if args_:
        return (h,) + tuple(args_[1:]), kwargs
    return args_, {**kwargs, "hidden_states": h}


def _identity(hidden_states, *a, **k):
    return hidden_states


def mixer(lay):
    return lay.self_attn if hasattr(lay, "self_attn") else lay.linear_attn


def module(site, k):
    lay = m.layers[k]
    mix = mixer(lay)
    return {"out": lay, "n1": lay.input_layernorm, "n2": lay.post_attention_layernorm, "mlp": lay.mlp, "mix": mix,
            "oin": getattr(mix, "o_proj", None) or getattr(mix, "out_proj", None), "din": lay.mlp.down_proj}[site]


def head_dim(k):
    mix = mixer(m.layers[k])
    return getattr(mix, "head_v_dim", None) or mix.head_dim


class Runner:
    """One context. run() starts at block `start` from the cached base residual (blocks < start are skipped),
    adds deltas at sites, records site outputs (or inputs, for in/oin/din), and stops after the last record."""

    def __init__(self, ids1):
        self.ids1, self.T = ids1, ids1.shape[1]
        self.valid = valid_position_mask(self.T).nonzero(as_tuple=True)[0].to(ids1.device)
        self.base_in = {}

        def cache(k):
            def hook(_m, a, kw):
                self.base_in[k] = _hidden(a, kw).detach()
            return hook
        hs = [lay.register_forward_pre_hook(cache(k), with_kwargs=True) for k, lay in enumerate(m.layers)]
        try:
            with torch.no_grad():
                m.forward(ids1)
        finally:
            for h in hs:
                h.remove()

    def spread(self, U):
        A = torch.zeros(U.shape[0], self.T, U.shape[1], device=U.device, dtype=U.dtype)
        A[:, self.valid] = U[:, None, :]
        return A

    def run(self, B, start, inject, record, stop=True):
        box, hs = {}, []
        assert all(k >= start for (_, k) in inject) and all(r[0] == "L" or r[1] >= start for r in record)
        last = None if (not stop or any(r[0] == "L" for r in record)) else max(record, key=lambda r: (r[1], RANK[r[0]]))

        def pre_start(_m, a, kw):
            h = self.base_in[start].expand(B, -1, -1).contiguous()
            d = inject.get(("in", start))
            return _with_hidden(a, kw, h if d is None else h + d)
        hs.append(m.layers[start].register_forward_pre_hook(pre_start, with_kwargs=True))
        for (site, k), d in inject.items():
            if site == "in" and k != start:
                hs.append(m.layers[k].register_forward_pre_hook(
                    lambda _m, a, kw, d=d: _with_hidden(a, kw, _hidden(a, kw) + d), with_kwargs=True))
            elif site != "in":
                hs.append(module(site, k).register_forward_hook(lambda _m, _i, o, d=d: _add(o, d)))
        for r in record:
            site, k = r
            if site == "L":
                hs.append(m.layers[L].register_forward_hook(
                    lambda _m, _i, o, r=r: box.__setitem__(r, _first(o)[:, self.valid].sum(1))))
                continue

            def grab(v, r=r):
                box[r] = v
                if r == last:
                    raise Stop
            if site in ("in", "oin", "din"):
                mod = m.layers[k] if site == "in" else module(site, k)
                hs.append(mod.register_forward_pre_hook(lambda _m, a, kw, g=grab: g(_hidden(a, kw)), with_kwargs=True))
            else:
                hs.append(module(site, k).register_forward_hook(lambda _m, _i, o, g=grab: g(_first(o))))
        for k in range(start):
            m.layers[k].forward = _identity
        try:
            with math_sdpa():
                m.forward(self.ids1.expand(B, -1))
        except Stop:
            pass
        finally:
            for h in hs:
                h.remove()
            for k in range(start):
                m.layers[k].__dict__.pop("forward", None)
        return tuple(box[r] for r in record)


# ---------------------------------------------------------------- derivatives
def d2(fn, zs, t, u):
    """(D fn[t], D^2 fn[t, u]) at zs; zs, t, u tuples of tensors."""
    inner = lambda *ds: jvp(fn, ds, t)[1]  # noqa: E731
    return jvp(inner, zs, u)


def d3(fn, zs, t):
    """(D^2 fn[t,t], D^3 fn[t,t,t]) at zs."""
    inner = lambda *ds: jvp(fn, ds, t)[1]  # noqa: E731
    inner2 = lambda *ds: jvp(inner, ds, t)[1]  # noqa: E731
    return jvp(inner2, zs, t)


def all_sites(R, l, A1):
    """Nested jvp of the whole span with every site recorded: first- and second-order tangents at every site."""
    blocks = range(l + 1, L + 1)
    rec = [(X, k) for k in blocks for X in SITES] + [("out", k) for k in blocks] + [("L", None)]
    f = lambda d: R.run(A1.shape[0], l + 1, {("in", l + 1): d}, rec, stop=False)  # noqa: E731
    t1, t2 = d2(f, (torch.zeros_like(A1),), (A1,), (A1,))
    return dict(zip(rec, t1)), dict(zip(rec, t2))


def site_inputs(l, h0, tans):
    """Tangent at each site's input, from recorded site outputs (h0 = tangent at block l+1's input)."""
    tin = {}
    for k in range(l + 1, L + 1):
        hin = h0 if k == l + 1 else tans[("out", k - 1)]
        tin[("n1", k)] = hin
        tin[("mix", k)] = tans[("n1", k)]
        tin[("n2", k)] = hin + tans[("mix", k)]
        tin[("mlp", k)] = tans[("n2", k)]
    return tin


def resid_check(l, h0, tans):
    worst = 0.0
    for k in range(l + 1, L + 1):
        hin = h0 if k == l + 1 else tans[("out", k - 1)]
        pred = hin + tans[("mix", k)] + tans[("mlp", k)]
        worst = max(worst, float((pred - tans[("out", k)]).norm() / tans[("out", k)].norm().clamp_min(1e-30)))
    return worst


def rows(ts, b):
    """4 tensors [b,...] -> 4 tensors [4b,...], tensor i nonzero only in row block i (one site per row block)."""
    outs = []
    for i, t in enumerate(ts):
        z = torch.zeros((4 * b,) + tuple(t.shape[1:]), device=t.device, dtype=t.dtype)
        z[i * b:(i + 1) * b] = t
        outs.append(z)
    return tuple(outs)


def block_local(R, b, k, extra):
    inj = [(INJ[X], k) for X in SITES]
    rec = [("n1", k)] + ([("oin", k)] if extra else []) + [("mix", k), ("n2", k)] + ([("din", k)] if extra else []) \
        + [("mlp", k)]
    return (lambda *ds: R.run(4 * b, k, dict(zip(inj, ds)), rec, stop=True)), rec


def pick(res, rec, k, b):
    d = dict(zip(rec, res))
    o = {X: d[(X, k)][i * b:(i + 1) * b] for i, X in enumerate(SITES)}
    if ("oin", k) in d:
        o["oin"], o["din"] = d[("oin", k)][b:2 * b], d[("din", k)][3 * b:4 * b]
    return o


def block_d2(R, b, k, t, u=None, extra=False):
    """Per-site D^2 f_X[t_X, u_X] for the four sites of block k in one batched call (t, u: dicts site -> [b,T,d])."""
    fn, rec = block_local(R, b, k, extra)
    T_ = rows([t[(X, k)] for X in SITES], b)
    U_ = T_ if u is None else rows([u[(X, k)] for X in SITES], b)
    _, r2 = d2(fn, tuple(torch.zeros_like(x) for x in T_), T_, U_)
    return pick(r2, rec, k, b)


def block_d3(R, b, k, t):
    fn, rec = block_local(R, b, k, False)
    T_ = rows([t[(X, k)] for X in SITES], b)
    _, r3 = d3(fn, tuple(torch.zeros_like(x) for x in T_), T_)
    return pick(r3, rec, k, b)


def push_down(R, k, s):
    """J_{X_k -> L} s_X for the four sites of block k (s: dict X -> [b,T,d]); one batched jvp."""
    b = s["n1"].shape[0]
    T_ = rows([s[X] for X in SITES], b)
    fn = lambda *ds: R.run(4 * b, k, dict(zip([(X, k) for X in SITES], ds)), [("L", None)], stop=False)  # noqa: E731
    _, (c,) = jvp(fn, tuple(torch.zeros_like(x) for x in T_), T_)
    return {X: c[i * b:(i + 1) * b] for i, X in enumerate(SITES)}


def cotangents(R, b, start, sites, cot):
    """J_{X->L}^T cot at each site output (reverse mode, one pass). cot [b, d] on F."""
    ds = {s: torch.zeros(b, R.T, D, device=DEV, dtype=dtype, requires_grad=True) for s in sites}
    with torch.enable_grad():
        (F,) = R.run(b, start, ds, [("L", None)], stop=False)
        gs = torch.autograd.grad(F, list(ds.values()), grad_outputs=cot)
    return {s: g.detach() for s, g in zip(sites, gs)}


def forward_F(R, l, A):
    with torch.no_grad():
        return R.run(A.shape[0], l + 1, {("in", l + 1): A}, [("L", None)], stop=False)[0]


# ---------------------------------------------------------------- statistics
def lg(y):
    z = y.float() @ RU
    return z - z.mean(-1, keepdim=True)  # centred logits (a uniform shift is irrelevant to the softmax)


def n_stay(err, eps):
    bad = (err > eps).nonzero()
    return int(bad.max()) + 2 if len(bad) else 1


def vec_stats(C, Q):
    """C [N, d] contributions (sum = Q up to the gate). Localisation of Q over components."""
    qn = Q.norm().clamp_min(1e-30)
    nrm = C.norm(dim=-1)
    S = C.sum(0)
    order = nrm.argsort(descending=True)
    err = (Q[None] - C[order].cumsum(0)).norm(dim=-1) / qn
    single = (Q[None] - C).norm(dim=-1) / qn
    return {"gate": float((S - Q).norm() / qn), "pr": float(nrm.sum() ** 2 / nrm.pow(2).sum().clamp_min(1e-30)),
            "ci": float(nrm.sum() / S.norm().clamp_min(1e-30)), "top1_norm_share": float(nrm.max() / nrm.sum()),
            "err_top": [float(err[i]) for i in range(min(10, len(err)))],
            "n50": n_stay(err, 0.5), "n90": n_stay(err, 0.1),
            "best_single_err": float(single.min()), "best_single": int(single.argmin())}


def scal_stats(p):
    """p [N] scalar contributions; total S = sum p."""
    S = p.sum()
    a = p.abs()
    order = a.argsort(descending=True)
    err = (S - p[order].cumsum(0)).abs() / S.abs().clamp_min(1e-30)
    return {"total": float(S), "pr": float(a.sum() ** 2 / p.pow(2).sum().clamp_min(1e-30)),
            "ci": float(a.sum() / S.abs().clamp_min(1e-30)),
            "top_share": {n: float(a[order][:n].sum() / a.sum()) for n in (1, 10, 100, 1000) if n <= len(p)},
            "err_top": {n: float(err[n - 1]) for n in (1, 10, 100, 1000) if n <= len(p)},
            "n50": n_stay(err, 0.5), "n90": n_stay(err, 0.1), "N": len(p)}


def med(xs):
    xs = [x for x in xs if x == x]
    return float(torch.tensor(xs).median()) if xs else None


# ---------------------------------------------------------------- A2: second-order anatomy for one (context, direction batch)
def anatomy2(R, l, U, timing=None):
    b = U.shape[0]
    A1 = R.spread(U)
    t0 = time.perf_counter()
    T1, T2 = all_sites(R, l, A1)
    Jf, Qf = T1[("L", None)], T2[("L", None)]
    tin = site_inputs(l, A1, T1)
    rc = resid_check(l, A1, T1)
    del T2
    qhat = Qf / Qf.norm(dim=-1, keepdim=True)
    g = cotangents(R, b, l + 1, [(X, k) for k in range(l + 1, L + 1) for X in ("mix", "mlp")], qhat)
    t1 = time.perf_counter()
    blocks = list(range(l + 1, L + 1))
    C = {}
    heads, neur_p, neur_top, chk_h, chk_n = {}, {}, {}, 0.0, 0.0
    for k in blocks:
        s = block_d2(R, b, k, tin, extra=True)
        c = push_down(R, k, s)
        for X in SITES:
            C[(X, k)] = c[X]
        Wo, Wd = module("oin", k).weight.float(), module("din", k).weight.float()
        ph = ((g[("mix", k)] @ Wo) * s["oin"]).view(b, R.T, -1, head_dim(k)).sum((1, 3))  # [b, H]
        pn = ((g[("mlp", k)] @ Wd) * s["din"]).sum(1)  # [b, I]
        pm = (qhat * c["mix"]).sum(-1); pl = (qhat * c["mlp"]).sum(-1)
        chk_h = max(chk_h, float(((ph.sum(1) - pm).abs() / Qf.norm(dim=-1)).max()))
        chk_n = max(chk_n, float(((pn.sum(1) - pl).abs() / Qf.norm(dim=-1)).max()))
        heads[k], neur_p[k] = ph, pn
        tv, ti = pn.abs().topk(16, dim=-1)
        neur_top[k] = (ti.cpu(), pn.gather(1, ti).cpu())
        del s, c
    t2 = time.perf_counter()
    if timing is not None:
        timing.update(A2_all_sites_and_vjp=t1 - t0, A2_blocks=t2 - t1)
    rows_out = []
    for i in range(b):
        Cs = torch.stack([C[(X, k)][i] for k in blocks for X in SITES])  # [4K, d]
        Cb = Cs.view(len(blocks), 4, D).sum(1)  # [K, d]
        Q = Qf[i]
        Ql = lg(Q[None])[0]
        Cl = lg(Cs)
        qh = qhat[i]
        p_site = Cs @ qh
        p_n1, p_n2 = p_site.view(-1, 4)[:, 0], p_site.view(-1, 4)[:, 2]
        p_heads = torch.cat([heads[k][i] for k in blocks])
        p_neur = torch.cat([neur_p[k][i] for k in blocks])
        fine = torch.cat([p_n1, p_n2, p_heads, p_neur])
        r = {"Q_over_J": float(Q.norm() / Jf[i].norm()),
             "site": vec_stats(Cs, Q), "block": vec_stats(Cb, Q),
             "site_logit": vec_stats(Cl, Ql), "block_logit": vec_stats(Cl.view(len(blocks), 4, -1).sum(1), Ql),
             "type_share": {X: float(Cs.view(-1, 4, D)[:, j].sum(0) @ Q / Q.norm() ** 2) for j, X in enumerate(SITES)},
             "type_share_logit": {X: float(Cl.view(len(blocks), 4, -1)[:, j].sum(0) @ Ql / Ql.norm() ** 2)
                                  for j, X in enumerate(SITES)},
             "block_share": [float(x) for x in Cb @ Q / Q.norm() ** 2],
             "block_norm": [float(x) for x in Cb.norm(dim=-1) / Q.norm()],
             "scal_block": scal_stats(Cb @ qh), "scal_site": scal_stats(p_site), "scal_fine": scal_stats(fine),
             "scal_heads": scal_stats(p_heads), "scal_neurons": scal_stats(p_neur),
             "neuron_block_pr": [float(neur_p[k][i].abs().sum() ** 2 / neur_p[k][i].pow(2).sum()) for k in blocks],
             "head_block_pr": [float(heads[k][i].abs().sum() ** 2 / heads[k][i].pow(2).sum().clamp_min(1e-30))
                               for k in blocks]}
        rows_out.append(r)
        del Cl
    raw = {"blocks": blocks, "C": torch.stack([torch.stack([C[(X, k)] for X in SITES], 1) for k in blocks], 1).cpu(),
           "Q": Qf.cpu(), "J": Jf.cpu(), "heads": [heads[k].cpu() for k in blocks],  # head count differs by layer type
           "neur_top_idx": torch.stack([neur_top[k][0] for k in blocks], 1),
           "neur_top_val": torch.stack([neur_top[k][1] for k in blocks], 1)}
    checks = {"resid_structure": rc, "head_sum_vs_jvp": chk_h, "neuron_sum_vs_jvp": chk_n}
    return rows_out, raw, checks


def summarize2(rows_):
    def g(path):
        vals = []
        for r in rows_:
            v = r
            for p in path:
                v = v.get(p) if isinstance(v, dict) else (v[p] if v is not None and p < len(v) else None)
                if v is None:
                    break
            if v is not None:
                vals.append(v)
        return med(vals)
    s = {"n": len(rows_), "Q_over_J": g(["Q_over_J"])}
    for lvl in ("site", "block", "site_logit", "block_logit"):
        s[lvl] = {k: g([lvl, k]) for k in ("gate", "pr", "ci", "top1_norm_share", "n50", "n90", "best_single_err")}
        s[lvl]["err_top1"] = g([lvl, "err_top", 0])
    for lvl in ("scal_block", "scal_site", "scal_fine", "scal_heads", "scal_neurons"):
        s[lvl] = {k: g([lvl, k]) for k in ("pr", "ci", "n50", "n90", "N")}
        s[lvl]["top1_share"] = g([lvl, "top_share", 1])
        s[lvl]["err_top1"] = g([lvl, "err_top", 1])
        s[lvl]["err_top10"] = g([lvl, "err_top", 10])
    s["type_share"] = {X: g(["type_share", X]) for X in SITES}
    s["type_share_logit"] = {X: g(["type_share_logit", X]) for X in SITES}
    K = len(rows_[0]["block_share"])
    s["block_share"] = [g(["block_share", i]) for i in range(K)]
    s["block_norm"] = [g(["block_norm", i]) for i in range(K)]
    return s


# ---------------------------------------------------------------- A3: third order, self vs cross (and block pairs)
def anatomy3(R, l, U, alphas, pair, timing=None):
    b = U.shape[0]
    A1 = R.spread(U)
    z = torch.zeros_like(A1)
    t0 = time.perf_counter()
    T1, T2 = all_sites(R, l, A1)
    Jf, Qf = T1[("L", None)], T2[("L", None)]
    tin, qin = site_inputs(l, A1, T1), site_inputs(l, z, T2)
    del T1, T2
    fL = lambda d: R.run(b, l + 1, {("in", l + 1): d}, [("L", None)], stop=False)  # noqa: E731
    _, (Tf,) = d3(fL, (z,), (A1,))
    that = Tf / Tf.norm(dim=-1, keepdim=True)
    blocks = list(range(l + 1, L + 1))
    allsites = [(X, k) for k in blocks for X in SITES]
    g3 = cotangents(R, b, l + 1, allsites, that) if pair else None
    t1 = time.perf_counter()
    c3s, c3x, s2 = {}, {}, {}
    for k in blocks:
        selfT = block_d3(R, b, k, tin)
        cross = block_d2(R, b, k, tin, qin)
        if pair:
            src2 = block_d2(R, b, k, tin)
            for X in SITES:
                s2[(X, k)] = src2[X]
        c = push_down(R, k, {X: torch.cat([selfT[X], 3 * cross[X]]) for X in SITES})
        for X in SITES:
            c3s[(X, k)], c3x[(X, k)] = c[X][:b], c[X][b:]
        del selfT, cross, c
    t2 = time.perf_counter()
    M = None
    if pair:
        K = len(blocks)
        M = torch.zeros(b, K, K)
        for jj, j in enumerate(blocks):
            rec = [(X, k) for k in range(j, L + 1) for X in ("n1", "mix", "n2")] + [("out", k) for k in range(j, L)]
            fn = lambda *ds, j=j, rec=rec: R.run(b, j, dict(zip([(X, j) for X in SITES], ds)), rec, stop=True)  # noqa: E731
            sj = tuple(s2[(X, j)] for X in SITES)
            _, qj = jvp(fn, tuple(torch.zeros_like(x) for x in sj), sj)
            qj = dict(zip(rec, qj))
            for k in range(j, L + 1):
                hin = torch.zeros_like(A1) if k == j else qj[("out", k - 1)]
                qk = {("n1", k): hin, ("mix", k): qj[("n1", k)], ("n2", k): hin + qj[("mix", k)], ("mlp", k): qj[("n2", k)]}
                mixed = block_d2(R, b, k, tin, qk)
                M[:, jj, k - l - 1] = sum(3 * (g3[(X, k)] * mixed[X]).sum((1, 2)) for X in SITES).cpu()
            del qj
    t3 = time.perf_counter()
    if timing is not None:
        timing.update(A3_all_sites_T_vjp=t1 - t0, A3_blocks=t2 - t1, A3_pairs=t3 - t2)
    # large steps: actual F(alpha U) - F(0) against Taylor orders (fp32 without TF32 for the differences)
    with no_tf32():
        F0 = forward_F(R, l, z)
        act = {al: forward_F(R, l, al * A1) - F0 for al in alphas}
    Ts = torch.stack([torch.stack([c3s[s][i] for s in allsites]).sum(0) for i in range(b)])
    Tx = torch.stack([torch.stack([c3x[s][i] for s in allsites]).sum(0) for i in range(b)])
    rows_out = []
    for i in range(b):
        Cs_ = torch.stack([c3s[s][i] for s in allsites]); Cx_ = torch.stack([c3x[s][i] for s in allsites])
        T = Tf[i]; tn = T.norm()
        r = {"T_over_J": float(tn / Jf[i].norm()), "T_over_Q": float(tn / Qf[i].norm()),
             "gate": float((Ts[i] + Tx[i] - T).norm() / tn),
             "self_share": float(Ts[i] @ T / tn ** 2), "cross_share": float(Tx[i] @ T / tn ** 2),
             "self_norm": float(Ts[i].norm() / tn), "cross_norm": float(Tx[i].norm() / tn),
             "cos_self_cross": float(torch.nn.functional.cosine_similarity(Ts[i], Tx[i], dim=0)),
             "site_self": vec_stats(Cs_, Ts[i]), "site_cross": vec_stats(Cx_, Tx[i]),
             "site_all": vec_stats(torch.cat([Cs_, Cx_]), T),
             "large_step": {}}
        for al in alphas:
            a_ = act[al][i]
            preds = {"J": al * Jf[i], "JQ": al * Jf[i] + al ** 2 / 2 * Qf[i],
                     "JQT": al * Jf[i] + al ** 2 / 2 * Qf[i] + al ** 3 / 6 * T,
                     "JQT_self": al * Jf[i] + al ** 2 / 2 * Qf[i] + al ** 3 / 6 * Ts[i]}
            la = lg(a_[None])[0]
            r["large_step"][al] = {k_: [float((a_ - p).norm() / a_.norm()),
                                        float((la - lg(p[None])[0]).norm() / la.norm())] for k_, p in preds.items()}
        if pair:
            Mi = M[i]
            tot = float(Mi.sum()) + float(sum(c3s[s][i] @ that[i] for s in allsites))
            r["pair"] = {"gate": abs(tot - float(tn)) / float(tn),
                         "cross_jk_share": float(torch.triu(Mi, 1).sum() / tn),
                         "within_block_share": float(Mi.diagonal().sum() / tn),
                         "self_share_t": float(sum(c3s[s][i] @ that[i] for s in allsites) / tn),
                         "by_distance": [float(Mi.diagonal(dd).sum() / tn) for dd in range(Mi.shape[0])],
                         "pair_scal": scal_stats(Mi[torch.triu(torch.ones_like(Mi), 0).bool()])}
        rows_out.append(r)
    raw = {"blocks": blocks, "J": Jf.cpu(), "Q": Qf.cpu(), "T": Tf.cpu(),
           "C3_self": torch.stack([torch.stack([c3s[(X, k)] for X in SITES], 1) for k in blocks], 1).cpu(),
           "C3_cross": torch.stack([torch.stack([c3x[(X, k)] for X in SITES], 1) for k in blocks], 1).cpu(),
           "M": M, "act": {al: v.cpu() for al, v in act.items()}}
    return rows_out, raw


def summarize3(rows_, alphas):
    s = {"n": len(rows_)}
    for k in ("T_over_J", "T_over_Q", "gate", "self_share", "cross_share", "self_norm", "cross_norm", "cos_self_cross"):
        s[k] = med([r[k] for r in rows_])
    for lvl in ("site_self", "site_cross", "site_all"):
        s[lvl] = {k: med([r[lvl][k] for r in rows_]) for k in ("pr", "ci", "n50", "n90", "best_single_err")}
    s["large_step"] = {al: {k: [med([r["large_step"][al][k][j] for r in rows_]) for j in (0, 1)]
                            for k in ("J", "JQ", "JQT", "JQT_self")} for al in alphas}
    if "pair" in rows_[0]:
        s["pair"] = {k: med([r["pair"][k] for r in rows_]) for k in
                     ("gate", "cross_jk_share", "within_block_share", "self_share_t")}
        nd = len(rows_[0]["pair"]["by_distance"])
        s["pair"]["by_distance"] = [med([r["pair"]["by_distance"][d] for r in rows_]) for d in range(nd)]
        s["pair"]["scal"] = {k: med([r["pair"]["pair_scal"][k] for r in rows_]) for k in ("pr", "ci", "n50", "n90", "N")}
    return s


ALPHAS = [float(x) for x in args.alphas.split(",")]

# ---------------------------------------------------------------- stage: check
if "check" in stages:
    l = SRC_CHK
    st, dirs = directions([l], max(args.check_n_dir, 8))
    U = dirs[l]["iso"][:args.check_n_dir] * st[l]["rho"]
    R = Runner(ctx_ids[0])
    res = {"source": l, "rho": st[l]["rho"], "K": L - l}
    # 1) the skip/stop runner reproduces the plain full-model nested jvp (so_core, as in D1)
    A1 = R.spread(U)
    T1, T2 = all_sites(R, l, A1)
    f = Perturbed(m, ctx_ids[0].expand(len(U), -1), l, L)
    ja, _, q = first_and_second(f, U, None, signs=False)
    nV = len(f.valid)
    res["runner_vs_full_J"] = float((T1[("L", None)] - ja * nV).norm() / (ja * nV).norm())
    res["runner_vs_full_Q"] = float((T2[("L", None)] - q * nV).norm() / (q * nV).norm())
    res["resid_structure"] = resid_check(l, A1, T1)
    # 2) Q and T against finite differences (fp32, no TF32), two step sizes
    with no_tf32():
        Fz = forward_F(R, l, torch.zeros_like(A1))
        Fh = {h: forward_F(R, l, h * A1) for h in (-0.2, -0.1, -0.05, 0.05, 0.1, 0.2)}
        fL = lambda d: R.run(len(U), l + 1, {("in", l + 1): d}, [("L", None)], stop=False)  # noqa: E731
        _, (Tf,) = d3(fL, (torch.zeros_like(A1),), (A1,))
    Qf = T2[("L", None)]
    for h in (0.05, 0.1):
        qfd = (Fh[h] - 2 * Fz + Fh[-h]) / h ** 2
        tfd = (Fh[2 * h] - 2 * Fh[h] + 2 * Fh[-h] - Fh[-2 * h]) / (2 * h ** 3)
        res[f"Q_vs_fd_h{h}"] = float((Qf - qfd).norm() / Qf.norm())
        res[f"T_vs_fd_h{h}"] = float((Tf - tfd).norm() / Tf.norm())
    del T1, T2
    # 3) the decomposition gates, TF32 off and on, with timing
    for tf in (0, 1):
        torch.backends.cuda.matmul.allow_tf32 = bool(tf)
        tm = {}
        t0 = time.perf_counter()
        r2, _, ck = anatomy2(R, l, U, tm)
        tm["A2_total"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        r3, _ = anatomy3(R, l, U, ALPHAS, pair=True, timing=tm)
        tm["A3_total"] = time.perf_counter() - t0
        res[f"tf32={tf}"] = {"A2_gate_site": max(r["site"]["gate"] for r in r2),
                             "A2_gate_logit": max(r["site_logit"]["gate"] for r in r2), **ck,
                             "A3_gate": max(r["gate"] for r in r3), "A3_pair_gate": max(r["pair"]["gate"] for r in r3),
                             "timing_s": {k: round(v, 2) for k, v in tm.items()},
                             "A2_summary": summarize2(r2), "A3_summary": summarize3(r3, ALPHAS)}
        log(f"check tf32={tf}", json.dumps({k: v for k, v in res[f"tf32={tf}"].items()
                                             if k not in ("A2_summary", "A3_summary")}))
    torch.backends.cuda.matmul.allow_tf32 = bool(args.tf32)
    res["peak_mem_gb"] = round(mem_gb(peak=True), 1)
    # gate: dependent bulk jobs (afterok) run only if the decomposition is exact; pick TF32 for them
    ok = {}
    for tf in (0, 1):
        r = res[f"tf32={tf}"]
        ok[tf] = (r["A2_gate_site"] < 1e-2 and r["A3_gate"] < 5e-2 and r["A3_pair_gate"] < 5e-2
                  and r["head_sum_vs_jvp"] < 1e-2 and r["neuron_sum_vs_jvp"] < 1e-2 and r["resid_structure"] < 1e-4)
    res["gates_ok"] = {str(k): v for k, v in ok.items()}
    res["runner_ok"] = res["runner_vs_full_Q"] < 1e-2 and res["runner_vs_full_J"] < 1e-3
    choice = 1 if ok[1] else (0 if ok[0] else None)
    res["tf32_choice"] = choice
    log("check", json.dumps({k: v for k, v in res.items() if not k.startswith("tf32=")}))
    dump("check", res)
    if choice is None or not res["runner_ok"]:
        log("CHECK FAILED: gates not met, dependent jobs must not run")
        sys.exit(3)
    (out / "tf32_choice").write_text(str(choice))

# ---------------------------------------------------------------- stage: A2
if "A2" in stages:
    t00 = time.perf_counter()
    st, dirs = directions(SRC_A2, args.n_dir)
    types = args.types.split(",")
    res = {"n_ctx": args.n_ctx, "n_dir": args.n_dir, "n_layers": m.n_layers, "tf32": args.tf32, "sources": {}}
    raw = {}
    for l in sorted(SRC_A2, reverse=True):  # short spans first: a timeout keeps the cheap sources
        t0 = time.perf_counter()
        res["sources"][l] = {"rho": st[l]["rho"], "nominal": round(l * 64 / m.n_layers), "blocks": list(range(l + 1, L + 1))}
        raw[l] = {}
        for tn in types:
            rows_all, raws, checks = [], [], []
            for ci in range(args.n_ctx):
                R = Runner(ctx_ids[ci])
                r_, w_, c_ = anatomy2(R, l, dirs[l][tn] * st[l]["rho"])
                rows_all += r_; raws.append(w_); checks.append(c_)
                del R
            res["sources"][l][tn] = {"summary": summarize2(rows_all),
                                     "checks": {k: max(c[k] for c in checks) for k in checks[0]}, "rows": rows_all}
            raw[l][tn] = {k: (torch.stack([w[k] for w in raws]) if torch.is_tensor(raws[0][k]) else [w[k] for w in raws])
                          for k in raws[0] if k != "blocks"}
            sm = res["sources"][l][tn]["summary"]
            log(f"A2 L{l} {tn} QJ={sm['Q_over_J']:.2f} gate={sm['site']['gate']:.1e} site pr={sm['site']['pr']:.1f} "
                f"ci={sm['site']['ci']:.2f} err1={sm['site']['err_top1']:.2f} n90={sm['site']['n90']} | block pr="
                f"{sm['block']['pr']:.1f} ci={sm['block']['ci']:.2f} n90={sm['block']['n90']} | fine pr="
                f"{sm['scal_fine']['pr']:.0f} n90={sm['scal_fine']['n90']} | heads n90={sm['scal_heads']['n90']} "
                f"neur n90={sm['scal_neurons']['n90']} | types {json.dumps({k: round(v, 2) for k, v in sm['type_share'].items()})}")
        res["sources"][l]["secs"] = round(time.perf_counter() - t0, 1)
        dump("A2", res)
        if args.save_raw:
            torch.save(raw, out / f"A2{args.name}_raw.pt")
    res["secs"] = round(time.perf_counter() - t00, 1)
    res["defs"] = ("Push U = rho_l*unit(dir) at every valid position of block l; F = sum over valid positions of block L "
                   "(pre-norm). c_X = J_{X->L} D^2 f_X[tau_X,tau_X] per site X in {n1,mix,n2,mlp} of each block; "
                   "sum_X c_X = Q_F(U,U) (gate = rel. error). vec_stats over components: pr = (sum|c|)^2/sum|c|^2 "
                   "(effective number), ci = sum|c|/|sum c| (cancellation), err_top = |Q - sum of the n largest|/|Q|, "
                   "n50/n90 = components (largest first) needed to stay within 50%/10% of Q, best_single_err = "
                   "min_X |Q - c_X|/|Q|. *_logit: same on centred logits (W_U gamma). scal_*: scalar contributions "
                   "along q_hat = Q/|Q|; fine = n1 + n2 sites + every attention head + every MLP neuron. "
                   "type_share = <sum of a site type, Q>/|Q|^2. block_share/norm per block l+1..L.")
    dump("A2", res)

# ---------------------------------------------------------------- stage: A3
if "A3" in stages:
    t00 = time.perf_counter()
    st, dirs = directions(SRC_A3, args.n_dir)
    res = {"n_ctx": args.a3_n_ctx, "n_dir": args.n_dir, "alphas": ALPHAS, "pair": args.a3_pair, "tf32": args.tf32,
           "sources": {}}
    raw = {}
    for l in sorted(SRC_A3, reverse=True):
        t0 = time.perf_counter()
        res["sources"][l] = {"rho": st[l]["rho"], "nominal": round(l * 64 / m.n_layers), "blocks": list(range(l + 1, L + 1))}
        raw[l] = {}
        for tn in args.a3_types.split(","):
            rows_all, raws = [], []
            for ci in range(args.a3_n_ctx):
                R = Runner(ctx_ids[ci])
                r_, w_ = anatomy3(R, l, dirs[l][tn] * st[l]["rho"], ALPHAS, bool(args.a3_pair))
                rows_all += r_; raws.append(w_)
                del R
            res["sources"][l][tn] = {"summary": summarize3(rows_all, ALPHAS), "rows": rows_all}
            raw[l][tn] = raws
            sm = res["sources"][l][tn]["summary"]
            msg = (f"A3 L{l} {tn} T/Q={sm['T_over_Q']:.2f} gate={sm['gate']:.1e} self={sm['self_share']:.2f} "
                   f"cross={sm['cross_share']:.2f}")
            if "pair" in sm:
                msg += (f" pair: cross-block={sm['pair']['cross_jk_share']:.2f} within={sm['pair']['within_block_share']:.2f}"
                        f" gate={sm['pair']['gate']:.1e}")
            al = ALPHAS[-1]
            msg += " | a=%g J/JQ/JQT/JQT_self=%s" % (al, [round(sm["large_step"][al][k][0], 3)
                                                     for k in ("J", "JQ", "JQT", "JQT_self")])
            log(msg)
        res["sources"][l]["secs"] = round(time.perf_counter() - t0, 1)
        dump("A3", res)
        if args.save_raw:
            torch.save(raw, out / f"A3{args.name}_raw.pt")
    res["secs"] = round(time.perf_counter() - t00, 1)
    res["defs"] = ("T = D^3 F[U,U,U] (triple-nested jvp). self = sum_X J D^3 f_X[tau,tau,tau]; cross = sum_X J 3 D^2 "
                   "f_X[tau_X, q_X] (q_X = second-order tangent at X's input: curvature made upstream). gate = "
                   "|self+cross-T|/|T|; *_share = <part, T>/|T|^2. pair: M[j,k] = 3 sum_{X in block k} <J^T t_hat, "
                   "D^2 f_X[tau_X, q_X^(j)]>, q^(j) = the part of q made by block j; cross_jk_share = sum_{j<k} M / |T|, "
                   "within_block_share = sum_j M[j,j] / |T|, self_share_t = self along t_hat; by_distance[d] = "
                   "sum_{k-j=d} M / |T|. large_step[alpha][pred] = [residual rel. error, centred-logit rel. error] of "
                   "F(alpha U)-F(0) against J, J+Q/2, J+Q/2+T/6, J+Q/2+T_self/6.")
    dump("A3", res)

log("DONE", f"peak_mem={mem_gb(peak=True):.1f} GB")
