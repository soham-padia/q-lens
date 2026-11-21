"""Loop lens: first- and second-order ("double Jacobian") transport in looped (weight-tied, recurrent-depth) LMs.
Stages L0, L1a, L1b.

Models (transformers 4.55, trust_remote_code, pinned revisions; run in a separate env, see requirements-loop.txt):
  Huginn-0125   RavenForCausalLM: wte*emb_scale -> 2 prelude blocks (= e) -> r x [adapter(cat[x, e]) -> 4 core
                SandwichBlocks] from a RANDOM state x0 -> ln_f -> 2 coda blocks -> ln_f -> lm_head.
  Ouro-1.4B/2.6B OuroForCausalLM: embed -> 4 x [NL decoder layers -> model.norm] -> lm_head (the normed state feeds
                the next loop; we always exit at the last loop, i.e. the model's exit_at_step = n_loops - 1).

Everything runs through an explicit forward built from the models' submodules (LoopModel.run), so push and read sites
are addressed directly, with no hooks on reused modules:
  site (t, i), 0 <= i < NL  output of loop layer i in loop t (0-based; Ouro: pre-norm, Huginn: core block output)
  site (t, NL)              output state of loop t = h_{t+1} (Ouro: after model.norm; Huginn: same tensor as (t, NL-1))
  site (t, -1)              input of loop t (only as a start site; (0, -1) holds x0: Huginn random, Ouro embeddings)
  site ("coda", t)          Huginn: ln_f -> coda applied to h_{t+1}, pre-final-norm = the model's final block output
                            had it stopped after t+1 recurrences (D1's target L)
  block index g             t*NL + i (loop output t*NL + NL-1; coda t*NL + NL-1 + n_coda); k = g_target - g_source.
Huginn's x0 is drawn from a seeded CPU generator with the model's own distribution (initialize_state: trunc-normal,
std = init_values["std"], times emb_scale) and passed as input_states when we compare with the model's forward.

L1a/L1b metrics are D1's (src/second_order.py) exactly: steer alpha*rho*unit(d) at every valid position (16..T-2) of
the source site, F = sum over valid positions of the target state; J d and Q(d, d) from so_core.first_and_second
(nested forward-mode jvp, signs=False, i.e. steer at every position, cross-position terms included); actual change by
finite steps; rel_lin = |act - a J d| / |act|, rel_quad with + a^2 Q(d,d)/2, Q_over_J = |Q(d,d)| / |J d| (alpha = 1);
medians over contexts x directions; directions iso (seeded randn) and data (centred states at one random valid position
of contexts disjoint from the steered ones); rho = mean |state| over valid positions of 8 contexts.

Outputs (<out>/<stage>.json, written after every unit of work; tensors in <stage>_raw.pt):
  L0   gate: explicit forward == model forward (fp32, max |dlogit| <= 1e-4); nested-jvp Q vs central finite
       differences (cos >= 0.995, D1 setting); site/counter checks; Huginn
       seed reproducibility and convergence |h_{r+1}-h_r|/|h_r| < tol at r = 32. Exits nonzero on failure.
  L1a  transport across loops: Huginn push at h_t (t in 1,4,16), read h_{t'} for every later t' <= r and the coda;
       Ouro push in loop 1 at layers {1/4, 1/2, 3/4} NL, read every 8th layer, at every loop end (pre- and post-norm).
  L1b  Huginn, same weights, variable depth: push at h_2, read h_r and coda@r for r in 4..64, per position, split by
       convergence at r.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
from torch.func import jvp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from so_core import Perturbed, first_and_second, math_sdpa, no_tf32  # noqa: E402  (puts jacobian-lens on sys.path)
from jlens.fitting import valid_position_mask  # noqa: E402

REVS = {"tomg-group-umd/huginn-0125": "bb6621b65e90b6a4b9b29ef88dc83866d450470c",
        "ByteDance/Ouro-1.4B": "574fa66cb8bf5abdc979642d01cf2b79b16bfab1",
        "ByteDance/Ouro-2.6B": "1ed04250da1a9936042725d302e81c8fa2ab5abd"}
WORK = os.environ.get("QLENS_OUT", "outputs")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
OUT: Path = Path(".")
DTYPE = torch.float32


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def dump(name, obj):
    (OUT / f"{name}.json").write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=str))
    log(f"wrote {OUT / (name + '.json')}")


def short_name(model: str) -> str:
    m = model.lower()
    if "huginn" in m:
        return "huginn"
    if "ouro-" in m:
        return "ouro" + m.split("ouro-")[-1].replace(".", "").rstrip("b")
    return m.split("/")[-1]


def r5(x):
    x = float(x)
    return None if not math.isfinite(x) else round(x, 5)


def med(x):
    x = torch.as_tensor(x).float().flatten()
    x = x[torch.isfinite(x)]
    return float(x.median()) if len(x) else float("nan")


# ====================================================================== explicit forwards
class LoopModel:
    """Explicit forward over sites (t, i). Subclasses define NL, n_coda, d, W_U, default_r and
    prep / _layer / loop_end / final_hidden / head_from_read / logit_site / reference_logits / gamma_for."""
    kind = ""

    def __init__(self, hf):
        self.hf = hf
        self.n_layer_calls = 0  # counter: every application of a (reused) loop layer

    def layer(self, x, ctx, j):
        self.n_layer_calls += 1
        return self._layer(x, ctx, j)

    def g(self, site) -> int:
        if site[0] == "coda":
            return site[1] * self.NL + self.NL - 1 + self.n_coda
        t, i = site
        return t * self.NL + min(i, self.NL - 1)

    def order(self, site):
        return (site[1], self.NL + 0.5) if site[0] == "coda" else (site[0], site[1])

    def label(self, site) -> str:
        if site[0] == "coda":
            return f"coda@r{site[1] + 1}"
        t, i = site
        if self.kind == "huginn":
            return f"h{t + 1}" if i in (self.NL - 1, self.NL) else (f"h{t}" if i == -1 else f"r{t + 1}.b{i}")
        return f"out{t + 1}" if i == self.NL else (f"in{t + 1}" if i == -1 else f"loop{t + 1}.L{i}")

    def run(self, ctx, x, start, n_loops, reads=(), steer=None, reduce=None):
        """x is the state at site `start` ((t0, i0); i0 = -1: input of loop t0; i0 = NL: output of loop t0).
        Runs layer by layer to the end of loop n_loops-1, or until the last site in `reads` has been recorded.
        steer (broadcastable to the state) is added to every loop output (after loop_end). reduce is applied to every
        recorded state. Returns ({site: reduce(state)}, x at the stopping point)."""
        reads = set(reads)
        last = max(reads, key=self.order) if reads else None
        red = (lambda h: h) if reduce is None else reduce
        out = {}
        t, i = start
        if i == self.NL:
            t, i = t + 1, -1
        while t < n_loops:
            for j in range(i + 1, self.NL):
                x = self.layer(x, ctx, j)
                if (t, j) in reads:
                    out[(t, j)] = red(x)
                    if (t, j) == last:
                        return out, x
            x = self.loop_end(x)
            if steer is not None:
                x = x + steer
            if (t, self.NL) in reads:
                out[(t, self.NL)] = red(x)
            if ("coda", t) in reads:
                out[("coda", t)] = red(self.final_hidden(x, ctx))
            if last in ((t, self.NL), ("coda", t)):
                return out, x
            t, i = t + 1, -1
        return out, x

    def logits_from_state(self, x, ctx):
        """Full logits [B, T, V] from a loop-output state (all positions)."""
        site = self.logit_site(0)
        y = self.final_hidden(x, ctx) if site[0] == "coda" else x
        return self.head_from_read(y)


class Huginn(LoopModel):
    kind = "huginn"

    def __init__(self, hf):
        super().__init__(hf)
        cfg = hf.config
        self.cfg, self.tr = cfg, hf.transformer
        self.NL, self.n_coda, self.d = len(self.tr.core_block), len(self.tr.coda), cfg.n_embd
        self.emb_scale = float(hf.emb_scale)
        self.std = float(cfg.init_values["std"])
        self.W_U = hf.lm_head.weight
        self.default_r = int(cfg.mean_recurrence)
        self.test_time_noise = float(getattr(cfg, "test_time_noise", 0) or 0)

    def init_state(self, T, seeds, device):
        """Seeded replica of RavenForCausalLM.initialize_state (scale 1): trunc-normal(0, std, +-3 std) * emb_scale."""
        xs = []
        for s in seeds:
            x = torch.empty(T, self.d, dtype=torch.float32)
            if self.std > 0:
                torch.nn.init.trunc_normal_(x, 0.0, self.std, -3 * self.std, 3 * self.std,
                                            generator=torch.Generator().manual_seed(int(s)))
                if self.emb_scale != 1:
                    x = x * self.emb_scale
            else:
                x.zero_()
            xs.append(x)
        return torch.stack(xs).to(device=device, dtype=DTYPE)

    def prep(self, ids, seeds):
        T = ids.shape[1]
        freqs = self.hf.freqs_cis[:, :T]
        e = self.tr.wte(ids)
        if self.emb_scale != 1:
            e = e * self.emb_scale
        for blk in self.tr.prelude:
            e = blk(e, freqs, 0, None, None)
        return {"T": T, "freqs": freqs, "e": e, "x0": self.init_state(T, seeds, ids.device), "seeds": list(seeds)}

    def _layer(self, x, ctx, j):
        if j == 0:  # one recurrence: x <- core(adapter(cat[x, e]))   (test_time_noise = 0, checked in L0)
            e = ctx["e"]
            if e.shape[0] != x.shape[0]:
                e = e.expand(x.shape[0], -1, -1)
            x = self.tr.adapter(torch.cat([x, e], dim=-1))
        return self.tr.core_block[j](x, ctx["freqs"], 0, None, None)

    def loop_end(self, x):
        return x

    def final_hidden(self, x, ctx):
        y = self.tr.ln_f(x)
        for blk in self.tr.coda:
            y = blk(y, ctx["freqs"], 0, None, None)
        return y

    def logit_site(self, t):
        return ("coda", t)

    def head_from_read(self, y):
        return self.hf.lm_head(self.tr.ln_f(y)).float()

    def gamma_for(self, site):  # every Huginn read goes through ln_f to the head (logit-lens for h_t)
        return self.tr.ln_f.weight

    def reference_logits(self, ids, ctx, n_loops):
        x0 = ctx["x0"]
        if x0.shape[0] != ids.shape[0]:
            x0 = x0.expand(ids.shape[0], -1, -1)
        return self.hf(input_ids=ids, input_states=x0.clone(), num_steps=(int(n_loops), 0), use_cache=False).logits.float()


def ouro_rms(mod, x):
    """OuroRMSNorm.forward, verbatim arithmetic."""
    dt = x.dtype
    h = x.to(torch.float32)
    h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + mod.variance_epsilon)
    return mod.weight * h.to(dt)


class Ouro(LoopModel):
    kind = "ouro"

    def __init__(self, hf):
        super().__init__(hf)
        cfg = hf.config
        self.cfg = cfg
        self.layers = hf.model.layers[: cfg.num_hidden_layers]
        self.NL, self.n_coda, self.d = len(self.layers), 0, cfg.hidden_size
        self.norm = hf.model.norm
        self.W_U = hf.lm_head.weight
        self.default_r = int(getattr(cfg, "total_ut_steps", 4))
        rc = sys.modules[type(self.layers[0]).__module__]  # the remote modeling_ouro module
        self._rope, self._repeat_kv = rc.apply_rotary_pos_emb, rc.repeat_kv

    def prep(self, ids, seeds=None):
        T = ids.shape[1]
        x0 = self.hf.model.embed_tokens(ids)
        if ids.is_meta:  # meta check only: the rotary's autocast(device_type="meta") is unsupported; shapes suffice
            cos = sin = torch.empty(1, T, self.layers[0].self_attn.head_dim, device="meta")
        else:
            cos, sin = self.hf.model.rotary_emb(x0, torch.arange(T, device=ids.device)[None])
        return {"T": T, "x0": x0, "cos": cos, "sin": sin}

    def _layer(self, x, ctx, j):  # OuroDecoderLayer.forward with sdpa (no cache, causal)
        L = self.layers[j]
        a = L.self_attn
        B, T, _ = x.shape
        h = ouro_rms(L.input_layernorm, x)
        shp = (B, T, -1, a.head_dim)
        q = a.q_proj(h).view(shp).transpose(1, 2)
        k = a.k_proj(h).view(shp).transpose(1, 2)
        v = a.v_proj(h).view(shp).transpose(1, 2)
        q, k = self._rope(q, k, ctx["cos"], ctx["sin"])
        k, v = self._repeat_kv(k, a.num_key_value_groups), self._repeat_kv(v, a.num_key_value_groups)
        o = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True, scale=a.scaling)
        o = a.o_proj(o.transpose(1, 2).reshape(B, T, -1))
        x = x + ouro_rms(L.input_layernorm_2, o)
        m = L.mlp
        h = ouro_rms(L.post_attention_layernorm, x)
        h = m.down_proj(m.act_fn(m.gate_proj(h)) * m.up_proj(h))
        return x + ouro_rms(L.post_attention_layernorm_2, h)

    def loop_end(self, x):
        return ouro_rms(self.norm, x)

    def final_hidden(self, x, ctx):
        raise ValueError("Ouro has no coda")

    def logit_site(self, t):
        return (t, self.NL)

    def head_from_read(self, y):
        return self.hf.lm_head(y).float()

    def gamma_for(self, site):  # loop outputs are already normed (lm_head reads them directly)
        return None if (site[0] != "coda" and site[1] == self.NL) else self.norm.weight

    def reference_logits(self, ids, ctx, n_loops):
        old = self.hf.model.total_ut_steps
        self.hf.model.total_ut_steps = int(n_loops)
        try:
            return self.hf(input_ids=ids, use_cache=False, exit_at_step=int(n_loops) - 1).logits.float()
        finally:
            self.hf.model.total_ut_steps = old

    def default_exit_logits(self, ids):
        return self.hf(input_ids=ids, use_cache=False).logits.float()


def wrap(hf):
    name = type(hf).__name__
    if name.startswith("Raven"):
        return Huginn(hf)
    if name.startswith("Ouro"):
        return Ouro(hf)
    raise ValueError(f"unsupported architecture {name}")


# ====================================================================== perturbed callables
class LoopF(Perturbed):
    """so_core.Perturbed for the explicit loop forward: delta [B, T, d] is added to the state at site `src` (x_src,
    batch 1, held fixed); returns the targets' states summed over valid positions [B, nT, d] (per_pos: [B, nT, nV, d])."""

    def __init__(self, lm, ctx, x_src, src, targets, n_loops, per_pos=False):  # noqa: D107 (no super().__init__)
        self.lm, self.ctx, self.x_src, self.src, self.n_loops, self.per_pos = lm, ctx, x_src, src, n_loops, per_pos
        self.targets = sorted(set(targets), key=lm.order)
        self.T = ctx["T"]
        self.reduce = "valid_sum"
        self.valid = valid_position_mask(self.T).nonzero(as_tuple=True)[0].to(x_src.device)

    def __call__(self, delta):
        V = self.valid
        red = (lambda h: h[:, V]) if self.per_pos else (lambda h: h[:, V].sum(1))  # noqa: E731
        out, _ = self.lm.run(self.ctx, self.x_src + delta, self.src, self.n_loops, reads=self.targets, reduce=red)
        return torch.stack([out[t] for t in self.targets], 1)


def F_of(f, delta):
    with math_sdpa(), torch.no_grad():
        return f(delta).float()


def readout(y, gamma, W_U):
    y = y.float() if gamma is None else y.float() * gamma.float()
    return y @ W_U.float().T


def nested_Q(f, x0, A, B=None):
    """Exact second derivative D^2 f(x0)[A, B] (B = A by default) by nested forward-mode jvp."""
    B = A if B is None else B
    with math_sdpa():
        return jvp(lambda z: jvp(f, (z,), (A,))[1], (x0,), (B,))[1]


# ====================================================================== shared helpers
@torch.no_grad()
def states_at(lm, ctx, sites, reduce=None):
    """Primal states at `sites` for one context, from x0."""
    sites = sorted(set(sites), key=lm.order)
    n = max((s[1] if s[0] == "coda" else s[0]) for s in sites) + 1
    out, _ = lm.run(ctx, ctx["x0"], (0, -1), n, reads=sites, reduce=reduce)
    return out


@torch.no_grad()
def conv_curve(lm, ctx, R, pos):
    """|h_{t+1} - h_t| / |h_t| for t = 1..R at positions `pos`: [R, B, P] (row t-1 is recurrence t)."""
    out, _ = lm.run(ctx, ctx["x0"], (0, -1), R + 1, reads=[(t, lm.NL) for t in range(R + 1)],
                    reduce=lambda h: h[:, pos].float())
    hs = torch.stack([out[(t, lm.NL)] for t in range(R + 1)])
    return ((hs[1:] - hs[:-1]).norm(dim=-1) / hs[:-1].norm(dim=-1).clamp_min(1e-30)).cpu()


class Contexts:
    """Tokenised fineweb-edu contexts (re-tokenised with the model's tokenizer, full max_len only) with cached prep."""

    def __init__(self, lm, tok, args):
        docs = [json.loads(l)["text"] for l in open(args.corpus)]
        self.ids = []
        for t in docs[args.ctx_start: args.ctx_start + args.n_pool]:
            x = tok(t, return_tensors="pt", truncation=True, max_length=args.max_len).input_ids
            if x.shape[1] == args.max_len:
                self.ids.append(x.to(DEV))
        self.lm, self.seed0, self._c = lm, args.init_seed, {}
        log(f"contexts: {len(self.ids)} x {args.max_len} tokens from corpus[{args.ctx_start}:{args.ctx_start + args.n_pool}]")

    def __len__(self):
        return len(self.ids)

    def ctx(self, ci):
        if ci not in self._c:
            with torch.no_grad():
                self._c[ci] = self.lm.prep(self.ids[ci], [self.seed0 + ci])
        return self._c[ci]


def d1_unit(f, U, alphas, lg=None):
    """D1's metric block for one context and a batch of directions U [B, d] (already scaled to rho), copied from
    second_order.py stage D1. Returns CPU tensors: rel_lin, rel_quad, proj (+ lg_* if lg) [B, nA, nT(, nV)],
    norm_J, norm_Q [B, nT(, nV)], and (per_pos) the D1 position-summed versions under key 'sum'."""
    B = U.shape[0]
    ja, _, q = first_and_second(f, U, None, signs=False)
    nV = len(f.valid)
    ja, q = ja.float() * nV, q.float() * nV  # sum over steered sources, as D1
    A1 = f.spread(U, None)
    F0 = F_of(f, torch.zeros_like(A1))
    cs = torch.nn.functional.cosine_similarity
    nA = len(alphas)

    def block(ja_, q_, acts):
        shp = (B, nA) + tuple(ja_.shape[1:-1])
        m = {k: torch.zeros(shp) for k in ("rel_lin", "rel_quad", "proj")}
        for ai, al in enumerate(alphas):
            act = acts[ai]
            lin, quad = al * ja_, al * ja_ + 0.5 * al ** 2 * q_
            an = act.norm(dim=-1).clamp_min(1e-30)
            m["rel_lin"][:, ai] = ((act - lin).norm(dim=-1) / an).cpu()
            m["rel_quad"][:, ai] = ((act - quad).norm(dim=-1) / an).cpu()
            m["proj"][:, ai] = ((act * lin).sum(-1) / lin.pow(2).sum(-1).clamp_min(1e-30)).cpu()
        m["norm_J"], m["norm_Q"] = ja_.norm(dim=-1).cpu(), q_.norm(dim=-1).cpu()
        return m

    acts = [F_of(f, al * A1) - F0 for al in alphas]
    met = block(ja, q, acts)
    if f.per_pos:  # D1's position-summed quantities from the per-position ones (sum is linear)
        met["sum"] = block(ja.sum(2), q.sum(2), [a.sum(2) for a in acts])
    if lg is not None:  # D1's lg_* (W_U gamma readout of the residual-space change), per target
        W_U, gammas = lg
        nT = ja.shape[1]
        for k_ in ("lg_rel_lin", "lg_rel_quad", "lg_cos_lin", "lg_cos_quad"):
            met[k_] = torch.zeros(B, nA, nT)
        for ai, al in enumerate(alphas):
            act = acts[ai]
            lin, quad = al * ja, al * ja + 0.5 * al ** 2 * q
            for ti in range(nT):
                la, ll, lq = (readout(v_[:, ti], gammas[ti], W_U) for v_ in (act, lin, quad))
                lan = la.norm(dim=-1).clamp_min(1e-30)
                met["lg_rel_lin"][:, ai, ti] = ((la - ll).norm(dim=-1) / lan).cpu()
                met["lg_rel_quad"][:, ai, ti] = ((la - lq).norm(dim=-1) / lan).cpu()
                met["lg_cos_lin"][:, ai, ti] = cs(la, ll, dim=-1).cpu()
                met["lg_cos_quad"][:, ai, ti] = cs(la, lq, dim=-1).cpu()
                del la, ll, lq
    return met


def source_stats(lm, C, srcs, n_stat, n_dir):
    """rho and mu at each source site from contexts [0, n_stat) (D1's residual_stats), and D1's data directions:
    centred states at one random valid position (generator seed 4321) of each of the last n_dir contexts."""
    acc = {s: [] for s in srcs}
    for ci in range(min(n_stat, len(C))):
        ctx = C.ctx(ci)
        V = valid_position_mask(ctx["T"]).nonzero(as_tuple=True)[0].to(DEV)
        st = states_at(lm, ctx, srcs, reduce=lambda h: h[0, V].float())
        for s in srcs:
            acc[s].append(st[s])
    stats = {}
    for s in srcs:
        H = torch.cat(acc[s])
        stats[s] = {"mu": H.mean(0), "rho": float(H.norm(dim=-1).mean()),
                    "rho_centered": float((H - H.mean(0)).norm(dim=-1).mean())}
    dgen = torch.Generator().manual_seed(4321)
    data = {s: [] for s in srcs}
    for ci in range(len(C) - n_dir, len(C)):
        ctx = C.ctx(ci)
        V = valid_position_mask(ctx["T"]).nonzero(as_tuple=True)[0]
        tp = int(V[torch.randint(len(V), (1,), generator=dgen)])
        st = states_at(lm, ctx, srcs, reduce=lambda h: h[0, tp].float())
        for s in srcs:
            data[s].append(st[s] - stats[s]["mu"])
    return stats, {s: torch.stack(v) for s, v in data.items()}


def d1_summary(met, ai_list, ti, keys):
    row = {}
    for k_ in keys:
        if k_ in met:
            row[k_] = [r5(met[k_][:, :, ai, ti].median()) for ai in ai_list]
    nJ, nQ = met["norm_J"][:, :, ti], met["norm_Q"][:, :, ti]
    row["Q_over_J"] = r5((nQ / nJ).median())
    row["J_norm"], row["Q_norm"] = r5(nJ.median()), r5(nQ.median())
    return row


D1_KEYS = ("rel_lin", "rel_quad", "proj", "lg_rel_lin", "lg_rel_quad", "lg_cos_lin", "lg_cos_quad")


# ====================================================================== stage L0
def stage_L0(lm, C, args):
    t0 = time.perf_counter()
    NL, hf = lm.NL, lm.hf
    ok, res = {}, {"model": args.model, "revision": args.revision, "kind": lm.kind, "NL": NL, "d": lm.d,
                   "n_coda": lm.n_coda, "default_r": lm.default_r, "tol_logits": args.l0_tol_logits,
                   "tol_fd_cos": args.l0_tol_fd, "conv_tol": args.conv_tol}
    dts = sorted({str(p.dtype) for p in hf.parameters()} | {str(b.dtype) for b in hf.buffers() if b.is_floating_point()})
    res["param_dtypes"] = dts
    ok["fp32"] = dts == ["torch.float32"]
    res["tf32"] = [torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32]
    ok["no_tf32"] = not any(res["tf32"])
    if lm.kind == "huginn":
        res["test_time_noise"] = lm.test_time_noise
        ok["test_time_noise_0"] = lm.test_time_noise == 0
        res["emb_scale"], res["init_std"] = lm.emb_scale, lm.std
    log("L0 config", json.dumps({k: res[k] for k in res if k not in ("model",)}), json.dumps(ok))

    # ---- 1. sites: layer-application counter and restart consistency (the same modules are reused every loop)
    ctx = C.ctx(0)
    n = 3
    allsites = [(t, i) for t in range(n) for i in range(NL)] + [(t, NL) for t in range(n)]
    lm.n_layer_calls = 0
    with torch.no_grad():
        full, _ = lm.run(ctx, ctx["x0"], (0, -1), n, reads=allsites)
    calls = {"full": [lm.n_layer_calls, n * NL]}
    diffs = {}
    for mid in ((1, NL // 2), (0, NL)):
        lm.n_layer_calls = 0
        later = [s for s in allsites if lm.order(s) > lm.order(mid)]
        with torch.no_grad():
            part, _ = lm.run(ctx, full[mid], mid, n, reads=later + [(n - 1, NL)])
        calls[lm.label(mid)] = [lm.n_layer_calls, n * NL - (lm.g(mid) + 1)]
        diffs[lm.label(mid)] = max(float((part[s] - full[s]).abs().max()) for s in later)
    # a push at site s changes nothing before s and changes the next site
    mid = (1, NL // 2)
    dl = torch.randn(1, ctx["T"], lm.d, generator=torch.Generator().manual_seed(3)).to(DEV) * 1e-2
    with torch.no_grad():
        pushed, _ = lm.run(ctx, full[mid] + dl, mid, n, reads=[(1, NL // 2 + 1), (n - 1, NL)])
    push_moves = float((pushed[(1, NL // 2 + 1)] - full[(1, NL // 2 + 1)]).abs().max())
    res["sites"] = {"calls_[got,expected]": calls, "restart_maxdiff": diffs, "push_changes_next_site": push_moves}
    ok["sites"] = all(a == b for a, b in calls.values()) and max(diffs.values()) <= 1e-6 and push_moves > 0
    del full
    log("L0 sites", json.dumps(res["sites"]))

    # ---- 2. explicit forward == model forward (fp32, math SDPA, no TF32)
    lg_res = []
    nr = lm.default_r
    for ci in range(min(2, len(C))):
        ctx = C.ctx(ci)
        with torch.no_grad():
            _, x = lm.run(ctx, ctx["x0"], (0, -1), nr)
            mine = lm.logits_from_state(x, ctx)
            ref = lm.reference_logits(C.ids[ci], ctx, nr)
            d = (mine - ref).abs()
            row = {"ctx": ci, "n_loops": nr, "max_abs": float(d.max()), "mean_abs": float(d.mean()),
                   "max_abs_logit": float(ref.abs().max()), "top1_agree": float((mine.argmax(-1) == ref.argmax(-1)).float().mean())}
            if lm.kind == "ouro":  # model default: early-exit threshold from the config; we always exit at the last loop
                dflt = lm.default_exit_logits(C.ids[ci])
                row["default_exit_vs_last_max_abs"] = float((dflt - ref).abs().max())
                row["default_exit_frac_positions_differ"] = float(((dflt - ref).abs().amax(-1) > 0).float().mean())
        lg_res.append(row)
        del mine, ref
    res["logits_match"] = lg_res
    ok["logits_match"] = all(r_["max_abs"] <= args.l0_tol_logits for r_ in lg_res)
    log("L0 logits", json.dumps(lg_res))

    # ---- 3. nested-jvp Q vs central finite differences
    def fd_report(fun, x0_, A, shape_note):
        zero = torch.zeros_like(A) if x0_ is None else x0_
        with math_sdpa():
            _, JA = jvp(fun, (zero,), (A,))
        Q = nested_Q(fun, zero, A)
        with torch.no_grad():
            F0 = fun(zero).float()
            rows, D2s = {}, {}
            for eps in args.l0_eps:
                Fp, Fm = fun(zero + eps * A).float(), fun(zero - eps * A).float()
                D2 = (Fp - 2 * F0 + Fm) / eps ** 2
                D1 = (Fp - Fm) / (2 * eps)
                D2s[eps] = D2
                c2 = torch.nn.functional.cosine_similarity(D2, Q.float(), dim=-1)
                c1 = torch.nn.functional.cosine_similarity(D1, JA.float(), dim=-1)
                rows[eps] = {"cos_Q": c2.flatten().tolist(), "rel_err_Q": ((D2 - Q).norm(dim=-1) / Q.norm(dim=-1)).flatten().tolist(),
                             "cos_J": c1.flatten().tolist()}
            # Richardson: the central 2nd difference has O(eps^2) truncation error, so (4 D2(e/2) - D2(e)) / 3 is O(eps^4).
            # Deep maps (Huginn: ~112 blocks) have large higher-order terms, so plain FD is truncation-limited at large eps
            # and fp32-roundoff-limited at small eps; extrapolation is the better reference (added 2026-11-19 after the
            # first Huginn smoke 1850865 failed fd_main at best cos 0.9925 while J matched at 0.99999).
            rich = {}
            for e in args.l0_eps:
                h = next((x for x in args.l0_eps if abs(x - e / 2) < 1e-12), None)
                if h is None:
                    continue
                R = (4 * D2s[h] - D2s[e]) / 3
                rich[f"{e}/{h}"] = {"cos_Q": torch.nn.functional.cosine_similarity(R, Q.float(), dim=-1).flatten().tolist(),
                                    "rel_err_Q": ((R - Q).norm(dim=-1) / Q.norm(dim=-1)).flatten().tolist()}
        cols = [rows[e]["cos_Q"] for e in args.l0_eps] + [r_["cos_Q"] for r_ in rich.values()]
        best = torch.tensor(cols).T
        return {"shape": shape_note, "by_eps": rows, "richardson": rich, "best_cos_Q": best.max(1).values.tolist(),
                "best_cos_Q_plain": torch.tensor(cols[:len(args.l0_eps)]).T.max(1).values.tolist(),
                "Q_norm": Q.norm(dim=-1).flatten().tolist(), "J_norm": JA.norm(dim=-1).flatten().tolist()}

    ctx = C.ctx(0)
    if lm.kind == "huginn":
        src, tg, nloop = (3, NL), [(nr - 1, NL), ("coda", nr - 1)], nr  # push h_4, read h_r and the coda output
    else:
        src, tg, nloop = (0, NL // 2), [(nr - 1, NL - 1), (nr - 1, NL)], nr  # push mid loop 1, read the last loop end
    xs = states_at(lm, ctx, [src])[src]
    V = valid_position_mask(ctx["T"]).nonzero(as_tuple=True)[0].to(DEV)
    rho = float(xs[0, V].norm(dim=-1).mean())
    f = LoopF(lm, ctx, xs, src, tg, nloop)
    a = torch.randn(2, lm.d, generator=torch.Generator().manual_seed(7)).to(DEV)
    a = a / a.norm(dim=-1, keepdim=True) * rho
    fd_main = fd_report(f, None, f.spread(a, None), f"push {lm.label(src)} (all valid positions) -> {[lm.label(t) for t in f.targets]}, sum over valid")
    fd_main["rho"] = rho
    log("L0 fd main", json.dumps({"best_cos_Q": fd_main["best_cos_Q"], "Q_norm": fd_main["Q_norm"], "J_norm": fd_main["J_norm"]}))
    res["fd"] = {"main": fd_main}
    ok["fd_main"] = min(fd_main["best_cos_Q"]) >= args.l0_tol_fd
    del xs

    # ---- 4. Huginn: reproducibility under a fixed seed (model's own sampler and ours), and sensitivity to the seed
    if lm.kind == "huginn":
        ids = C.ids[0]
        with torch.no_grad():
            outs = []
            for s in (1234, 1234, 5678):
                torch.manual_seed(s)
                outs.append(hf(input_ids=ids, num_steps=(nr, 0), use_cache=False).logits.float())
            mine = []
            for _ in range(2):
                c_ = lm.prep(ids, [args.init_seed])
                _, x = lm.run(c_, c_["x0"], (0, -1), nr)
                mine.append(lm.logits_from_state(x, c_))
            c1 = lm.prep(ids, [args.init_seed + 1])
            _, x = lm.run(c1, c1["x0"], (0, -1), nr)
            other = lm.logits_from_state(x, c1)
        rep = {"model_same_seed_max_abs": float((outs[0] - outs[1]).abs().max()),
               "ours_same_seed_max_abs": float((mine[0] - mine[1]).abs().max()),
               "model_seed_change_max_abs": float((outs[0] - outs[2]).abs().max()),
               "model_seed_change_top1_agree": float((outs[0].argmax(-1) == outs[2].argmax(-1)).float().mean()),
               "ours_seed_change_max_abs": float((mine[0] - other).abs().max()),
               "ours_seed_change_top1_agree": float((mine[0].argmax(-1) == other.argmax(-1)).float().mean())}
        res["reproducible"] = rep
        ok["reproducible"] = max(rep["model_same_seed_max_abs"], rep["ours_same_seed_max_abs"]) <= args.l0_tol_logits
        log("L0 repro", json.dumps(rep))
        del outs, mine, other

    # ---- 5. convergence |h_{t+1} - h_t| / |h_t| per valid position
    R = 64 if lm.kind == "huginn" else 8
    rat = []
    for ci in range(min(args.l0_conv_ctx, len(C))):
        ctx = C.ctx(ci)
        V = valid_position_mask(ctx["T"]).nonzero(as_tuple=True)[0].to(DEV)
        rat.append(conv_curve(lm, ctx, R, V)[:, 0])  # [R, nV]
    rat = torch.cat(rat, 1)
    curve = {t: {"frac_conv": r5((rat[t - 1] < args.conv_tol).float().mean()), "median": r5(rat[t - 1].median()),
                 "p90": r5(rat[t - 1].quantile(0.9))} for t in range(1, R + 1)}
    head = 32 if lm.kind == "huginn" else lm.default_r
    res["convergence"] = {"definition": "|h_{t+1} - h_t| / |h_t| over valid positions (loop-output states)",
                          "n_ctx": min(args.l0_conv_ctx, len(C)), "n_positions": int(rat.shape[1]),
                          f"frac_converged_at_r{head}": curve[head]["frac_conv"], "by_r": curve}
    log("L0 convergence", json.dumps({t: curve[t] for t in curve if t in (1, 2, 4, 8, 16, 32, 64)}))
    torch.save({"conv_ratio": rat}, OUT / "L0_raw.pt")

    res["ok"] = ok
    res["pass"] = all(ok.values())
    res["secs"] = round(time.perf_counter() - t0, 1)
    dump("L0", res)
    log(f"L0 {'PASS' if res['pass'] else 'FAIL'}", json.dumps(ok))
    return res["pass"]


# ====================================================================== stage L1a
def l1a_sites(lm, args):
    NL, n = lm.NL, args.r
    ks = [int(x) for x in args.d1_ks.split(",")]
    if lm.kind == "huginn":
        srcs = [(int(t) - 1, NL) for t in args.l1a_push.split(",")]
        tg = {s: [(t, NL) for t in range(s[0] + 1, n)] + [("coda", n - 1)] for s in srcs}
        return srcs, tg
    srcs = [(0, min(NL - 2, max(0, round(float(fr) * NL)))) for fr in args.l1a_push.split(",")]
    tg = {}
    for s in srcs:
        gs, cand = lm.g(s), set()
        for t in range(n):
            for i in range(NL):
                g = t * NL + i
                if g > gs and ((i + 1) % args.l1a_every == 0 or (g - gs) in ks):
                    cand.add((t, i))
            cand.add((t, NL))
        tg[s] = sorted(cand, key=lm.order)
    return srcs, tg


def stage_L1a(lm, C, args):
    t0 = time.perf_counter()
    alphas = [float(x) for x in args.alphas.split(",")]
    nC, nD = args.n_ctx, args.n_dir
    assert len(C) >= nC + nD, f"need {nC + nD} contexts, have {len(C)}"
    srcs, tg = l1a_sites(lm, args)
    stats, data = source_stats(lm, C, srcs, 8, nD)
    res = {"alphas": alphas, "n_ctx": nC, "n_dir": nD, "n_loops": args.r, "NL": lm.NL, "kind": lm.kind,
           "sources": {}, "defs": None}
    nV = int(valid_position_mask(args.max_len).sum())
    raw = {}
    for s in srcs:
        lab, gs, rho = lm.label(s), lm.g(s), stats[s]["rho"]
        targets = sorted(tg[s], key=lm.order)
        gammas = [lm.gamma_for(t) for t in targets]
        types = {"iso": torch.randn(nD, lm.d, generator=torch.Generator().manual_seed(77 + gs)).to(DEV), "data": data[s]}
        M_ = {}
        for tn, U in types.items():
            U = (U / U.norm(dim=-1, keepdim=True)) * rho
            met = None
            for ci in range(nC):
                ctx = C.ctx(ci)
                xs = states_at(lm, ctx, [s])[s]
                for s0 in range(0, nD, args.batch):
                    tu = time.perf_counter()
                    aa = U[s0: s0 + args.batch].to(DTYPE)
                    f = LoopF(lm, ctx, xs, s, targets, args.r)
                    m = d1_unit(f, aa, alphas, lg=(lm.W_U, gammas))
                    if met is None:
                        met = {k_: torch.zeros((nC, nD) + v_.shape[1:]) for k_, v_ in m.items()}
                    for k_, v_ in m.items():
                        met[k_][ci, s0: s0 + len(aa)] = v_
                    log(f"L1a {lab} {tn} ctx{ci} dirs{s0}-{s0 + len(aa)} {time.perf_counter() - tu:.1f}s")
            M_[tn] = met
        raw[lab] = {"site": s, "targets": [lm.label(t) for t in targets], "k": [lm.g(t) - gs for t in targets], **M_}
        summ = {}
        for tn, met in M_.items():
            summ[tn] = {}
            for ti, t in enumerate(targets):
                row = {"k": lm.g(t) - gs, "site": list(t)}
                row.update(d1_summary(met, range(len(alphas)), ti, D1_KEYS))
                row["J_gain"] = r5(row["J_norm"] / (nV * rho)) if row["J_norm"] is not None else None
                summ[tn][lm.label(t)] = row
        res["sources"][lab] = {"site": list(s), "g": gs, "rho": rho, "rho_centered": stats[s]["rho_centered"],
                               "targets": [lm.label(t) for t in targets], "summary": summ}
        log(f"L1a {lab} rho={rho:.2f}", json.dumps({tn: {k_: (r_["k"], r_["rel_lin"][2], r_["rel_quad"][2], r_["Q_over_J"])
                                                         for k_, r_ in sm.items()} for tn, sm in summ.items()}))
        dump("L1a", res)
        torch.save(raw, OUT / "L1a_raw.pt")
    res["secs"] = round(time.perf_counter() - t0, 1)
    res["defs"] = ("D1 metrics (second_order.py) on the explicit loop forward. Steer alpha*rho*unit(dir) at every valid "
                   "position of the source site; response at each target summed over valid positions. k = blocks "
                   "traversed (Huginn: 4 per recurrence, the adapter not counted; coda adds 2). rel_lin = |act - alpha J d|"
                   "/|act|, rel_quad with + alpha^2 Q(d,d)/2; proj = <act, lin>/|lin|^2; lg_* the same through W_U gamma "
                   "(gamma = final-norm weight; none for Ouro's already-normed loop outputs). Medians over contexts x "
                   "directions; lists are per alpha. Q_over_J = |Q(d,d)|/|J d| at alpha = 1; J_gain = |J d|/(nV rho) "
                   "(|J d| relative to the summed push). Ouro: outN = loop output after model.norm, loopN.L{NL-1} the "
                   "same state before the norm. Huginn: hN = state after N recurrences.")
    dump("L1a", res)


# ====================================================================== stage L1b
def stage_L1b(lm, C, args):
    if lm.kind != "huginn":
        log("L1b is Huginn-only (variable recurrence with identical weights): skipped")
        dump("L1b", {"skipped": "Huginn-only stage"})
        return
    t0 = time.perf_counter()
    alphas = [float(x) for x in args.alphas.split(",")]
    rs = sorted(int(x) for x in args.l1b_rs.split(","))
    nC, nD, NL = args.l1b_n_ctx, args.n_dir, lm.NL
    src = (args.l1b_push - 1, NL)
    targets = sorted([(r - 1, NL) for r in rs] + [("coda", r - 1) for r in rs], key=lm.order)
    stats, data = source_stats(lm, C, [src], 8, nD)
    rho = stats[src]["rho"]
    types = {"iso": torch.randn(nD, lm.d, generator=torch.Generator().manual_seed(77 + lm.g(src))).to(DEV), "data": data[src]}
    res = {"alphas": alphas, "rs": rs, "push": lm.label(src), "rho": rho, "n_ctx": nC, "n_dir": nD,
           "conv_tol": args.conv_tol, "targets": [lm.label(t) for t in targets], "summary": {}}
    conv = {}
    for ci in range(nC):
        ctx = C.ctx(ci)
        V = valid_position_mask(ctx["T"]).nonzero(as_tuple=True)[0].to(DEV)
        conv[ci] = conv_curve(lm, ctx, max(rs), V)[:, 0]  # [R, nV]; row r-1 = |h_{r+1}-h_r|/|h_r|
    convm = {r: torch.stack([conv[ci][r - 1] < args.conv_tol for ci in range(nC)]) for r in rs}  # [nC, nV]
    res["frac_converged"] = {r: r5(convm[r].float().mean()) for r in rs}
    M_ = {}
    for tn, U in types.items():
        U = (U / U.norm(dim=-1, keepdim=True)) * rho
        met = None
        for ci in range(nC):
            ctx = C.ctx(ci)
            xs = states_at(lm, ctx, [src])[src]
            for s0 in range(0, nD, args.batch):
                tu = time.perf_counter()
                aa = U[s0: s0 + args.batch].to(DTYPE)
                f = LoopF(lm, ctx, xs, src, targets, max(rs), per_pos=True)
                m = d1_unit(f, aa, alphas)
                sm = m.pop("sum")
                if met is None:
                    met = {k_: torch.zeros((nC, nD) + v_.shape[1:]) for k_, v_ in m.items()}
                    met.update({f"sum_{k_}": torch.zeros((nC, nD) + v_.shape[1:]) for k_, v_ in sm.items()})
                for k_, v_ in m.items():
                    met[k_][ci, s0: s0 + len(aa)] = v_
                for k_, v_ in sm.items():
                    met[f"sum_{k_}"][ci, s0: s0 + len(aa)] = v_
                log(f"L1b {tn} ctx{ci} dirs{s0}-{s0 + len(aa)} {time.perf_counter() - tu:.1f}s")
        M_[tn] = met
        summ = {}
        for ti, t in enumerate(targets):
            r = (t[1] if t[0] == "coda" else t[0]) + 1
            row = {"r": r, "k": lm.g(t) - lm.g(src), "frac_converged": res["frac_converged"][r]}
            d1 = {k_: [r5(met[f"sum_{k_}"][:, :, ai, ti].median()) for ai in range(len(alphas))] for k_ in ("rel_lin", "rel_quad", "proj")}
            d1["Q_over_J"] = r5((met["sum_norm_Q"][:, :, ti] / met["sum_norm_J"][:, :, ti]).median())
            row["d1_sum"] = d1
            mask = convm[r][:, None, :].expand(nC, nD, -1)  # [nC, nD, nV]
            qj = met["norm_Q"][:, :, ti] / met["norm_J"][:, :, ti]
            for gname, gm in (("converged", mask), ("unconverged", ~mask), ("all", torch.ones_like(mask))):
                if gm.sum() == 0:
                    row[gname] = {"n": 0}
                    continue
                row[gname] = {"n": int(gm[:, 0].sum()), "Q_over_J": r5(med(qj[gm])), "J_norm": r5(med(met["norm_J"][:, :, ti][gm])),
                              "rel_lin": [r5(med(met["rel_lin"][:, :, ai, ti][gm])) for ai in range(len(alphas))],
                              "rel_quad": [r5(med(met["rel_quad"][:, :, ai, ti][gm])) for ai in range(len(alphas))]}
            summ[lm.label(t)] = row
        res["summary"][tn] = summ
        log(f"L1b {tn}", json.dumps({k_: (v_["r"], v_["frac_converged"], v_["d1_sum"]["Q_over_J"],
                                          v_["converged"].get("Q_over_J"), v_["unconverged"].get("Q_over_J")) for k_, v_ in summ.items()}))
        dump("L1b", res)
        torch.save({"targets": [lm.label(t) for t in targets], "conv": conv, **M_}, OUT / "L1b_raw.pt")
    res["secs"] = round(time.perf_counter() - t0, 1)
    res["defs"] = ("Huginn, identical weights: push alpha*rho*unit(dir) at every valid position of h_2, read the final "
                   "state h_r and coda@r (the model's final block output at r recurrences), for each r, in one forward "
                   "to max(r). d1_sum = D1's position-summed metrics. Per-position metrics (response at each valid "
                   "position to the push at all valid positions) are split by convergence at that r: "
                   "|h_{r+1}-h_r|/|h_r| < conv_tol at that position. Medians over contexts x directions x positions.")
    dump("L1b", res)


# ====================================================================== meta-device check (CPU, no weights)
def meta_check(model_dir, T=24):
    """Instantiate the architecture from a local config + remote-code dir on the meta device, wrap it, and run the
    explicit forward, a jvp and a nested jvp through it for shapes. No weights are loaded."""
    import transformers
    cfg = transformers.AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    if getattr(cfg, "model_type", "") == "ouro":
        cfg._attn_implementation = "sdpa"
    with torch.device("meta"):
        hf = transformers.AutoModelForCausalLM.from_config(cfg, trust_remote_code=True, torch_dtype=torch.float32)
    lm = wrap(hf)
    ids = torch.zeros(1, T, dtype=torch.long, device="meta")
    with torch.no_grad():
        ctx = lm.prep(ids, [0])
        lm.n_layer_calls = 0
        n = 2
        out, x = lm.run(ctx, ctx["x0"], (0, -1), n, reads=[(0, NL_) for NL_ in (lm.NL // 2, lm.NL)] + [lm.logit_site(n - 1)])
        assert lm.n_layer_calls == n * lm.NL, lm.n_layer_calls
        lg = lm.logits_from_state(x, ctx)
    src = (0, lm.NL // 2)
    f = LoopF(lm, ctx, out[src], src, [(1, lm.NL), lm.logit_site(1)], n)
    a = torch.randn(2, lm.d, device="meta")
    ja, _, q = first_and_second(f, a, None, signs=False)
    print(f"META OK {type(hf).__name__} kind={lm.kind} NL={lm.NL} d={lm.d} coda={lm.n_coda} default_r={lm.default_r} "
          f"x={tuple(x.shape)} logits={tuple(lg.shape)} J={tuple(ja.shape)} Q={tuple(q.shape)}")


# ====================================================================== main
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default="tomg-group-umd/huginn-0125")
    ap.add_argument("--revision", default="", help="default: the pinned revision for known models")
    ap.add_argument("--out", default="", help=f"default {WORK}/loop_<short>[_smoke]")
    ap.add_argument("--stages", default="L0")
    ap.add_argument("--smoke", action="store_true", help="tiny sizes (finishes in minutes)")
    ap.add_argument("--meta", default="", help="meta-device check from a local config+code dir, then exit")
    ap.add_argument("--corpus", default=f"{WORK}/corpus/fineweb_edu_2000.jsonl")
    ap.add_argument("--ctx-start", type=int, default=1000, help="as second_order.py: contexts from corpus[ctx-start:]")
    ap.add_argument("--n-pool", type=int, default=64, help="docs tokenised into the context pool (D1: 64)")
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--batch", type=int, default=8, help="directions per jvp batch")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init-seed", type=int, default=0, help="Huginn x0 seed base (context ci uses init_seed + ci)")
    ap.add_argument("--conv-tol", type=float, default=1e-3)
    ap.add_argument("--r", type=int, default=0, help="loops for L0/L1a (default: model default, Huginn 32, Ouro 4)")
    ap.add_argument("--n-ctx", type=int, default=8)
    ap.add_argument("--n-dir", type=int, default=8)
    ap.add_argument("--alphas", default="0.05,0.1,0.2,0.4")
    ap.add_argument("--d1-ks", default="1,2,4,8,16,24,32,48", help="Ouro: extra reads at these k from the source (D1's ks)")
    ap.add_argument("--l0-tol-logits", type=float, default=1e-4)
    ap.add_argument("--l0-tol-fd", type=float, default=0.995)
    ap.add_argument("--l0-eps", default="0.4,0.2,0.1,0.05,0.025,0.0125")  # factor-2 ladder for Richardson
    ap.add_argument("--l0-conv-ctx", type=int, default=8)
    ap.add_argument("--l1a-push", default="", help="Huginn: recurrences t (push at h_t), default 1,4,16; "
                                                   "Ouro: fractions of loop 1, default 0.25,0.5,0.75")
    ap.add_argument("--l1a-every", type=int, default=8, help="Ouro: read every n-th layer")
    ap.add_argument("--l1b-rs", default="4,8,16,32,64")
    ap.add_argument("--l1b-push", type=int, default=2)
    ap.add_argument("--l1b-n-ctx", type=int, default=8)
    args = ap.parse_args(argv)
    args.l0_eps = [float(x) for x in args.l0_eps.split(",")]
    if not args.revision:
        args.revision = REVS.get(args.model, "main")
    huginn = "huginn" in args.model.lower()
    if args.smoke:
        args.n_ctx, args.n_dir, args.batch, args.l0_conv_ctx = 2, 2, 2, 2
        args.l1a_push = args.l1a_push or ("4" if huginn else "0.5")
        args.l1b_rs, args.l1b_n_ctx = "4,8,16", 1
    args.l1a_push = args.l1a_push or ("1,4,16" if huginn else "0.25,0.5,0.75")
    if not args.out:
        args.out = f"{WORK}/loop_{short_name(args.model)}" + ("_smoke" if args.smoke else "")
    return args


def main(argv=None):
    global OUT
    args = parse_args(argv)
    if args.meta:
        with math_sdpa():
            meta_check(args.meta)
        return 0
    import transformers
    stages = [s for s in args.stages.split(",") if s]
    bad = set(stages) - {"L0", "L1a", "L1b"}
    assert not bad, f"unknown stages {bad}"
    OUT = Path(args.out)
    OUT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    log(f"loop_lens {args.model}@{args.revision} stages={stages} smoke={args.smoke} out={OUT} torch={torch.__version__} "
        f"transformers={transformers.__version__} host={os.uname().nodename}")
    (OUT / "args.json").write_text(json.dumps(vars(args), indent=1, default=str))
    tok = transformers.AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    kw = dict(revision=args.revision, trust_remote_code=True, torch_dtype=torch.float32, device_map=DEV)
    if "ouro" in args.model.lower():
        kw["attn_implementation"] = "sdpa"
    tl = time.perf_counter()
    hf = transformers.AutoModelForCausalLM.from_pretrained(args.model, **kw).eval()
    hf.requires_grad_(False)
    lm = wrap(hf)
    if not args.r:
        args.r = lm.default_r
    log(f"loaded {type(hf).__name__} in {time.perf_counter() - tl:.0f}s: NL={lm.NL} d={lm.d} coda={lm.n_coda} "
        f"default_r={lm.default_r} r={args.r} mem={torch.cuda.memory_allocated() / 1e9 if DEV == 'cuda' else 0:.1f} GB")
    C = Contexts(lm, tok, args)
    with no_tf32(), math_sdpa():
        if "L0" in stages and not stage_L0(lm, C, args):
            log("L0 FAILED: stopping (see L0.json)")
            return 1
        if "L1a" in stages:
            stage_L1a(lm, C, args)
        if "L1b" in stages:
            stage_L1b(lm, C, args)
    log("done", f"peak mem {torch.cuda.max_memory_allocated() / 1e9 if DEV == 'cuda' else 0:.1f} GB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
