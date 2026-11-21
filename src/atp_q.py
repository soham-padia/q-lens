"""AtP-Q: does a second-order correction fix attribution patching (AtP), and at what cost?

Setting. A clean / corrupt prompt pair of equal token length, metric m = logit(correct) - logit(incorrect) at the
last position (correct = the clean answer, incorrect = the corrupt answer). Noising direction: the base run is
clean, and a site s is patched with its corrupt activation, i.e. the clean activation a_s gets + delta_s with
delta_s = a_s(corrupt) - a_s(clean). Every site is an additive perturbation variable eps_s at a hook point, so the
single-site patch effect is exactly

    A_s = m(eps_s = delta_s) - m(0) = J_s + Q_s / 2 + O(delta^3),   J_s = grad_s m . delta_s,
                                                                     Q_s = delta_s^T H_ss delta_s,

with grad and H the total derivatives with respect to eps (everything downstream of the site moves).

Sites (all perturbation variables live in one dict E[(kind, layer)], batched over a leading B axis):
  head  o_proj's input slice of one attention head, last position     E[("head", l)]  [B, H*hd]
  mlp   the MLP output, last position                                  E[("mlp", l)]   [B, d]
  res   a block's output (layer -1 = embedding output), every position E[("res", l)]   [B, T, d]
Families for the stochastic estimator: comp = head + mlp sites, res = res sites (separate probes).

Methods per site:
  actual   batched patched forwards (each batch carries its own unpatched row as the baseline);
  AtP      J_s from one backward pass for all sites;
  exact    J_s and Q_s from one nested forward-mode jvp per site (batched over sites, tangents start at the site);
  stoch    z ~ Rademacher over the family's sites, v = z o Delta; one reverse-over-forward HVP (torch.func.vjp of a
           jvp) gives H v for every site at once; q_s = z_s delta_s^T (H v)_s is unbiased for Q_s. K draws.
  EAP-IG   gradients at m input-embedding interpolations clean -> corrupt (midpoint rule), times delta_s
           (Hanna et al. 2024), m backward passes for all sites;
  partial  actual patches at lambda * delta_s on a subset, against lambda J + lambda^2 Q / 2;
  ones     the deterministic all-sites HVP (z = 1): delta_s^T (H Delta)_s, which keeps every cross term;
  split    the exact Q-lens split of Q_s over the downstream blocks (chain rule), for the top sites.

Gates (every pair; the smoke run exits 1 if a hard gate fails, which cancels the dependent full run):
  AtP == exact J; one-hot HVP == exact Q; full-layer residual patch == corrupt run; nested jvp vs central finite
  differences (Richardson) on a few sites; stochastic mean vs exact Q (z-scores, reported).

Usage:  python src/atp_q.py --smoke
        python src/atp_q.py --tasks ioi          (resumable per pair)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.autograd.forward_ad as fwAD
from torch.func import jvp, vjp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from so_core import math_sdpa, no_tf32  # noqa: E402

KINDS = ("head", "mlp", "res")


# ================================================================== model with additive site perturbations
def _last_add(x, e):
    """x [B, T, w] + e [B, w] at the last position only (no in-place ops: works with dual / wrapped tensors)."""
    return torch.cat([x[:, :-1], (x[:, -1] + e)[:, None]], 1)


def _out_add(e):
    def hook(_mod, _inp, out):
        if torch.is_tensor(out):
            return out + e
        return (out[0] + e,) + tuple(out[1:])
    return hook


def _out_last_add(e):
    def hook(_mod, _inp, out):
        if torch.is_tensor(out):
            return _last_add(out, e)
        return (_last_add(out[0], e),) + tuple(out[1:])
    return hook


def _pre_last_add(e):
    def hook(_mod, args):
        return (_last_add(args[0], e),) + tuple(args[1:])
    return hook


class Net:
    """E -> m [B] (or the final normed hidden state at the last position [B, d] with vec=True) for one prompt."""

    def __init__(self, hf, ids, w_diff):
        self.hf, self.ids, self.wd = hf, ids, w_diff
        self.base = hf.model
        self.layers = hf.model.layers
        self.emb = hf.model.embed_tokens
        cfg = hf.config
        self.L, self.d, self.H = cfg.num_hidden_layers, cfg.hidden_size, cfg.num_attention_heads
        self.hd = getattr(cfg, "head_dim", None) or self.d // self.H
        self.T = ids.shape[1]
        self.interp = None  # (alpha [B], e_corrupt - e_clean [1, T, d]) for EAP-IG

    def module(self, kind, l):
        if kind == "head":
            return self.layers[l].self_attn.o_proj
        if kind == "mlp":
            return self.layers[l].mlp
        return self.emb if l < 0 else self.layers[l]

    def __call__(self, E, vec=False, B=None):
        if B is None:
            B = next(iter(E.values())).shape[0]
        hooks = []
        try:
            if self.interp is not None:
                a, ed = self.interp
                hooks.append(self.emb.register_forward_hook(_out_add(a[:, None, None] * ed)))
            for (kind, l), e in E.items():
                mod = self.module(kind, l)
                if kind == "head":
                    hooks.append(mod.register_forward_pre_hook(_pre_last_add(e)))
                elif kind == "mlp":
                    hooks.append(mod.register_forward_hook(_out_last_add(e)))
                else:
                    hooks.append(mod.register_forward_hook(_out_add(e)))
            h = self.base(input_ids=self.ids.expand(B, -1), use_cache=False).last_hidden_state[:, -1]
        finally:
            for hk in hooks:
                hk.remove()
        return h if vec else h @ self.wd


@torch.no_grad()
def record(net: Net, ids, res_layers):
    """Site activations of prompt ids: head [L, H*hd] and mlp [L, d] at the last position, res {l: [T, d]}, plus
    the embedding output [1, T, d] and m."""
    box = {}
    hooks = []

    def keep(key, last, pre=False):
        def hook(_mod, *a):
            x = a[0][0] if pre else (a[1] if torch.is_tensor(a[1]) else a[1][0])
            box[key] = (x[0, -1] if last else x[0]).detach().clone()
        return hook
    for l in range(net.L):
        hooks.append(net.module("head", l).register_forward_pre_hook(keep(("head", l), True, pre=True)))
        hooks.append(net.module("mlp", l).register_forward_hook(keep(("mlp", l), True)))
    for l in res_layers:
        hooks.append(net.module("res", l).register_forward_hook(keep(("res", l), False)))
    hooks.append(net.emb.register_forward_hook(keep("emb", False)))
    try:
        with math_sdpa():
            h = net.base(input_ids=ids, use_cache=False).last_hidden_state[:, -1]
    finally:
        for hk in hooks:
            hk.remove()
    box["m"] = float((h @ net.wd)[0])
    return box


# ================================================================== one clean/corrupt pair
class Lab:
    def __init__(self, hf, ids_clean, ids_corrupt, tok_correct, tok_incorrect, res_layers):
        W = hf.get_output_embeddings().weight
        wd = (W[tok_correct] - W[tok_incorrect]).detach().to(hf.model.norm.weight.dtype)
        self.net = net = Net(hf, ids_clean, wd)
        self.res_layers = list(res_layers)
        rc, rx = record(net, ids_clean, res_layers), record(net, ids_corrupt, res_layers)
        self.m_clean, self.m_corrupt = rc["m"], rx["m"]
        self.ediff = (rx["emb"] - rc["emb"])[None]
        self.keys = [(k, l) for l in range(net.L) for k in ("head", "mlp")] + [("res", l) for l in res_layers]
        self.D = {k: rx[k] - rc[k] for k in self.keys}
        self.shape = {k: tuple(self.D[k].shape) for k in self.keys}
        kind, layer, idx, key_of, slot_of = [], [], [], [], []
        for k in self.keys:
            n = self.nslots(k)
            for j in range(n):
                kind.append(KINDS.index(k[0])); layer.append(k[1]); idx.append(j if k[0] != "mlp" else -1)
                key_of.append(k)
                slot_of.append((slice(j * net.hd, (j + 1) * net.hd),) if k[0] == "head" else
                               ((slice(None),) if k[0] == "mlp" else (j,)))
        self.kind, self.layer, self.idx = np.array(kind), np.array(layer), np.array(idx)
        self.key_of, self.slot_of = key_of, slot_of
        self.S = len(kind)
        self.fam = (self.kind == 2).astype(int)  # 0 = comp (head + mlp), 1 = res
        self.fam_keys = [[k for k in self.keys if k[0] != "res"], [k for k in self.keys if k[0] == "res"]]
        self.fam_sites = [np.nonzero(self.fam == f)[0] for f in (0, 1)]
        dn = torch.stack([self.D[k][s].norm() for k, s in zip(key_of, slot_of)]).cpu().numpy()
        self.dnorm = dn
        self.null = dn <= 1e-6 * dn.max()  # e.g. residual positions before the first differing token

    def nslots(self, k):
        return self.net.H if k[0] == "head" else (1 if k[0] == "mlp" else self.net.T)

    # ---------------------------------------------------------------- building and reading E
    def place(self, idx, scale=None):
        """Batch element b carries scale[b] * delta at site idx[b] (idx < 0: nothing, the baseline row)."""
        B = len(idx)
        dev, dt = self.net.ids.device, self.net.wd.dtype
        keys = [k for k in self.keys if any(s >= 0 and self.key_of[s] == k for s in idx)]
        E = {k: torch.zeros((B,) + self.shape[k], device=dev, dtype=dt) for k in keys}
        for b, s in enumerate(idx):
            if s < 0:
                continue
            k, sl = self.key_of[s], self.slot_of[s]
            c = 1.0 if scale is None else float(scale[b])
            E[k][b][sl] = c * self.D[k][sl]
        return E

    def probe(self, keys, z):
        """z [B, n_sites(keys)] -> v = z o Delta as an E dict."""
        V, off = {}, 0
        Bz = z.shape[0]
        for k in keys:
            n = self.nslots(k)
            zk = z[:, off: off + n].to(self.net.wd.dtype)
            off += n
            if k[0] == "head":
                V[k] = (zk[:, :, None] * self.D[k].view(self.net.H, self.net.hd)[None]).reshape(Bz, -1)
            elif k[0] == "mlp":
                V[k] = zk * self.D[k][None]
            else:
                V[k] = zk[:, :, None] * self.D[k][None]
        return V

    def readoff(self, G, keys):
        """G dict key -> [B, *shape] (a gradient-like field) -> [B, n_sites(keys)]: <G_slot(s), delta_s> per site."""
        cols = []
        for k in keys:
            p = G[k] * self.D[k]
            if k[0] == "head":
                cols.append(p.reshape(p.shape[0], self.net.H, self.net.hd).sum(-1))
            elif k[0] == "mlp":
                cols.append(p.sum(-1, keepdim=True))
            else:
                cols.append(p.sum(-1))
        return torch.cat(cols, 1)

    # ---------------------------------------------------------------- methods
    @torch.no_grad()
    def patch(self, idx, scale=None, bs=128):
        """Actual effect of patching scale * delta at each site in idx (batched; each batch has a baseline row)."""
        out = []
        idx = list(idx)
        for i in range(0, len(idx), bs - 1):
            ch = idx[i: i + bs - 1]
            sc = None if scale is None else list(scale[i: i + bs - 1]) + [0.0]
            E = self.place(ch + [-1], sc)
            with math_sdpa():
                m = self.net(E, B=len(ch) + 1)
            out.append((m[:-1] - m[-1]).double().cpu())
        return torch.cat(out).numpy() if out else np.zeros(0)

    def atp(self):
        E0 = {k: torch.zeros((1,) + self.shape[k], device=self.net.ids.device, dtype=self.net.wd.dtype,
                             requires_grad=True) for k in self.keys}
        with math_sdpa():
            m = self.net(E0)
        G = torch.autograd.grad(m.sum(), [E0[k] for k in self.keys])
        return self.readoff(dict(zip(self.keys, G)), self.keys)[0].double().cpu().numpy()

    def jets(self, idx, vec=False):
        """Exact (J, Q) for single-site patches, batch element b = site idx[b]: nested forward-mode jvp."""
        T1 = self.place(list(idx))
        E0 = {k: torch.zeros_like(v) for k, v in T1.items()}
        f = lambda E: self.net(E, vec=vec)  # noqa: E731
        inner = lambda E: jvp(f, (E,), (T1,))[1]  # noqa: E731
        with math_sdpa():
            J, Q = jvp(inner, (E0,), (T1,))
        return J.detach(), Q.detach()

    def jets_all(self, idx, bs=64):
        J, Q = [], []
        for i in range(0, len(idx), bs):
            j, q = self.jets(idx[i: i + bs])
            J.append(j.double().cpu()); Q.append(q.double().cpu())
        return (torch.cat(J).numpy(), torch.cat(Q).numpy()) if J else (np.zeros(0), np.zeros(0))

    def hvp(self, fam, z):
        """z [B, n_fam] -> [B, n_fam] with entry (b, s) = delta_s^T (H v_b)_s, v_b = z_b o Delta.
        Reverse-over-forward: vjp of the jvp tangent, one pass for every site of the family."""
        keys = self.fam_keys[fam]
        V = self.probe(keys, z)
        E0 = {k: torch.zeros_like(v) for k, v in V.items()}
        tang = lambda E: jvp(self.net, (E,), (V,))[1]  # noqa: E731
        with math_sdpa():
            dm, back = vjp(tang, E0)
            (G,) = back(torch.ones_like(dm))
        return self.readoff(G, keys).detach()

    def stoch(self, fam, K, gen, bs=32):
        """K Rademacher draws: returns z [K, n] and qhat [K, n] (qhat_s = z_s delta_s^T (H (z o Delta))_s)."""
        n = len(self.fam_sites[fam])
        Z = (torch.randint(0, 2, (K, n), generator=gen) * 2 - 1).to(self.net.ids.device, self.net.wd.dtype)
        out = []
        for i in range(0, K, bs):
            z = Z[i: i + bs]
            out.append((z * self.hvp(fam, z)).double().cpu())
        return Z.cpu().numpy(), torch.cat(out).numpy()

    def eap_ig(self, steps=(4, 8, 16), bs=28):
        """EAP-IG: delta_s . mean_k grad_s m at input embeddings interpolated clean -> corrupt, alpha_k = (k+1/2)/m."""
        alphas = [(k + 0.5) / m for m in steps for k in range(m)]
        dev, dt = self.net.ids.device, self.net.wd.dtype
        rows = []
        for i in range(0, len(alphas), bs):
            a = torch.tensor(alphas[i: i + bs], device=dev, dtype=dt)
            self.net.interp = (a, self.ediff)
            try:
                E0 = {k: torch.zeros((len(a),) + self.shape[k], device=dev, dtype=dt, requires_grad=True)
                      for k in self.keys}
                with math_sdpa():
                    m = self.net(E0)
                G = torch.autograd.grad(m.sum(), [E0[k] for k in self.keys])
                rows.append(self.readoff(dict(zip(self.keys, G)), self.keys).double().cpu())
            finally:
                self.net.interp = None
        R = torch.cat(rows).numpy()
        out, off = {}, 0
        for m in steps:
            out[m] = R[off: off + m].mean(0)
            off += m
        return out

    # ---------------------------------------------------------------- Q-lens split over downstream blocks
    def site_block(self, s):
        """The block that contains (head, mlp) or precedes (res) site s; blocks after it are downstream."""
        return int(self.layer[s])

    def block_split(self, sites):
        """Exact split of Q_s over the blocks the patch passes through (chain rule for a composition):

            Q_s = sum_{k > l_s} <g_k, D^2 f_k[u_k, u_k]> + w . D^2 norm[u, u] + own_s

        u_k = first-order tangent of block k's input under the patch delta_s (all positions), g_k = dm/dh_k at the
        clean point, f_k = block k alone (its recorded call), norm = the final RMSNorm at the last position.
        own_s = Q_s - (the rest) is the site's own block: the MLP half of the layer for a head site, exactly 0 for
        mlp and res sites (a check). Returns terms [n, L + 1] (column L = final norm, NaN at blocks <= l_s)."""
        net, L = self.net, self.net.L
        sites = [int(s) for s in sites]
        n = len(sites)
        cap, hs = {}, []

        def grab(k):
            def hook(_mod, args, kwargs):
                x = args[0] if args else kwargs["hidden_states"]
                cap[k] = x.detach()
                cap[("kw", k)] = {kk: v for kk, v in kwargs.items() if kk != "hidden_states"}
            return hook
        hs = [net.layers[k].register_forward_pre_hook(grab(k), with_kwargs=True) for k in range(L)]
        hs.append(net.layers[L - 1].register_forward_hook(lambda _m, _i, o: cap.__setitem__(
            "out", (o if torch.is_tensor(o) else o[0]).detach())))
        try:
            with torch.no_grad(), math_sdpa():
                net.base(input_ids=net.ids, use_cache=False)
        finally:
            for h in hs:
                h.remove()
        dev, dt = net.ids.device, net.wd.dtype
        E0 = {("res", k): torch.zeros(1, net.T, net.d, device=dev, dtype=dt, requires_grad=True) for k in range(L)}
        with math_sdpa():
            m = net(E0)
        G = torch.autograd.grad(m.sum(), list(E0.values()))
        U, hs = {}, []

        def rec(k):
            def hook(_mod, args, kwargs):
                x = args[0] if args else kwargs["hidden_states"]
                t = fwAD.unpack_dual(x).tangent
                U[k] = None if t is None else t.detach()
            return hook
        hs = [net.layers[k].register_forward_pre_hook(rec(k), with_kwargs=True) for k in range(L)]
        hs.append(net.layers[L - 1].register_forward_hook(lambda _m, _i, o: U.__setitem__(
            L, fwAD.unpack_dual(o if torch.is_tensor(o) else o[0]).tangent.detach())))
        T1 = self.place(sites)
        try:
            with fwAD.dual_level(), math_sdpa():
                net({k: fwAD.make_dual(torch.zeros_like(v), v) for k, v in T1.items()})
        finally:
            for h in hs:
                h.remove()
        terms = torch.full((n, L + 1), float("nan"), dtype=torch.float64)
        for k in range(L):
            need = [b for b, s in enumerate(sites) if k > self.site_block(s)]
            if not need:
                continue
            if U.get(k) is None:  # no site of this batch is upstream of block k: zero tangent
                continue
            u = U[k][need]
            kw, x0 = cap[("kw", k)], cap[k].expand(len(need), -1, -1).contiguous()
            lay = net.layers[k]
            fk = lambda x, lay=lay, kw=kw: (lambda o: o if torch.is_tensor(o) else o[0])(lay(x, **kw))  # noqa: E731
            with math_sdpa():
                _, D2 = jvp(lambda x: jvp(fk, (x,), (u,))[1], (x0,), (u,))
            terms[need, k] = (D2 * G[k]).sum((1, 2)).double().cpu()
        fn = lambda x: net.base.norm(x) @ net.wd  # noqa: E731
        u, x0 = U[L][:, -1].contiguous(), cap["out"][:, -1].expand(n, -1).contiguous()
        _, qn = jvp(lambda x: jvp(fn, (x,), (u,))[1], (x0,), (u,))
        terms[:, L] = qn.double().cpu()
        return terms.numpy()

    # ---------------------------------------------------------------- gates
    def fd_gate(self, sites, hs=((0.2, 0.1), (0.1, 0.05), (0.05, 0.025)), hjs=((0.1, 0.05), (0.05, 0.025), (0.02, 0.01))):
        """Nested-jvp J, Q against central differences of the final hidden state (Richardson on h, h/2; Q_rel), and
        Q against central differences of the first-order jvp tangent (Qj_rel, the hard gate)."""
        Jv, Qv = self.jets(list(sites), vec=True)
        Jv, Qv = Jv.double(), Qv.double()
        res = []
        for b, s in enumerate(sites):
            best = None
            for h1, h2 in hs:
                sc = [h1, -h1, h2, -h2, 0.0]
                E = self.place([s] * 5, sc)
                with torch.no_grad(), math_sdpa():
                    F = self.net(E, vec=True).double()
                f0 = F[4]
                Jh1, Jh2 = (F[0] - F[1]) / (2 * h1), (F[2] - F[3]) / (2 * h2)
                Qh1, Qh2 = (F[0] - 2 * f0 + F[1]) / h1 ** 2, (F[2] - 2 * f0 + F[3]) / h2 ** 2
                r = (h1 / h2) ** 2
                Jfd, Qfd = (r * Jh2 - Jh1) / (r - 1), (r * Qh2 - Qh1) / (r - 1)
                wd = self.net.wd.double()
                g = {"h": h1, "J_rel": float((Jfd - Jv[b]).norm() / Jv[b].norm()),
                     "Q_rel": float((Qfd - Qv[b]).norm() / Qv[b].norm()),
                     "Q_cos": float(torch.nn.functional.cosine_similarity(Qfd, Qv[b], 0)),
                     "Qm_rel": float(abs(Qfd @ wd - Qv[b] @ wd) / abs(Qv[b] @ wd).clamp_min(1e-30)),
                     "Jm_rel": float(abs(Jfd @ wd - Jv[b] @ wd) / abs(Jv[b] @ wd).clamp_min(1e-30))}
                if best is None or g["Q_rel"] < best["Q_rel"]:
                    best = g
            best.update(site=int(s), kind=KINDS[self.kind[s]], layer=int(self.layer[s]), idx=int(self.idx[s]))
            # Q as a central difference of the jvp tangent J(h delta): replaces only the outer AD level by FD, with
            # roundoff ~ eps / h instead of eps / h^2 (value second differences drown small per-head deltas in fp32)
            bj = None
            for h1, h2 in hjs:
                E = self.place([s] * 4, [h1, -h1, h2, -h2])
                T1 = self.place([s] * 4)
                with torch.no_grad(), math_sdpa():
                    _, Jt = jvp(lambda e: self.net(e, vec=True), (E,), (T1,))
                Jt = Jt.double()
                q1, q2 = (Jt[0] - Jt[1]) / (2 * h1), (Jt[2] - Jt[3]) / (2 * h2)
                r = (h1 / h2) ** 2
                qj = (r * q2 - q1) / (r - 1)
                gj = {"hj": h1, "Qj_rel": float((qj - Qv[b]).norm() / Qv[b].norm()),
                      "Qj_cos": float(torch.nn.functional.cosine_similarity(qj, Qv[b], 0))}
                if bj is None or gj["Qj_rel"] < bj["Qj_rel"]:
                    bj = gj
            best.update(bj)
            res.append(best)
        return res

    @torch.no_grad()
    def full_res_gate(self):
        """Patching a whole residual layer (every position) must reproduce the corrupt run."""
        out = {}
        for l in (self.res_layers[0], self.res_layers[-1]):
            with math_sdpa():
                m = float(self.net({("res", l): self.D[("res", l)][None]})[0])
            out[l] = abs(m - self.m_corrupt)
        return out


# ================================================================== the per-pair experiment (job and tests)
def _oom_retry(fn, bs, lo=1):
    """fn(bs), halving bs on CUDA out-of-memory."""
    while True:
        try:
            return fn(bs)
        except torch.cuda.OutOfMemoryError:
            if bs <= lo:
                raise
            torch.cuda.empty_cache()
            bs = max(lo, bs // 2)
            print(f"OOM: retrying with bs={bs}", flush=True)


def run_pair(lab: Lab, *, K=128, ig_steps=(4, 8, 16), lams=(0.25, 0.5), n_partial=20, exact_max=0,
             bs_patch=128, bs_jet=64, bs_hvp=32, bs_ig=28, seed=0, n_split=(10, 5), log=print):
    sync = (lambda: torch.cuda.synchronize()) if lab.net.ids.is_cuda else (lambda: None)  # noqa: E731
    times, rng = {}, np.random.default_rng(seed)
    S = lab.S
    live = np.nonzero(~lab.null)[0]

    def clock(name, fn):
        sync(); t = time.perf_counter()
        r = fn()
        sync(); times[name] = time.perf_counter() - t
        return r

    actual = np.zeros(S)
    actual[live] = clock("patch", lambda: _oom_retry(lambda b: lab.patch(live, bs=b), bs_patch, 2))
    atp = clock("atp", lab.atp)
    # exact jets: every live site, or a random subset (always including the partial-patch and gate sites)
    part = []
    for f in (0, 1):
        fl = [s for s in lab.fam_sites[f] if not lab.null[s]]
        if not fl:
            continue
        top = sorted(fl, key=lambda s: -abs(actual[s]))[:n_partial]
        rest = [s for s in fl if s not in set(top)]
        part += top + list(rng.choice(rest, min(n_partial, len(rest)), replace=False))
    part = np.array(sorted(set(int(s) for s in part)))
    comp_live = [s for s in lab.fam_sites[0] if not lab.null[s]]
    res_live = [s for s in lab.fam_sites[1] if not lab.null[s]]
    top_comp = sorted(comp_live, key=lambda s: -abs(actual[s]))
    # FD gate sites. Hard (they test the AD, so they must sit inside the Taylor regime at h <= 0.2): the sites at
    # the 25th and 50th percentile of AtP's relative error among the comp top-60 by |A| (layers < L-2: Q not
    # negligible, not wildly non-perturbative), and the deepest-layer residual site at the last
    # position. Reported only: the top comp site by |A| and a random residual site (may be non-perturbative).
    t50 = [s for s in top_comp[:60] if lab.layer[s] < lab.net.L - 2] or top_comp[:50]
    calm = sorted(t50, key=lambda s: abs(actual[s] - atp[s]) / max(abs(actual[s]), 1e-30))
    fd_sites, fd_hard = [calm[len(calm) // 4], calm[len(calm) // 2]], [True, True]  # moderately nonlinear sites
    if top_comp[0] not in fd_sites:
        fd_sites.append(top_comp[0]); fd_hard.append(False)
    if res_live:
        last_pos = [s for s in res_live if lab.layer[s] == lab.res_layers[-1] and lab.idx[s] == lab.net.T - 1]
        fd_sites.append((last_pos[:1] or [res_live[-1]])[0]); fd_hard.append(True)
        r_ = int(rng.choice(res_live))
        if r_ not in fd_sites:
            fd_sites.append(r_); fd_hard.append(False)
    fd_sites = [int(s) for s in fd_sites]
    ex = live if not exact_max or exact_max >= len(live) else \
        np.array(sorted(set(rng.choice(live, exact_max, replace=False).tolist()) | set(part.tolist()) | set(fd_sites)))
    J = np.full(S, np.nan); Q = np.full(S, np.nan)
    J[lab.null], Q[lab.null] = 0.0, 0.0
    jj, qq = clock("exact", lambda: _oom_retry(lambda b: lab.jets_all(ex, bs=b), bs_jet))
    J[ex], Q[ex] = jj, qq
    # stochastic diagonal, per family
    gen = torch.Generator().manual_seed(10_000 + seed)
    stoch = {}
    for f in (0, 1):
        if len(lab.fam_sites[f]):
            z, qh = clock(f"stoch{f}", lambda: _oom_retry(lambda b: lab.stoch(f, K, gen, bs=b), bs_hvp))
            stoch[f] = qh.astype(np.float32)
    # deterministic all-sites HVP (z = 1): delta_s^T (H Delta)_s keeps every cross term (one HVP per family)
    ones = {}
    for f in (0, 1):
        if len(lab.fam_sites[f]):
            z1 = torch.ones(1, len(lab.fam_sites[f]), device=lab.net.ids.device, dtype=lab.net.wd.dtype)
            ones[f] = clock(f"ones{f}", lambda: lab.hvp(f, z1))[0].double().cpu().numpy()
    ig = clock("ig", lambda: _oom_retry(lambda b: lab.eap_ig(ig_steps, bs=b), bs_ig))
    # Q-lens split of Q_s over downstream blocks, for the top sites by |actual| of each family
    split_sites = top_comp[:n_split[0]] + sorted(res_live, key=lambda s: -abs(actual[s]))[:n_split[1]]
    split = clock("split", lambda: lab.block_split(split_sites)) if split_sites else np.zeros((0, lab.net.L + 1))
    own = Q[split_sites] - np.nansum(split, 1) if split_sites else np.zeros(0)
    # partial patches
    partial = {}
    for lam in lams:
        partial[lam] = clock(f"partial{lam}", lambda: _oom_retry(lambda b: lab.patch(part, [lam] * len(part), bs=b),
                                                                  bs_patch, 2))
    # gates
    g = {}
    exok = ~np.isnan(J)
    g["atp_vs_J"] = float(np.linalg.norm(atp[exok] - J[exok]) / max(np.linalg.norm(J[exok]), 1e-30))
    oh_rel = []
    for f in (0, 1):
        ss = [s for s in fd_sites if lab.fam[s] == f]
        if not ss:
            continue
        pos = {int(s): i for i, s in enumerate(lab.fam_sites[f])}
        z = torch.zeros(len(ss), len(lab.fam_sites[f]), device=lab.net.ids.device, dtype=lab.net.wd.dtype)
        for b, s in enumerate(ss):
            z[b, pos[s]] = 1.0
        hv = lab.hvp(f, z).double().cpu().numpy()
        for b, s in enumerate(ss):
            oh_rel.append(abs(hv[b, pos[s]] - Q[s]) / max(abs(Q[s]), 1e-4 * abs(lab.m_clean - lab.m_corrupt), 1e-30))
    g["hvp_onehot_rel"] = float(max(oh_rel))
    g["fd"] = clock("fd", lambda: lab.fd_gate(fd_sites))
    for rec_, h_ in zip(g["fd"], fd_hard):
        rec_["hard"] = h_
    g["full_res_abs"] = {int(k): float(v) for k, v in lab.full_res_gate().items()}
    zs = {}
    for f, qh in stoch.items():
        sites = lab.fam_sites[f]
        ok = exok[sites] & ~lab.null[sites]
        mu, sd = qh.mean(0)[ok], qh.std(0, ddof=1)[ok] / math.sqrt(qh.shape[0])
        zz = (mu - Q[sites][ok]) / np.maximum(sd, 1e-30)
        zs[f] = {"frac_abs_z_lt3": float(np.mean(np.abs(zz) < 3)), "median_abs_z": float(np.median(np.abs(zz)))}
    g["stoch_z"] = zs
    chk = [abs(o) / max(np.nansum(np.abs(t)), 1e-30) for s, o, t in zip(split_sites, own, split) if lab.kind[s] != 0]
    g["split_resid_rel"] = float(max(chk)) if chk else 0.0  # mlp / res sites: the split must sum to Q exactly
    counts = {"S": S, "live": int(len(live)), "exact": int(len(ex)), "K": K, "ig_rows": int(sum(ig_steps)),
              "partial": int(len(part)), "fam_sizes": [int(len(x)) for x in lab.fam_sites],
              "fam_live": [int((~lab.null[x]).sum()) for x in lab.fam_sites]}
    return {"actual": actual, "atp": atp, "J": J, "Q": Q, "stoch": stoch, "ig": ig, "partial_sites": part,
            "partial": partial, "gates": g, "times": times, "counts": counts, "kind": lab.kind, "layer": lab.layer,
            "idx": lab.idx, "fam": lab.fam, "null": lab.null, "dnorm": lab.dnorm, "m_clean": lab.m_clean,
            "m_corrupt": lab.m_corrupt, "fd_sites": np.array(fd_sites), "exact_sites": np.asarray(ex),
            "ones": ones, "split_sites": np.array(split_sites, dtype=int), "split": split, "split_own": own}


def hard_gates(g, tol_atp=1e-3, tol_oh=1e-3, tol_full=2e-2, fd_J=1e-2, fd_Q=2e-2, tol_split=1e-2):
    """Pass/fail list for one pair's gates (FD gate on the sites flagged hard)."""
    fails = []
    if not g["atp_vs_J"] < tol_atp:
        fails.append(f"atp_vs_J {g['atp_vs_J']:.2e}")
    if not g["hvp_onehot_rel"] < tol_oh:
        fails.append(f"hvp_onehot {g['hvp_onehot_rel']:.2e}")
    for l, v in g["full_res_abs"].items():
        if not v < tol_full:
            fails.append(f"full_res l{l} {v:.2e}")
    if not g.get("split_resid_rel", 0.0) < tol_split:
        fails.append(f"split_resid {g['split_resid_rel']:.2e}")
    for r in g["fd"]:
        if r.get("hard", True) and not (r["J_rel"] < fd_J and r["Qj_rel"] < fd_Q):
            fails.append(f"fd {r['kind']} l{r['layer']} i{r['idx']} J {r['J_rel']:.1e} Qj {r['Qj_rel']:.1e} "
                         f"(value-FD Q {r['Q_rel']:.1e})")
    return fails


# ================================================================== tasks
NAMES = ("Mary John Tom James David Sarah Michael Lisa Anna Paul Mark Kate Emma Peter Jack Alice Bob Daniel Laura "
         "Steve Rachel Kevin Linda Brian Susan George Helen Frank Nancy Robert Jane Henry Amy Eric Karen Adam Grace "
         "Ryan Claire Sam Julia Alex Lucy Chris Sophie Ben Joe Tim Dan Mike Jessica").split()
IOI_T = [
    "When {X} and {Y} went to the store, {S} gave a drink to",
    "When {X} and {Y} got a snack at the cafe, {S} decided to give it to",
    "Then, {X} and {Y} were working at the office. {S} decided to give a file to",
    "While {X} and {Y} were commuting to school, {S} gave a bag to",
    "After {X} and {Y} went to the garden, {S} gave a flower to",
    "Friends {X} and {Y} found a ring at the beach. {S} gave it to",
    "When {X} and {Y} arrived at the party, {S} handed a gift to",
    "Yesterday {X} and {Y} cleaned the kitchen, and then {S} passed the towel to",
]
CAPITAL = {"France": "Paris", "Germany": "Berlin", "Spain": "Madrid", "Italy": "Rome", "Japan": "Tokyo",
           "Russia": "Moscow", "China": "Beijing", "Egypt": "Cairo", "Greece": "Athens", "Austria": "Vienna",
           "Ireland": "Dublin", "Poland": "Warsaw", "Portugal": "Lisbon", "Norway": "Oslo", "Sweden": "Stockholm",
           "Finland": "Helsinki", "Denmark": "Copenhagen", "Hungary": "Budapest", "Belgium": "Brussels",
           "Netherlands": "Amsterdam", "Canada": "Ottawa", "Australia": "Canberra", "Korea": "Seoul",
           "Thailand": "Bangkok", "Philippines": "Manila", "Indonesia": "Jakarta", "Iran": "Tehran",
           "Iraq": "Baghdad", "Afghanistan": "Kabul", "Peru": "Lima", "Chile": "Santiago", "Cuba": "Havana",
           "Kenya": "Nairobi", "Turkey": "Ankara", "Syria": "Damascus", "Lebanon": "Beirut", "England": "London",
           "Argentina": "Buenos", "Bulgaria": "Sofia", "Pakistan": "Islamabad", "Tunisia": "Tunis",
           "Scotland": "Edinburgh", "Wales": "Cardiff", "Switzerland": "Bern", "Malaysia": "Kuala",
           "Jamaica": "Kingston", "Taiwan": "Taipei"}
FACT_T = ["The capital of {C} is", "The capital city of {C} is the city of", "Q: What is the capital of {C}?\nA:",
          "In {C}, the capital city is"]


def one_tok(tok, s):
    ids = tok(s, add_special_tokens=False).input_ids
    return ids[0] if len(ids) == 1 else None


def candidates(tok, task, seed=0):
    rng = random.Random(seed)
    out = []
    if task == "ioi":
        names = [n for n in NAMES if one_tok(tok, " " + n) is not None]
        for i in range(160):
            t = IOI_T[i % len(IOI_T)]
            io, s = rng.sample(names, 2)
            x, y = (io, s) if (i // len(IOI_T)) % 2 == 0 else (s, io)  # ABBA / BABA
            out.append({"clean": t.format(X=x, Y=y, S=s), "corrupt": t.format(X=x, Y=y, S=io),
                        "correct": " " + io, "incorrect": " " + s, "template": i % len(IOI_T)})
    elif task == "fact":
        cs = [c for c in CAPITAL if one_tok(tok, " " + c) is not None and one_tok(tok, " " + CAPITAL[c]) is not None]
        for ti, t in enumerate(FACT_T):
            for c in cs:
                c2 = rng.choice([x for x in cs if x != c and CAPITAL[x] != CAPITAL[c]])
                out.append({"clean": t.format(C=c), "corrupt": t.format(C=c2), "correct": " " + CAPITAL[c],
                            "incorrect": " " + CAPITAL[c2], "template": ti})
        rng.shuffle(out)
    else:
        raise ValueError(task)
    keep = []
    for p in out:
        a = tok(p["clean"], add_special_tokens=False).input_ids
        b = tok(p["corrupt"], add_special_tokens=False).input_ids
        ca, ia = one_tok(tok, p["correct"]), one_tok(tok, p["incorrect"])
        if len(a) == len(b) and ca is not None and ia is not None and ca != ia and a != b:
            p.update(ids_clean=a, ids_corrupt=b, tok_correct=ca, tok_incorrect=ia,
                     diff_pos=[i for i in range(len(a)) if a[i] != b[i]])
            keep.append(p)
    return keep


@torch.no_grad()
def filter_pairs(hf, cands, n, dev):
    """Keep pairs the model gets right on the clean prompt (m_clean > 0), in candidate order, up to n."""
    W = hf.get_output_embeddings().weight
    out = []
    for p in cands:
        ms = []
        for key in ("ids_clean", "ids_corrupt"):
            ids = torch.tensor([p[key]], device=dev)
            with math_sdpa():
                h = hf.model(input_ids=ids, use_cache=False).last_hidden_state[0, -1]
            ms.append(float(h @ (W[p["tok_correct"]] - W[p["tok_incorrect"]])))
        p["m_clean"], p["m_corrupt"] = ms
        if ms[0] > 0:
            out.append(p)
        if len(out) >= n:
            break
    return out


# ================================================================== main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--out", default=os.path.join(os.environ.get("QLENS_OUT", "outputs"), "atp_q_qwen3-8b"))
    ap.add_argument("--tasks", default="ioi,fact")
    ap.add_argument("--n-pairs", type=int, default=50)
    ap.add_argument("--pairs", default="", help="a:b slice of the selected pairs (sharding)")
    ap.add_argument("--res-layers", default="0,4,8,12,16,20,24,28,32")
    ap.add_argument("--K", type=int, default=128)
    ap.add_argument("--ig-steps", default="4,8,16")
    ap.add_argument("--lams", default="0.25,0.5")
    ap.add_argument("--n-partial", type=int, default=20)
    ap.add_argument("--exact-max", type=int, default=0, help="0 = exact J+Q at every live site")
    ap.add_argument("--bs-patch", type=int, default=128)
    ap.add_argument("--bs-jet", type=int, default=64)
    ap.add_argument("--bs-hvp", type=int, default=32)
    ap.add_argument("--bs-ig", type=int, default=28)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true", help="2 pairs per task, hard gates decide the exit code")
    args = ap.parse_args()

    import transformers
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    out = args.out + ("_smoke" if args.smoke else "")
    os.makedirs(out, exist_ok=True)
    t0 = time.perf_counter()
    log = lambda *a: print(f"[{time.perf_counter() - t0:7.1f}s]", *a, flush=True)  # noqa: E731
    tok = transformers.AutoTokenizer.from_pretrained(args.model)
    hf = transformers.AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32,
                                                           attn_implementation="sdpa").cuda().eval()
    for p in hf.parameters():
        p.requires_grad_(False)
    dev = torch.device("cuda")
    log(f"loaded {args.model}: L={hf.config.num_hidden_layers} d={hf.config.hidden_size} "
        f"H={hf.config.num_attention_heads} gpu={torch.cuda.get_device_name(0)}")
    res_layers = [int(x) for x in args.res_layers.split(",")]
    n_pairs = 2 if args.smoke else args.n_pairs
    ig_steps = tuple(int(x) for x in args.ig_steps.split(","))
    lams = tuple(float(x) for x in args.lams.split(","))
    all_fail = []
    for task in args.tasks.split(","):
        tf = os.path.join(out, f"pairs_{task}.json")
        if os.path.exists(tf):
            pairs = json.load(open(tf))
        else:
            c = candidates(tok, task, seed=args.seed)
            pairs = filter_pairs(hf, c, n_pairs, dev)
            json.dump(pairs, open(tf + f".tmp{os.getpid()}", "w"), indent=1)
            os.replace(tf + f".tmp{os.getpid()}", tf)  # atomic: shards of one task may race to write it
            log(f"{task}: {len(c)} candidates, kept {len(pairs)} with m_clean > 0; m_clean median "
                f"{np.median([p['m_clean'] for p in pairs]):.2f}, m_corrupt median "
                f"{np.median([p['m_corrupt'] for p in pairs]):.2f}")
        sl = slice(*[int(x) if x else None for x in args.pairs.split(":")]) if args.pairs else slice(None)
        for i in range(len(pairs))[sl]:
            fn = os.path.join(out, f"{task}_{i:03d}.pt")
            if os.path.exists(fn):
                log("skip", fn)
                continue
            p = pairs[i]
            tc = time.perf_counter()
            with no_tf32():
                lab = Lab(hf, torch.tensor([p["ids_clean"]], device=dev), torch.tensor([p["ids_corrupt"]], device=dev),
                          p["tok_correct"], p["tok_incorrect"], res_layers)
                r = run_pair(lab, K=args.K, ig_steps=ig_steps, lams=lams, n_partial=args.n_partial,
                             exact_max=args.exact_max, bs_patch=args.bs_patch, bs_jet=args.bs_jet,
                             bs_hvp=args.bs_hvp, bs_ig=args.bs_ig, seed=args.seed * 1000 + i, log=log)
            r.update(task=task, i=i, pair={k: p[k] for k in ("clean", "corrupt", "correct", "incorrect",
                                                             "template", "diff_pos")},
                     T=lab.net.T, res_layers=res_layers, ig_steps=ig_steps, lams=lams, secs=time.perf_counter() - tc,
                     peak_mem_gb=torch.cuda.max_memory_allocated() / 2 ** 30)
            torch.save(r, fn)
            fails = hard_gates(r["gates"])
            all_fail += [f"{task}_{i}: {f}" for f in fails]
            a, j, q = r["actual"], r["atp"], r["Q"]
            ok = ~r["null"] & ~np.isnan(q)
            err_j = np.abs(j - a)[ok].sum() / np.abs(a[ok]).sum()
            err_jq = np.abs(j + q / 2 - a)[ok].sum() / np.abs(a[ok]).sum()
            g = r["gates"]
            log(f"{task}_{i:03d} T={lab.net.T} m {lab.m_clean:.2f}->{lab.m_corrupt:.2f} | sum|err|/sum|A|: AtP "
                f"{err_j:.3f} J+Q {err_jq:.3f} | gates atp_vs_J {g['atp_vs_J']:.1e} onehot {g['hvp_onehot_rel']:.1e} "
                f"fullres {max(g['full_res_abs'].values()):.1e} fd(Q_rel) {[round(x['Q_rel'], 4) for x in g['fd']]} "
                f"fd-of-jvp(Qj_rel) {[round(x['Qj_rel'], 5) for x in g['fd']]} "
                f"split_resid {g['split_resid_rel']:.1e} stoch_z {g['stoch_z']} | times {json.dumps({k: round(v, 1) for k, v in r['times'].items()})} "
                f"| {r['secs']:.0f}s peak {r['peak_mem_gb']:.1f} GB {'FAIL ' + str(fails) if fails else 'ok'}")
    json.dump({"fails": all_fail}, open(os.path.join(out, "gates_summary.json"), "w"), indent=1)
    log("GATES", "PASS" if not all_fail else f"FAIL {all_fail}")
    log("done")
    if args.smoke and all_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
