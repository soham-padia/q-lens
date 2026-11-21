"""Second-order ("double Jacobian") primitives for a J-lens-style analysis of an HF decoder.

Notation (matches the J-lens repo): source layer l, target L = final block output (pre-norm),
valid positions V = 16..T-2 (jlens.fitting.valid_position_mask). For a prompt with residual
stack x = h_l[:, V] we define the target functional

    F(x) = sum_{t' in V} h_L,t'(x)                     (a d-vector)

so the repo's lens is  J_l a = E_prompt mean_{t in V} dF/dh_l,t . a .

Second order ("double Jacobian", the Jacobian of the Jacobian):

    Q_l(a, b) = E_prompt mean_{t in V} d^2F / dh_l,t^2 [a, b]      (a d-vector)

i.e. the same-position block of the Hessian of every target coordinate, contracted with two
residual directions. Computed exactly with nested forward-mode AD (jvp of jvp). The same-position
block is isolated with Rademacher position signs: perturbing by A[t]=e_t a, B[t]=e_t b gives
sum_{t1,t2} e_t1 e_t2 D2F[a@t1, b@t2], whose expectation over e is sum_t D2F[a@t, b@t].
With signs=False every valid position gets +a/+b (the "steer at every position" setting, which
includes the cross-position terms).

Readout of any d-vector y in h_L space: logits = W_U (gamma * y)  (rank-equivalent to the
J-lens readout W_U norm(y), since RMSNorm without centring is a positive scalar times gamma*).
"""
from __future__ import annotations

import contextlib
import os
import sys

import torch
from torch.func import jvp

JLENS_DIR = os.environ.get("JLENS_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "jacobian-lens"))
sys.path.insert(0, JLENS_DIR)
from jlens.fitting import valid_position_mask  # noqa: E402

MATH = torch.nn.attention.SDPBackend.MATH


def math_sdpa():
    return torch.nn.attention.sdpa_kernel(MATH)


class Perturbed:
    """Callable delta -> F: runs the model with h_l += delta, returns sum_{t' in V} h_L,t'."""

    def __init__(self, m, ids: torch.Tensor, layer: int, target: int, reduce: str = "valid_sum"):
        self.m, self.ids, self.layer, self.target, self.reduce = m, ids, layer, target, reduce
        T = ids.shape[1]
        self.valid = None if reduce == "last" else valid_position_mask(T).nonzero(as_tuple=True)[0].to(ids.device)
        self.T = T

    def __call__(self, delta: torch.Tensor) -> torch.Tensor:
        box = {}

        def add_hook(_mod, _inp, out):
            if torch.is_tensor(out):
                return out + delta
            return (out[0] + delta,) + tuple(out[1:])

        def rec_hook(_mod, _inp, out):
            box["h"] = out if torch.is_tensor(out) else out[0]

        h1 = self.m.layers[self.layer].register_forward_hook(add_hook)
        h2 = self.m.layers[self.target].register_forward_hook(rec_hook)
        try:
            self.m.forward(self.ids)
        finally:
            h1.remove(); h2.remove()
        if self.reduce == "last":
            return box["h"][:, -1]  # [B, d]
        return box["h"][:, self.valid].sum(1)  # [B, d]

    def spread(self, vecs: torch.Tensor, signs: torch.Tensor | None) -> torch.Tensor:
        """[B, d] directions -> [B, T, d] perturbation on valid positions (optionally signed)."""
        B, d = vecs.shape
        out = torch.zeros(B, self.T, d, device=vecs.device, dtype=vecs.dtype)
        w = torch.ones(B, len(self.valid), 1, device=vecs.device, dtype=vecs.dtype) if signs is None \
            else signs.to(vecs.dtype)[..., None]
        out[:, self.valid] = w * vecs[:, None, :]
        return out


def rademacher(B: int, n: int, device, gen: torch.Generator | None = None) -> torch.Tensor:
    return torch.randint(0, 2, (B, n), device="cpu", generator=gen).to(device).float() * 2 - 1


def first_and_second(f: Perturbed, a: torch.Tensor, b: torch.Tensor | None = None, *, signs: bool, gen=None,
                     second: bool = True):
    """For a batch of direction pairs (a_i, b_i) [B, d] (batch = copies of one prompt):
    returns (Ja, Jb, Qab) each [B, d] for this prompt, already divided by |V| (mean over sources).
    Ja, Jb: first-order pushforwards (unsigned: J_l a in the repo's sense). Qab: second order."""
    B = a.shape[0]
    nV = len(f.valid)
    same = b is None
    b = a if same else b
    e = rademacher(B, nV, a.device, gen) if signs else None
    A1 = f.spread(a, None)
    zero = torch.zeros_like(A1)
    Jb = Q = None
    with math_sdpa():
        _, Ja = jvp(f, (zero,), (A1,))
        if not same:
            _, Jb = jvp(f, (zero,), (f.spread(b, None),))
        if second:
            A, Bv = f.spread(a, e), f.spread(b, e)
            inner = lambda dlt: jvp(f, (dlt,), (A,))[1]  # noqa: E731
            _, Q = jvp(inner, (zero,), (Bv,))
    Jb = Ja if same else Jb
    return Ja / nV, Jb / nV, (Q / nV if second else torch.zeros_like(Ja))


def readout(y: torch.Tensor, W_U: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    """[..., d] -> [..., vocab] logits (fp32)."""
    return (y.float() * gamma) @ W_U.float().T


def rank_of(logits: torch.Tensor, ids: list[int]) -> int:
    """Best (min) 0-based rank among a set of token ids, for one [vocab] logit row."""
    sub = logits[ids]
    return int((logits[None, :] > sub[:, None]).sum(-1).min())


@contextlib.contextmanager
def no_tf32():
    old = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = old


# ---------------------------------------------------------------- big models (multi-GPU, bf16-stored weights)
# Mixed mode: weights stay in their checkpoint dtype (bf16) and are upcast per call, so every op runs in the
# compute dtype (fp32). bf16 -> fp32 is exact, so this is numerically the same model as loading in fp32
# (HF checkpoints are bf16), at half the memory. Only F.linear / F.conv1d see raw weights in the text path;
# norms, embedding and lm_head are cast to fp32 outright.
UPCAST = {"on": False, "dtype": torch.float32}
_F_linear, _F_conv1d = torch.nn.functional.linear, torch.nn.functional.conv1d


def _linear(x, w, b=None):
    if UPCAST["on"] and w.dtype != x.dtype and x.dtype == UPCAST["dtype"]:
        w = w.to(x.dtype); b = None if b is None else b.to(x.dtype)
    return _F_linear(x, w, b)


def _conv1d(x, w, b=None, *a, **k):
    if UPCAST["on"] and w.dtype != x.dtype and x.dtype == UPCAST["dtype"]:
        w = w.to(x.dtype); b = None if b is None else b.to(x.dtype)
    return _F_conv1d(x, w, b, *a, **k)


def enable_upcast(hf, dtype=torch.float32):
    torch.nn.functional.linear, torch.nn.functional.conv1d = _linear, _conv1d
    UPCAST.update(on=True, dtype=dtype)
    hf.get_input_embeddings().to(dtype); hf.get_output_embeddings().to(dtype)
    for mod in hf.modules():
        if "norm" in type(mod).__name__.lower():
            mod.to(dtype)


@contextlib.contextmanager
def bf16_backward():
    """For reverse-mode stages in mixed mode: autocast to bf16 with the upcast off, so the autograd graph
    saves bf16 params (no per-layer fp32 weight copies). Used only for the concept directions (vecs)."""
    was = UPCAST["on"]
    if not was:
        yield; return
    UPCAST["on"] = False
    try:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            yield
    finally:
        UPCAST["on"] = was


def layer_device_map(cls, cfg, n_gpu: int, gpu0_fewer: int = 2) -> dict:
    """Explicit device map: decoder layers split over n_gpu GPUs (GPU0 gets `gpu0_fewer` fewer), everything
    else (embedding, final norm, lm_head, rotary, vision tower) on GPU0."""
    from accelerate import init_empty_weights
    with init_empty_weights():
        meta = cls.from_config(cfg)
    nL = getattr(cfg, "text_config", cfg).num_hidden_layers
    path = next(n for n, mod in meta.named_modules() if isinstance(mod, torch.nn.ModuleList) and len(mod) == nL
                and n.endswith("layers") and "visual" not in n)
    n0 = max(1, round(nL / n_gpu) - gpu0_fewer)
    rest = nL - n0
    counts = [n0] + [rest // (n_gpu - 1) + (1 if i < rest % (n_gpu - 1) else 0) for i in range(n_gpu - 1)]
    gpu_of = [g for g, c in enumerate(counts) for _ in range(c)]
    dm = {}

    def walk(prefix, mod):
        for name, child in mod.named_children():
            full = f"{prefix}.{name}" if prefix else name
            if full == path:
                for i in range(nL):
                    dm[f"{full}.{i}"] = gpu_of[i]
            elif path.startswith(full + "."):
                walk(full, child)
            else:
                dm[full] = 0
    walk("", meta)
    uncovered = [n for n, _ in meta.named_parameters() if not any(n == k or n.startswith(k + ".") for k in dm)]
    assert not uncovered, f"device map misses {uncovered[:5]}"
    return dm


def gather_layer_outputs(layers, device="cuda:0"):
    """Register FIRST (before any analysis hook): every block output is moved to `device`, so hooks and
    downstream code see one device; the next block's accelerate pre-hook moves it back to its GPU."""
    def hook(_m, _i, out):
        if torch.is_tensor(out):
            return out.to(device)
        return (out[0].to(device),) + tuple(out[1:])
    return [layer.register_forward_hook(hook) for layer in layers]


# ---------------------------------------------------------------- MoE: frozen routing
# jvp differentiates the network with every token's top-k expert set held fixed (top-k is piecewise
# constant). A finite step can flip a token's expert set: a jump that no Taylor term sees, and whose
# contribution to a second difference diverges as eps -> 0. ROUTE lets a finite-difference pass record the
# base point's expert choices ("record") and replay them ("replay"): that is exactly the function jvp
# differentiates, so frozen FD -> jvp validates the AD, and actual - frozen isolates expert switching.
ROUTE = {"mode": None, "store": {}}


def patch_routers(hf) -> int:
    n = 0
    for mod in hf.modules():
        if not type(mod).__name__.endswith("TopKRouter") or getattr(mod, "_dj_patched", False):
            continue
        orig = mod.forward

        def fwd(hidden_states, _m=mod, _orig=orig):
            if ROUTE["mode"] is None:
                return _orig(hidden_states)
            hs = hidden_states.reshape(-1, _m.hidden_dim)
            logits = torch.nn.functional.linear(hs, _m.weight)
            probs = torch.softmax(logits, dtype=torch.float, dim=-1)
            if ROUTE["mode"] == "record":
                idx = torch.topk(probs, _m.top_k, dim=-1).indices
                ROUTE["store"][id(_m)] = idx
            else:
                idx = ROUTE["store"][id(_m)]
            val = probs.gather(-1, idx)
            val = val / val.sum(-1, keepdim=True)
            return logits, val.to(logits.dtype), idx
        mod.forward = fwd
        mod._dj_patched = True
        n += 1
    return n


@contextlib.contextmanager
def routing(mode):
    old = ROUTE["mode"]
    ROUTE["mode"] = mode
    try:
        yield
    finally:
        ROUTE["mode"] = old
