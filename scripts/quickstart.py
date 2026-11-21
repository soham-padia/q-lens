"""Quickstart: J and Q for one push on a small model, checked against the actual change.

    python scripts/quickstart.py [--model Qwen/Qwen3.5-0.8B] [--layer N] [--alpha 0.1] [--device cpu|cuda]

Pushes two random directions u of size rho (rho = mean residual norm of block `layer`'s output) at every valid
position (16..T-2) of that block's output, and reads F = the sum over valid positions of the last block's output,
as in the depth setting (D1) of src/second_order.py. J u and Q(u, u) come from nested forward-mode jvp
(so_core.first_and_second); no Hessian is formed. Prints |J u|, |Q(u, u)|, r = |Q|/|J|, and the relative error of
the first-order (alpha J u) and second-order (alpha J u + alpha^2 Q(u, u) / 2) predictions of the actual change
F(alpha u) - F(0). Takes about 15 s on 4 CPU cores for the default model, once the weights are downloaded.
"""
import argparse
import os
import sys

import torch
import transformers

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from so_core import Perturbed, first_and_second, math_sdpa, no_tf32  # noqa: E402  (puts $JLENS_DIR on sys.path)
import jlens  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
ap.add_argument("--layer", type=int, default=None, help="source block (default: n_layers // 2)")
ap.add_argument("--alpha", type=float, default=0.1, help="push size as a fraction of rho")
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
args = ap.parse_args()

cfg = transformers.AutoConfig.from_pretrained(args.model)
cls = (transformers.AutoModelForImageTextToText if "ConditionalGeneration" in (cfg.architectures or [""])[0]
       else transformers.AutoModelForCausalLM)
tok = transformers.AutoTokenizer.from_pretrained(args.model)
hf = cls.from_pretrained(args.model, dtype=torch.float32, attn_implementation="sdpa").to(args.device).eval()
m = jlens.from_hf(hf, tok)
L = m.n_layers - 1
layer = args.layer if args.layer is not None else m.n_layers // 2

text = ("The river runs through the old town, past the market square and under the stone bridge, and on towards "
        "the sea. In spring the water rises, the meadows along its banks turn green, and the ferry starts again.")
ids = tok(text, return_tensors="pt").input_ids.to(args.device)
f = Perturbed(m, ids.expand(2, -1), layer, L)  # batch of 2 = two directions on the same prompt

box = {}
hook = m.layers[layer].register_forward_hook(lambda _m, _i, o: box.__setitem__("h", o if torch.is_tensor(o) else o[0]))
with torch.no_grad(), math_sdpa():
    m.forward(ids)
hook.remove()
rho = float(box["h"][0, f.valid].norm(dim=-1).mean())

u = torch.randn(2, m.d_model, generator=torch.Generator().manual_seed(0)).to(args.device)
U = u / u.norm(dim=-1, keepdim=True) * rho
with no_tf32():
    Ju, _, Quu = first_and_second(f, U, None, signs=False)
    nV = len(f.valid)
    Ju, Quu = Ju * nV, Quu * nV  # first_and_second averages over the pushed positions; F sums them
    A1 = f.spread(U, None)
    with torch.no_grad(), math_sdpa():
        act = f(args.alpha * A1) - f(torch.zeros_like(A1))

a = args.alpha
lin, quad = a * Ju, a * Ju + 0.5 * a ** 2 * Quu
print(f"{args.model}: push at block {layer} of {m.n_layers}, read at block {L}, T = {ids.shape[1]}, "
      f"{nV} valid positions, rho = {rho:.1f}, alpha = {a}")
for i in range(2):
    an = act[i].norm()
    print(f"direction {i}: |J u| = {Ju[i].norm():.4g}  |Q(u,u)| = {Quu[i].norm():.4g}  r = {Quu[i].norm() / Ju[i].norm():.3f}  "
          f"rel. error J = {(act[i] - lin[i]).norm() / an:.4f}  J+Q/2 = {(act[i] - quad[i]).norm() / an:.4f}")
