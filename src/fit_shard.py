"""First-order J-lens fit on Qwen3.8-27B, one shard of the prompt set (official jlens.fit).

Shard i of N takes prompts[i::N] from the first --n-prompts corpus docs. Resumable: the
running sum is checkpointed every --ckpt-every prompts. Shards are merged on load
(second_order.py --n1-lens-dir takes the n_prompts-weighted mean of every shard*of*.pt in the directory).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import transformers

JLENS_DIR = os.environ.get("JLENS_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "jacobian-lens"))
sys.path.insert(0, JLENS_DIR)
import jlens  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen3.8-27B")
ap.add_argument("--corpus", default=os.path.join(os.environ.get("QLENS_OUT", "outputs"), "corpus/fineweb_edu_2000.jsonl"))
ap.add_argument("--out", default=os.path.join(os.environ.get("QLENS_OUT", "outputs"), "fit_v1"))
ap.add_argument("--shard", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
ap.add_argument("--n-shards", type=int, default=6)
ap.add_argument("--n-prompts", type=int, default=120)
ap.add_argument("--dim-batch", type=int, default=32)
ap.add_argument("--ckpt-every", type=int, default=5)
ap.add_argument("--target-layer", type=int, default=-1, help="-1 = final block (repo default)")
args = ap.parse_args()

out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
final = out / f"shard{args.shard:02d}of{args.n_shards:02d}.pt"
if final.exists():
    sys.exit(f"{final} exists; nothing to do")

prompts = [json.loads(l)["text"] for l in open(args.corpus)][: args.n_prompts][args.shard :: args.n_shards]
print(f"shard {args.shard}/{args.n_shards}: {len(prompts)} prompts, dim_batch={args.dim_batch}", flush=True)

tok = transformers.AutoTokenizer.from_pretrained(args.model)
hf = transformers.AutoModelForImageTextToText.from_pretrained(
    args.model, dtype=torch.bfloat16, attn_implementation="sdpa", device_map="cuda")
m = jlens.from_hf(hf, tok)
L = m.n_layers - 1 if args.target_layer < 0 else args.target_layer  # block output, pre-norm

t0 = time.perf_counter()
lens = jlens.fit(m, prompts, source_layers=list(range(L)), target_layer=L, dim_batch=args.dim_batch,
                 checkpoint_path=str(out / f"ckpt_shard{args.shard:02d}.pt"), checkpoint_every=args.ckpt_every)
lens.save(str(final))
print(f"saved {final}  n_prompts={lens.n_prompts}  fit_min={(time.perf_counter() - t0) / 60:.1f}", flush=True)
(out / f"shard{args.shard:02d}.json").write_text(json.dumps(
    {**vars(args), "n_prompts_fit": lens.n_prompts, "fit_min": round((time.perf_counter() - t0) / 60, 1)}))
