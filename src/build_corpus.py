"""Build the pretraining-like prompt corpus for J-lens fitting.

Streams HuggingFaceFW/fineweb-edu (sample-10BT) with a seeded shuffle buffer and keeps
documents long enough to fill 128 tokens. The paper used 1000 x 128-token sequences from a
pretraining-like distribution; we save more so the remainder can serve as held-out text.

Needs internet access, no GPU.
"""
import argparse
import hashlib
import json
import os
import socket
from pathlib import Path

# Force IPv4 (IPv6 to huggingface.co hung on the machine this was run on).
_orig_getaddrinfo = socket.getaddrinfo
socket.getaddrinfo = lambda *a, **k: [r for r in _orig_getaddrinfo(*a, **k) if r[0] == socket.AF_INET]

from datasets import load_dataset  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=2000)
ap.add_argument("--min-chars", type=int, default=1500)
ap.add_argument("--max-chars", type=int, default=3000)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default=os.path.join(os.environ.get("QLENS_OUT", "outputs"), "corpus/fineweb_edu_2000.jsonl"))
args = ap.parse_args()

ds = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True)
ds = ds.shuffle(seed=args.seed, buffer_size=20_000)

rows = []
for ex in ds:
    text = ex["text"]
    if len(text) >= args.min_chars:
        rows.append({"id": ex.get("id"), "text": text[: args.max_chars]})
        if len(rows) >= args.n:
            break

out = Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
digest = hashlib.sha256(out.read_bytes()).hexdigest()
print(f"wrote {len(rows)} docs -> {out}  sha256={digest}")
