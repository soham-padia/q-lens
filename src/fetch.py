"""Download HF model snapshots (IPv4 forced) and verify bytes on disk against the safetensors index.

Usage: python src/fetch.py Qwen/Qwen3.5-2B [repo@revision ...]   (exit 1 if any model is incomplete)
Remote-code files (*.py) are fetched too, for trust_remote_code models (Huginn, Ouro).
"""
import json
import os
import socket
import sys
from pathlib import Path

_orig = socket.getaddrinfo
socket.getaddrinfo = lambda *a, **k: [r for r in _orig(*a, **k) if r[0] == socket.AF_INET]  # IPv6 hung on the machine this was run on
from huggingface_hub import snapshot_download  # noqa: E402

bad = 0
for spec in sys.argv[1:]:
    repo, _, rev = spec.partition("@")
    d = Path(snapshot_download(repo, revision=rev or None,
                               allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja", "*.tiktoken", "*.py"]))
    st = sorted(d.glob("*.safetensors"))
    on_disk = sum(f.resolve().stat().st_size for f in st)
    idx = d / "model.safetensors.index.json"
    if idx.exists():
        meta = json.load(open(idx))
        want_files = sorted(set(meta["weight_map"].values()))
        missing = [f for f in want_files if not (d / f).exists()]
        ok = not missing and on_disk >= meta.get("metadata", {}).get("total_size", 0)
    else:
        missing, ok = [], len(st) == 1 and on_disk > 0
    print(f"{repo}: {len(st)} shards, {on_disk / 1e9:.2f} GB on disk, missing={missing}, ok={ok}", flush=True)
    bad += not ok
sys.exit(1 if bad else 0)
