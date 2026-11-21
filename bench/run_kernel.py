"""Experiment F: the Metal paged-attention decode kernel vs the PyTorch gather + SDPA path.

Qwen2.5-0.5B, fp16, Apple GPU. B sequences of a given context length sit in the paged pool, scattered
across shuffled blocks the way a long-running server leaves them. We time one decode step of the
whole model (every layer, every request gets one new token) and, separately, the attention call of
one layer, for each backend.

    python -m bench.run_kernel
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import snapshot_download

from bench.env import describe
from kiln.attention.metadata import build_metadata
from kiln.attention.torch_paged import PagedContext
from kiln.cache import BlockManager
from kiln.cache.paged import PagedKVCache
from kiln.model.loader import load_model

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
RESULTS = Path(__file__).parent / "results" / "kernel.json"
BLOCK = 16
CONFIGS = [(1, 128), (1, 512), (1, 1024), (1, 2048), (8, 512), (8, 2048), (32, 256), (32, 512), (32, 1024)]


def timed(fn, reps: int) -> float:
    fn()
    torch.mps.synchronize()
    start = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.mps.synchronize()
    return (time.perf_counter() - start) / reps * 1000


def measure(model: Any, config: Any, batch: int, ctx: int) -> dict[str, Any]:
    need = batch * -(-ctx // BLOCK)
    bm = BlockManager(need + 8, BLOCK)
    bm._free = random.Random(0).sample(range(need + 8), need + 8)  # scatter blocks like a busy pool
    cache = PagedKVCache(config, need + 8, BLOCK, torch.float16, "mps")
    for pool in (*cache.k, *cache.v):
        pool.normal_()
    for b in range(batch):
        bm.allocate(b, ctx)
    meta = build_metadata(bm, list(range(batch)), [1] * batch, "mps")
    ids = torch.randint(100, 1000, (batch,), device="mps")
    pos = torch.full((batch,), ctx - 1, device="mps")
    q = torch.randn(batch, config.num_heads, config.head_dim, device="mps", dtype=torch.float16)
    kv = torch.randn(batch, config.num_kv_heads, config.head_dim, device="mps", dtype=torch.float16)
    row: dict[str, Any] = {"batch": batch, "context": ctx}
    for backend in ["torch", "metal"]:
        row[f"step_ms_{backend}"] = timed(lambda b=backend: model(ids, pos, PagedContext(cache, meta, b)), 10)
        row[f"attn_ms_{backend}"] = timed(lambda b=backend: PagedContext(cache, meta, b).attend(0, q, kv, kv), 50)
    return row


def main() -> None:
    torch.set_grad_enabled(False)
    model, config = load_model(Path(snapshot_download(MODEL, allow_patterns=["*.json", "*.safetensors"])),
                               torch.float16, "mps")
    out: dict[str, Any] = {"meta": {"env": describe(), "model": MODEL, "dtype": "fp16", "block_size": BLOCK},
                           "runs": []}
    for batch, ctx in CONFIGS:
        row = measure(model, config, batch, ctx)
        out["runs"].append(row)
        print(f"B={batch:3d} ctx={ctx:5d}: step {row['step_ms_torch']:6.1f} -> {row['step_ms_metal']:6.1f} ms "
              f"({row['step_ms_torch'] / row['step_ms_metal']:.2f}x); attention {row['attn_ms_torch']:6.2f} -> "
              f"{row['attn_ms_metal']:6.2f} ms ({row['attn_ms_torch'] / row['attn_ms_metal']:.1f}x)", flush=True)
        torch.mps.empty_cache()
    RESULTS.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
