"""Experiment B: how many requests fit in a fixed KV-cache budget.

Kiln (paged) is measured: all requests are submitted at once and we record the peak number running
together. Contiguous allocation is computed from the same workload: a server that reserves each
request's worst case up front (prompt + max_tokens), or the full context window.

    python -m bench.run_memory --data bench/data/sharegpt_64_seed0.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from huggingface_hub import snapshot_download

from bench.data import load
from bench.env import describe
from kiln.cache.paged import PagedKVCache
from kiln.engine.engine import LLMEngine
from kiln.model.loader import load_model
from kiln.sampling import SamplingParams

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
RESULTS = Path(__file__).parent / "results" / "memory.json"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--budget-mb", type=int, default=64)
    p.add_argument("--context", type=int, default=2048, help="context window a contiguous server reserves")
    a = p.parse_args()
    reqs = load(Path(a.data))
    torch.set_grad_enabled(False)

    path = Path(snapshot_download(MODEL, allow_patterns=["*.json", "*.safetensors"]))
    model, config = load_model(path, dtype=torch.float16, device="mps")
    budget = a.budget_mb * 2**20
    per_token = PagedKVCache.bytes_per_block(config, 16, torch.float16) // 16
    num_blocks = PagedKVCache.blocks_for_memory(config, budget, 16, torch.float16)

    engine = LLMEngine(model, config, None, num_blocks=num_blocks, block_size=16, max_num_seqs=1024,
                       max_batched_tokens=8192)
    for i, r in enumerate(reqs):
        engine.add_request(str(i), r["prompt_ids"], SamplingParams(temperature=0, max_tokens=r["output_len"],
                                                                   ignore_eos=True))
    peak, tokens, start = 0, 0, time.perf_counter()
    while engine.has_unfinished():
        tokens += sum(len(o.new_token_ids) for o in engine.step())
        peak = max(peak, engine.stats().num_running)
    elapsed = time.perf_counter() - start

    worst_case = [len(r["prompt_ids"]) + r["output_len"] for r in reqs]
    fit_exact, used = 0, 0
    for need in worst_case:  # admit in arrival order until the next reservation doesn't fit
        if used + need * per_token > budget:
            break
        used += need * per_token
        fit_exact += 1
    result = {
        "meta": {"env": describe(), "model": MODEL, "dtype": "fp16", "budget_mb": a.budget_mb,
                 "bytes_per_token": per_token, "budget_tokens": budget // per_token, "num_requests": len(reqs),
                 "context_reserved": a.context},
        "kiln_paged": {"peak_concurrent": peak, "preemptions": engine.stats().num_preemptions_total,
                       "output_tokens": tokens, "seconds": elapsed},
        "contiguous_reserve_request_max": fit_exact,
        "contiguous_reserve_context": budget // (a.context * per_token),
    }
    RESULTS.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
