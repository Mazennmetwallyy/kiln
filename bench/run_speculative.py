"""Experiment E: speculative decoding, Qwen2.5-0.5B drafting for Qwen2.5-1.5B, fp16 on the Mac GPU.

12 ShareGPT prompts, greedy, up to 128 new tokens each. Batch 1 (one request at a time: the latency
case speculative decoding is for) and batch 4. k = 0 is the target alone through the same engine.

    python -m bench.run_speculative --data bench/data/sharegpt_64_seed0.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import snapshot_download

from bench.data import load
from bench.env import describe
from kiln.engine.engine import LLMEngine
from kiln.model.loader import load_model
from kiln.sampling import SamplingParams

TARGET, DRAFT = "Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-0.5B-Instruct"
RESULTS = Path(__file__).parent / "results" / "speculative.json"
N_PROMPTS, MAX_NEW = 12, 128


def run(target, tconfig, draft, dconfig, prompts, out_lens, k: int, batch: int) -> tuple[dict, list[list[int]]]:
    kw: dict[str, Any] = {"draft_model": draft, "draft_config": dconfig, "num_speculative_tokens": k} if k else {}
    engine = LLMEngine(target, tconfig, None, num_blocks=128, block_size=16, max_num_seqs=batch, **kw)
    outs: dict[str, list[int]] = {}
    torch.mps.synchronize()
    start = time.perf_counter()
    for lo in range(0, len(prompts), batch):
        for i in range(lo, min(lo + batch, len(prompts))):
            engine.add_request(str(i), prompts[i], SamplingParams(temperature=0, max_tokens=out_lens[i],
                                                                  ignore_eos=True))
            outs[str(i)] = []
        while engine.has_unfinished():
            for o in engine.step():
                outs[o.request_id] += o.new_token_ids
    torch.mps.synchronize()
    seconds = time.perf_counter() - start
    s = engine.stats()
    tokens = sum(out_lens)
    return ({"k": k, "batch": batch, "seconds": seconds, "tok_per_s": tokens / seconds,
             "acceptance": s.spec_accepted_tokens / s.spec_proposed_tokens if s.spec_proposed_tokens else None,
             "tokens_per_target_forward": tokens / s.num_target_forwards},
            [outs[str(i)] for i in range(len(prompts))])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--ks", nargs="+", type=int, default=[0, 2, 4, 6])
    p.add_argument("--batches", nargs="+", type=int, default=[1, 4])
    a = p.parse_args()
    torch.set_grad_enabled(False)
    requests = load(Path(a.data))[:N_PROMPTS]
    prompts = [r["prompt_ids"] for r in requests]
    out_lens = [min(r["output_len"], MAX_NEW) for r in requests]
    pat = ["*.json", "*.safetensors"]
    target, tconfig = load_model(Path(snapshot_download(TARGET, allow_patterns=pat)), torch.float16, "mps")
    draft, dconfig = load_model(Path(snapshot_download(DRAFT, allow_patterns=pat)), torch.float16, "mps")
    run(target, tconfig, draft, dconfig, prompts[:2], [8, 8], 2, 2)  # warm-up
    out: dict[str, Any] = {"meta": {"env": describe(), "target": TARGET, "draft": DRAFT, "dtype": "fp16",
                                    "device": "mps", "num_prompts": len(prompts), "output_tokens": sum(out_lens)},
                           "runs": []}
    for batch in a.batches:
        baseline: list[list[int]] = []
        for k in a.ks:
            r, outs = run(target, tconfig, draft, dconfig, prompts, out_lens, k, batch)
            if k == 0:
                baseline = outs
            same = sum(x == y for x, y in zip(outs, baseline, strict=True)) if baseline else None
            r["identical_to_target_alone"] = same
            out["runs"].append(r)
            acc = f"{r['acceptance']:.2f}" if r["acceptance"] is not None else "  - "
            print(f"batch {batch} k={k}: {r['tok_per_s']:6.1f} tok/s, acceptance {acc}, "
                  f"{r['tokens_per_target_forward']:.2f} tokens/target forward, "
                  f"identical {r['identical_to_target_alone']}/{len(prompts)}", flush=True)
            RESULTS.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
