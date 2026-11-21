"""Experiment C: generation throughput against batch size, Kiln vs Hugging Face static batching.

Each run generates 64 tokens for B identical-length prompts (128 tokens), timing prefill and decode
together, in-process on the same model and device.

    python -m bench.run_decode_scaling
"""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM

from bench.env import describe
from bench.hf_server import static_batch_generate
from kiln.engine.engine import LLMEngine
from kiln.model.loader import load_model
from kiln.sampling import SamplingParams

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
RESULTS = Path(__file__).parent / "results" / "decode_scaling.json"
BATCHES = [1, 2, 4, 8, 16, 32, 64]
PROMPT, NEW = 128, 64


def prompts(b: int) -> list[list[int]]:
    return [[(31 * i + 7 * j) % 5000 + 100 for j in range(PROMPT)] for i in range(b)]


def kiln_run(model, config, b: int) -> float:
    engine = LLMEngine(model, config, None, num_blocks=2048, block_size=16, max_num_seqs=128,
                       max_batched_tokens=PROMPT * b)
    for i, p in enumerate(prompts(b)):
        engine.add_request(str(i), p, SamplingParams(temperature=0, max_tokens=NEW, ignore_eos=True))
    torch.mps.synchronize()
    start = time.perf_counter()
    while engine.has_unfinished():
        engine.step()
    torch.mps.synchronize()
    return b * NEW / (time.perf_counter() - start)


def hf_run(hf, b: int) -> float:
    torch.mps.synchronize()
    start = time.perf_counter()
    static_batch_generate(hf, prompts(b), [NEW] * b, pad_id=0)
    torch.mps.synchronize()
    return b * NEW / (time.perf_counter() - start)


def main() -> None:
    torch.set_grad_enabled(False)
    path = Path(snapshot_download(MODEL, allow_patterns=["*.json", "*.safetensors"]))
    out: dict = {"meta": {"env": describe(), "model": MODEL, "dtype": "fp16", "prompt_tokens": PROMPT,
                          "new_tokens": NEW}, "kiln": {}, "hf_static": {}}
    model, config = load_model(path, dtype=torch.float16, device="mps")
    kiln_run(model, config, 2)  # warm-up
    for b in BATCHES:
        out["kiln"][b] = kiln_run(model, config, b)
        print(f"kiln b={b}: {out['kiln'][b]:.0f} tok/s", flush=True)
    del model
    gc.collect()
    torch.mps.empty_cache()
    hf: Any = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float16)
    hf = hf.to("mps").eval()
    hf_run(hf, 2)
    for b in BATCHES:
        out["hf_static"][b] = hf_run(hf, b)
        print(f"hf   b={b}: {out['hf_static'][b]:.0f} tok/s", flush=True)
    RESULTS.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
