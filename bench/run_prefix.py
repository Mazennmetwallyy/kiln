"""Experiment D: prefix caching on vs off, for requests that share a long system prompt.

Each of the 64 ShareGPT questions is re-templated behind the same ~400-token system prompt, the way
an assistant product sends every user message. All requests are submitted at once, in-process.

    python -m bench.run_prefix --data bench/data/sharegpt_64_seed0.json
"""

from __future__ import annotations

import argparse
import json
import statistics
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
from kiln.tokenizer import Tokenizer

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
RESULTS = Path(__file__).parent / "results" / "prefix_caching.json"

SYSTEM = """You are the support assistant for Northwind Cloud, a hosting provider for small software teams.

Scope. You help customers with: deploying applications, managing databases, configuring domains and TLS
certificates, reading logs, understanding invoices, and choosing a plan. For anything outside that list,
you may still answer general programming or writing questions briefly, but say that it is outside
Northwind support and that the answer is best-effort.

Style. Be direct and concrete. Lead with the answer, then the steps. Use numbered steps for procedures
and code blocks for commands and configuration. Keep answers under 300 words unless the customer asks
for more detail. Do not use marketing language. Do not apologise more than once per conversation.

Accuracy. If you are not sure, say so and suggest how the customer can check. Never invent command
flags, API fields, prices, limits or error codes. Prices and limits change, so point customers to the
pricing page instead of quoting numbers. When a question is ambiguous, state which interpretation you
are answering and offer to answer the other one.

Safety and privacy. Never ask for passwords, API keys or full card numbers. If a customer pastes a
secret, tell them to rotate it immediately and explain how. Do not discuss other customers' accounts.
Do not help with anything that would break the acceptable-use policy: sending spam, scanning networks
the customer does not own, hosting malware, or evading bans.

Escalation. Hand over to a human engineer, and say that you are doing so, when: the customer reports
data loss, a security incident, a billing dispute over 100 euros, an outage affecting production for
more than 15 minutes, or when they ask for a human. Include a one-paragraph summary of the issue for
the engineer.

Formatting. Write in British English. Use metric units and 24-hour times in UTC. Refer to the product
areas by their names in the dashboard: Apps, Databases, Domains, Logs, Billing and Team."""


def build_prompts(tok: Tokenizer, requests: list[dict]) -> list[list[int]]:
    out = []
    for r in requests:
        text = tok.decode(r["prompt_ids"])  # special tokens are dropped: "system\n...\nuser\n<question>\nassistant\n"
        user = text.split("\nuser\n", 1)[1].rsplit("\nassistant\n", 1)[0]
        chat = tok.apply_chat_template([{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}])
        out.append(tok.encode(chat, add_special_tokens=False))
    return out


def run(model, config, prompts: list[list[int]], out_lens: list[int], caching: bool) -> dict:
    engine = LLMEngine(model, config, None, num_blocks=3072, block_size=16, max_num_seqs=64,
                       max_batched_tokens=512, enable_prefix_caching=caching)
    for i, (p, n) in enumerate(zip(prompts, out_lens, strict=True)):
        engine.add_request(str(i), p, SamplingParams(temperature=0, max_tokens=n, ignore_eos=True))
    ttft: dict[str, float] = {}
    torch.mps.synchronize()
    start = time.perf_counter()
    while engine.has_unfinished():
        for o in engine.step():
            if o.new_token_ids and o.request_id not in ttft:
                ttft[o.request_id] = time.perf_counter() - start
    torch.mps.synchronize()
    seconds = time.perf_counter() - start
    s = engine.stats()
    return {"caching": caching, "seconds": seconds, "output_throughput": sum(out_lens) / seconds,
            "ttft_mean": statistics.mean(ttft.values()), "ttft_p50": statistics.median(ttft.values()),
            "prompt_tokens": s.prefix_cache_query_tokens, "cached_tokens": s.prefix_cache_hit_tokens,
            "prefill_tokens_computed": s.prefix_cache_query_tokens - s.prefix_cache_hit_tokens}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    a = p.parse_args()
    torch.set_grad_enabled(False)
    path = Path(snapshot_download(MODEL, allow_patterns=["*.json", "*.safetensors", "*.jinja"]))
    tok = Tokenizer.from_hf_dir(path)
    requests = load(Path(a.data))
    prompts = build_prompts(tok, requests)
    out_lens = [r["output_len"] for r in requests]
    system_tokens = len(tok.encode(tok.apply_chat_template([{"role": "system", "content": SYSTEM}]),
                                   add_special_tokens=False))
    model, config = load_model(path, dtype=torch.float16, device="mps")
    run(model, config, prompts[:4], [8] * 4, caching=True)  # warm-up
    out: dict[str, Any] = {"meta": {"env": describe(), "model": MODEL, "dtype": "fp16", "num_requests": len(prompts),
                    "system_prompt_tokens": system_tokens,
                    "mean_prompt_tokens": sum(map(len, prompts)) / len(prompts)}, "runs": []}
    for caching in (False, True):
        r = run(model, config, prompts, out_lens, caching)
        out["runs"].append(r)
        print(f"caching={caching!s:5}: {r['seconds']:.1f}s, {r['output_throughput']:.0f} tok/s, "
              f"TTFT mean {r['ttft_mean']:.2f}s p50 {r['ttft_p50']:.2f}s, prefill computed "
              f"{r['prefill_tokens_computed']} of {r['prompt_tokens']} prompt tokens", flush=True)
    RESULTS.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
