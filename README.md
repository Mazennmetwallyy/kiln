# Kiln

An LLM inference engine built from scratch in Python and PyTorch. It has its own transformer implementation, and the roadmap adds a paged KV cache, continuous batching, prefix caching, speculative decoding and a hand-written Metal attention kernel, served behind an OpenAI-compatible API.

## Status

| Milestone | | |
|---|---|---|
| M1 Model: own Qwen2/Llama forward pass, sampling, streaming, CLI | ✅ | |
| M2 Paged KV cache: block manager, copy-on-write, paged attention | ✅ | |
| M3 Continuous batching | ⏳ | |
| M4 OpenAI-compatible server | | |
| M5 Benchmarks | | |
| M6 Prefix caching · M7 Speculative decoding · M8 Metal kernel · M9 Visualiser | | |

**M1 results** (Apple M2, 8 GB):
- Greedy output is **token-for-token identical to Hugging Face `transformers`** on 12/12 test prompts for both Qwen2.5-0.5B-Instruct and SmolLM2-135M-Instruct (fp32, 64 tokens each). Logits match exactly.
- In fp16 on the Mac GPU (MPS), the top-1 token agrees on every prompt (worst logit difference 0.064).
- Single-request decode, Qwen2.5-0.5B, fp16, MPS: 32.0 tokens/s on the paged cache (23.5 in M1, before GPU syncs were removed from the attention layers). That's the unbatched baseline M3 improves on.

## Quickstart

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/kiln generate --model Qwen/Qwen2.5-0.5B-Instruct --chat --prompt "Explain a KV cache in one sentence." --device mps --dtype fp16
```

Tests: `pytest` (fast, CPU, no downloads). `pytest --runslow --runmps tests/integration -s` runs the real-model checks.

## How it works

See [`docs/how-it-works.md`](docs/how-it-works.md) and the design decisions in [`docs/decisions/`](docs/decisions/).

## Credits

The ideas implemented here come from: Kwon et al., *Efficient Memory Management for LLM Serving with PagedAttention* (vLLM, 2023); Yu et al., *Orca* (2022); Leviathan et al., *Fast Inference from Transformers via Speculative Decoding* (2023); Zheng et al., *SGLang / RadixAttention* (2024); Su et al., *RoFormer* (2021).
