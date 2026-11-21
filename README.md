# Kiln

An LLM inference engine built from scratch in Python and PyTorch. It has its own transformer implementation, and the roadmap adds a paged KV cache, continuous batching, prefix caching, speculative decoding and a hand-written Metal attention kernel, served behind an OpenAI-compatible API.

## Status

| Milestone | | |
|---|---|---|
| M1 Model: own Qwen2/Llama forward pass, sampling, streaming, CLI | ✅ | |
| M2 Paged KV cache: block manager, copy-on-write, paged attention | ✅ | |
| M3 Continuous batching: iteration-level scheduler, chunked prefill, preemption | ✅ | |
| M4 OpenAI-compatible server: SSE streaming, cancellation, Prometheus metrics | ✅ | |
| M5 Benchmarks | ⏳ | |
| M6 Prefix caching · M7 Speculative decoding · M8 Metal kernel · M9 Visualiser | | |

**M1 results** (Apple M2, 8 GB):
- Greedy output is **token-for-token identical to Hugging Face `transformers`** on 12/12 test prompts for both Qwen2.5-0.5B-Instruct and SmolLM2-135M-Instruct (fp32, 64 tokens each). Logits match exactly.
- **Batching doesn't change outputs:** all 12 prompts submitted to the engine at once (ragged prefill + batched decode) are still token-for-token identical to Hugging Face for both models.
- In fp16 on the Mac GPU (MPS), the top-1 token agrees on every prompt (worst logit difference 0.064).
- Single-request decode, Qwen2.5-0.5B, fp16, MPS: 32.0 tokens/s on the paged cache (23.5 in M1, before GPU syncs were removed from the attention layers). That's the unbatched baseline M3 improves on.

## Quickstart

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/kiln generate --model Qwen/Qwen2.5-0.5B-Instruct --chat --prompt "Explain a KV cache in one sentence." --device mps --dtype fp16
```

Serve it with an OpenAI-compatible API and call it from any OpenAI client:

```bash
.venv/bin/kiln serve --model Qwen/Qwen2.5-0.5B-Instruct --port 8000   # Apple GPU, fp16, 1 GB KV pool
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
stream = client.chat.completions.create(model="qwen", messages=[{"role": "user", "content": "Hi!"}], stream=True)
for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="") if chunk.choices else None
```

Endpoints: `/v1/completions`, `/v1/chat/completions` (streaming and non-streaming), `/v1/models`, `/health`, `/metrics`.

Tests: `pytest` (fast, CPU, no downloads). `pytest --runslow --runmps tests/integration -s` runs the real-model checks.

## How it works

See [`docs/how-it-works.md`](docs/how-it-works.md) and the design decisions in [`docs/decisions/`](docs/decisions/).

## Credits

The ideas implemented here come from: Kwon et al., *Efficient Memory Management for LLM Serving with PagedAttention* (vLLM, 2023); Yu et al., *Orca* (2022); Leviathan et al., *Fast Inference from Transformers via Speculative Decoding* (2023); Zheng et al., *SGLang / RadixAttention* (2024); Su et al., *RoFormer* (2021).
