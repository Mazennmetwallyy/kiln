# How Kiln works

An interview-prep guide to every part of Kiln. Each milestone adds a section.

---

## M1: The model

### What a decoder-only transformer does, one token at a time

A language model maps a sequence of token ids to a probability distribution over the next token. Kiln's model (`kiln/model/`) is the Qwen2/Llama architecture:

```
token ids ──► embedding ──► [ decoder layer ] × N ──► final RMSNorm ──► lm_head ──► logits (one score per vocab entry)

decoder layer:   x = x + Attention(RMSNorm(x))
                 x = x + SwiGLU(RMSNorm(x))
```

This is a **pre-norm residual** design: each block reads a normalised copy of the residual stream `x` and adds its result back. The residual path keeps gradients healthy in training, and in inference it means each layer makes a small edit to a running representation.

For Qwen2.5-0.5B: N = 24 layers, hidden size 896, 14 query heads, 2 key/value heads, head dim 64, vocabulary 151,936. The `lm_head` shares its weights with the embedding table ("tied embeddings"). That's why the checkpoint has no `lm_head.weight`, and why the loader points one at the other.

### RMSNorm

`y = x / sqrt(mean(x²) + eps) * g`. It's LayerNorm without subtracting the mean and without the bias. It's cheaper and works just as well in practice. Kiln computes the statistic in fp32 even when the model runs in fp16, because `x²` can overflow fp16's range (max ~65,504).

### RoPE: rotary position embeddings

Attention on its own has no sense of order. RoPE fixes this by **rotating** each query and key vector by an angle proportional to its position. The dimensions are split into pairs, and pair *i* rotates at frequency θ^(−2i/d) (θ = 1,000,000 for Qwen2.5, 100,000 for SmolLM2).

Why it works: rotating q by angle *mα* and k by *nα* gives `q·k = |q||k| cos(φ + (m−n)α)`. The score depends only on the **distance** m − n, not on absolute positions. `tests/model/test_layers.py::test_rope_relative` checks exactly this: positions (3, 1) give the same score as (12, 10). A large θ means the slowest pairs barely rotate, so far-apart tokens can still be told apart. That's why long-context models use big θ values.

Hugging Face rotates dimension *i* with dimension *i + d/2* (the "rotate_half" layout), not adjacent pairs. Get this wrong and every output is garbage. The test against HF's own `apply_rotary_pos_emb` catches it.

### Attention and GQA

Each head computes `softmax(q·kᵀ / √d) · v` over all earlier tokens (the causal mask stops a token seeing the future).

**Grouped-query attention (GQA):** Qwen2.5-0.5B has 14 query heads but only **2** key/value heads. Each group of 7 query heads shares one K/V head. Quality barely changes, but the KV cache (below) gets **7× smaller**:

| | per token (fp16) | 32k-token context |
|---|---|---|
| full multi-head (14 KV heads) | 2 × 24 layers × 14 × 64 × 2 bytes = **86 KB** | 2.7 GB |
| GQA (2 KV heads) | 2 × 24 × 2 × 64 × 2 bytes = **12 KB** | 384 MB |

On an 8 GB laptop that difference decides how many users you can serve at once. It's why M2's paged cache matters.

### The KV cache

Generating token *t+1* needs attention over tokens 1…t. Without a cache, every step recomputes K and V for the whole sequence: O(n²) work to generate n tokens. The K and V for past tokens never change, so Kiln stores them (`kiln/model/kv_cache.py`) and each new token computes only its own q, k, v, then attends over the stored ones. That's O(n) per step.

Generation therefore has two phases with very different costs:
- **Prefill:** process the whole prompt in one forward pass. It's compute-bound, with big matrix multiplies.
- **Decode:** one token per forward pass. It's **memory-bandwidth-bound**: every step reads all the weights and the whole cache to produce one token. Batching many requests' decode steps together (M3) is how servers get efficient.

The attention mask for "T new tokens after a cached prefix" is `key_position ≤ query_position`, built from explicit positions. That's why prefill and decode share one code path (see ADR 0001).

### SwiGLU MLP

`down(silu(gate(x)) ⊙ up(x))`. The `gate` branch decides, element by element, how much of the `up` branch passes through. It's a learned gate that beats a plain ReLU MLP for the same parameter count.

### Sampling

The logits become one token in `kiln/sampling/sampler.py`:
1. **Temperature 0 → greedy:** take the argmax.
2. **Temperature T:** divide the logits by T. Below 1 sharpens the distribution, above 1 flattens it.
3. **Top-k:** keep only the k highest-scoring tokens.
4. **Top-p (nucleus):** sort by probability and keep the smallest set whose total probability reaches p. Top-p adapts: it keeps few tokens when the model is confident and many when it's unsure, which top-k can't do.
5. Draw from what's left with a per-request seeded generator, so results are reproducible.

### Streaming text correctly

Tokens aren't characters. With byte-level BPE, one emoji like 👋🏽 is 8 UTF-8 bytes split across several tokens, and decoding any one of them alone gives the replacement character `�`. `IncrementalDetokenizer` re-decodes a small window of recent tokens and releases text only once it no longer ends in `�`.

Stop strings add a second wrinkle. If the stop string is `"o w"` and the model emits `"Hello"` then `" world"`, the text `"o"` must be **held back** after the first token, because it might be the start of the stop string. `generate.py` keeps back the longest suffix that could begin a stop string.

### How we know it's correct

- Every layer is checked against Hugging Face's implementation.
- Tiny randomly initialised models (built in the tests, so CI needs no downloads) must match HF logits to 1e-5.
- On the **real** Qwen2.5-0.5B-Instruct and SmolLM2-135M-Instruct, 64 greedy tokens for each of 12 varied prompts (code, maths, Chinese, emoji) are **token-for-token identical** to `transformers`. Kiln's fp32 logits matched HF's with a maximum difference of exactly 0.0.
- In fp16 on the Mac GPU, the worst logit difference is 0.064, and the top-1 token always agrees.

**A bug story worth telling in an interview.** The first real-model run "failed" at token 12 of one prompt. Kiln's logits were bit-identical to HF's forward pass. The HF *reference* was wrong: Qwen's `generation_config.json` sets `repetition_penalty: 1.1`, and HF applies it even with `do_sample=False`. The fix was in the test harness. The lesson: **check the reference before assuming your own code is wrong.**

### Questions to be ready for
- Why does a KV cache make decoding O(n) per token instead of O(n²)?
- How much smaller is Qwen2.5-0.5B's KV cache because of GQA, and why does that matter for serving?
- Why does RoPE make attention depend on relative position?
- Why is decode memory-bandwidth-bound while prefill is compute-bound?
- Top-p vs top-k: when does each behave badly?
- Why can't you stream text by decoding one token at a time?
