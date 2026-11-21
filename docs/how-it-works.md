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

---

## M2: The paged KV cache

### The problem: reserving memory you might never use

A simple server gives each request one contiguous KV buffer sized for the **maximum** length it might reach, because the cache can't easily grow in place. Most requests stop far short of the maximum, so most of that memory sits reserved and unused. The vLLM paper measured only 20–38% of KV memory actually holding tokens in such systems. On an 8 GB laptop, that waste directly limits how many people you can serve at once.

### The fix: paging, borrowed from operating systems

Kiln splits KV memory into fixed-size **blocks** of 16 tokens (`kiln/cache/`):

```
physical pool (one flat tensor per layer):  [blk0][blk1][blk2][blk3][blk4][blk5] ...
sequence A, 40 tokens -> block table [0, 3, 4]   (16 + 16 + 8 used)
sequence B, 10 tokens -> block table [1]
free list: [2, 5, ...]
```

- A sequence's **block table** maps its logical positions to physical blocks, just as a page table maps virtual pages to physical frames. Blocks need not be next to each other.
- Memory is allocated one block at a time as a sequence grows. The waste is at most 15 token slots per sequence.
- Token *p* of a sequence lives at **slot** `table[p // 16] * 16 + p % 16`. Each step, `slot_mapping` tells attention where to write the new tokens' K and V.

### Sharing: reference counts and copy-on-write

Blocks are **reference counted**, so several sequences can point at the same block:
- `fork(parent, child)` lets the child share all of the parent's blocks. This is free: it's just refcount + 1. M6 uses the same mechanism to share a common system prompt across many requests.
- If a sequence wants to write into a **shared, partly filled** last block, the block manager gives it a fresh block and asks the caller to copy the old one first (**copy-on-write**). Full shared blocks are never written again, so they stay shared.
- `rollback(n)` forgets the last *n* tokens and returns emptied blocks. Speculative decoding (M7) needs this to discard rejected guesses.

The block manager does bookkeeping only. It never touches tensors, which makes it easy to test. A Hypothesis *stateful* test throws random sequences of allocate/append/fork/rollback/free at it, 300 times, and checks after every step that:
- refcounts equal the number of block tables pointing at each block;
- every table has exactly `ceil(len / 16)` blocks;
- free + used = total;
- freeing everything returns every block.

### Attention over paged memory

`PagedContext` (`kiln/attention/torch_paged.py`) works in two steps:
1. **Write:** scatter the new K/V into the pool at `slot_mapping`.
2. **Read:**
   - **Prefill, or a mix of prefill and decode:** for each sequence, gather its slots for positions 0…len−1 and run causal attention, where the query at position *p* sees keys ≤ *p*.
   - **Pure decode:** every sequence adds one token, so gather all sequences' K/V into one padded tensor with a length mask and run **one** batched attention call.

The tests check paged attention against the simple contiguous cache in five cases:
- a single sequence crossing many blocks;
- two prompts in one flat batch (any mask leak between sequences would show up here);
- one sequence decoding while another prefills;
- a batch of decodes;
- two forked sequences writing different tokens after copy-on-write.

### Why lengths are Python ints

Asking a GPU tensor for a value (`int(tensor)`) makes the CPU wait for the GPU to finish everything queued so far. M1 did this twice per layer per token. M2 keeps all lengths as plain Python ints in the step's metadata and builds gather indices once per step. Single-request decode on the Mac GPU went from 23.5 to **32.0 tokens/s** from that change alone. Output stayed token-for-token identical to Hugging Face on all 24 real-model tests.

### Questions to be ready for
- Why does contiguous per-request allocation waste memory, and how much does paging waste at most?
- What's in a block table, and how do you find the slot for token *p*?
- Why does writing into a shared block need copy-on-write, and why only for the *last, partly filled* block?
- What does `rollback` do, and which feature needs it?
- Why does `int(gpu_tensor)` hurt performance?

---

## M3: Continuous batching

### Why batch at all

Decoding one token means reading **every weight** of the model from memory. That's about 1 GB for Qwen2.5-0.5B in fp16, read to produce a single token. The GPU's arithmetic units mostly sit idle waiting for memory. If 16 requests decode together, the weights are read once and used 16 times. The cost per step hardly changes, but you get 16 tokens instead of 1. Batching is the single biggest lever in LLM serving.

### Static vs continuous batching

**Static batching** collects a batch, pads every prompt to the same length, and runs until the **slowest** request finishes. Requests that finish early leave empty slots that do nothing, and new requests wait for the whole batch.

**Continuous (iteration-level) batching**, from the Orca paper, makes the scheduling decision **every step**:

```
step 1:  [A prefill 40 tok][B prefill 12 tok]
step 2:  [A][B]                                <- both decoding, 1 token each
step 3:  [A][B][C prefill 30 tok]               <- C joins mid-flight
step 4:  [A][C]                                 <- B finished, its slot is reused at once
```

Kiln's engine (`kiln/engine/engine.py`) loops: **schedule → build one flat batch → one forward pass → sample → update**. The flat batch mixes prompt chunks and single decode tokens from any number of requests. That's exactly what the flat 1-D forward (ADR 0001) and the paged attention backend (ADR 0002) were built for.

### The scheduler's rules (`kiln/engine/scheduler.py`)

1. **Running requests first, oldest first.** Each gets its next token, or its next prompt chunk, within the step's **token budget** (default 512).
2. **Chunked prefill.** A 2,000-token prompt doesn't get processed in one giant step that freezes everyone else's output. It's split across steps to fit the budget, so streaming users keep getting tokens.
3. **Strict first-come, first-served admission.** If the request at the front of the queue doesn't fit yet, smaller ones behind it don't jump ahead. That would starve big requests forever.
4. **Preemption when memory runs out.** If a running request needs a new block and none are free, the **most recently admitted** request is kicked out. Its blocks are freed, and it goes back to the front of the queue to recompute later (ADR 0003 explains why recompute rather than swap on a Mac). Newest-first preemption means the oldest requests always make progress, so the system can't livelock.
5. **Requests that could never fit are rejected up front.** If prompt + max_tokens exceeds the whole pool, there's an immediate error rather than a deadlock.

### Correctness under batching

Batching changes which matrices get multiplied together, so it's a classic source of subtle bugs. Kiln's tests check that **every request produces the same tokens it would produce alone**:
- 8 requests of different lengths in one engine, compared against Hugging Face;
- the same with a deliberately tiny pool that forces preemptions;
- with a 7-token budget that forces chunked prefill;
- a request added while others are mid-generation;
- seeded random sampling, which gives identical tokens batched or alone because each request owns its own random number generator.

A Hypothesis test also runs 150 random workloads through the scheduler. It checks the budget and sequence limits on every step, that every request finishes, and that no KV block leaks.

On the real models, **all 12 test prompts submitted at once** produce output token-for-token identical to Hugging Face for both Qwen2.5-0.5B and SmolLM2-135M.

### Smaller details that matter

- **Only compute the logits you need.** A prefill of 500 tokens doesn't need 500 × 151,936 logits (300 MB in fp32), only the last row. `forward(..., logits_indices=...)` applies the final projection to just the rows that get sampled.
- **One GPU sync per step.** All sampled rows move to the CPU together.
- **Errors come early.** `add_request` validates everything before queuing, so a server can return a clean 400 error instead of breaking a stream midway.

### Questions to be ready for
- Why is batching decode steps almost free, in terms of memory bandwidth?
- Static vs continuous batching: where does static waste work?
- What problem does chunked prefill solve?
- Which request gets preempted, and why that one?
- Why recompute instead of swap on Apple Silicon?
- How do you keep seeded sampling the same whether a request runs alone or in a batch?
