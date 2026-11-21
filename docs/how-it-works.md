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

---

## M4: The OpenAI-compatible server

### Shape

```
HTTP clients ──► FastAPI (asyncio event loop) ──commands──► engine thread (owns the LLMEngine)
     ▲                     ▲                                       │ step(): schedule → forward → sample
     └── SSE chunks ◄── per-request asyncio.Queue ◄─call_soon_threadsafe─┘
```

- **One thread owns the engine.** The engine isn't thread-safe and doesn't need to be. The web side sends it commands (`add`, `abort`, `stats`) through a thread-safe queue. After each step, the engine thread posts every request's new tokens to that request's `asyncio.Queue` with `loop.call_soon_threadsafe`. While the model runs a step, the event loop keeps accepting connections and streaming tokens.
- **Why not run the model on the event loop?** A forward pass blocks for tens of milliseconds. On the event loop it would freeze every other connection for that time.

### Errors before bytes

Once a streaming response has sent its `200 OK` header, you can't turn it into a `400`. So `submit()` runs the engine's validation (prompt too long, bad sampling values, `n > 1`, ...) on the engine thread and **awaits the result** before the response starts. Invalid requests get a proper OpenAI-style error: `{"error": {"message": ..., "type": "invalid_request_error"}}` with HTTP 400.

### Cancellation

If a client disconnects mid-stream, Starlette cancels the response task. That closes Kiln's async generator, and its `finally:` block sends an `abort` to the engine thread, which frees the request's KV blocks. A test opens a stream, reads a few chunks, hangs up, and checks that all blocks are free again within 2 seconds. Without this, every closed browser tab would leak GPU memory until its `max_tokens` ran out.

### Streaming format

Server-Sent Events, the same format OpenAI uses:
- each chunk is a line `data: {json}` followed by a blank line;
- a chat stream opens with a `{"role": "assistant"}` delta, then sends content deltas;
- the final chunk carries `finish_reason` (`"stop"` or `"length"`);
- an optional usage chunk follows (`stream_options.include_usage`);
- the stream ends with `data: [DONE]`.

Every test drives the server with the official `openai` Python client, so "OpenAI-compatible" is tested, not claimed.

### Metrics (`/metrics`, Prometheus)

| metric | why it matters |
|---|---|
| `kiln_time_to_first_token_seconds` | what a user feels before anything appears; dominated by queueing + prefill |
| `kiln_inter_token_latency_seconds` | smoothness of the stream; dominated by decode-step time |
| `kiln_e2e_request_latency_seconds` | total request time |
| `kiln_running_requests`, `kiln_waiting_requests` | load and queueing |
| `kiln_kv_blocks_used` / `_total` | memory pressure, which predicts preemptions |
| `kiln_preemptions`, token counters | throughput and thrashing |

### Questions to be ready for
- Why does the model run on its own thread rather than on the asyncio event loop?
- How does an output produced on the engine thread reach the right HTTP response?
- Why must validation finish before a streaming response starts?
- What happens, step by step, when a client disconnects mid-stream?
- TTFT vs inter-token latency: what does each measure, and what makes each worse?

## M5: Benchmarks

### What is measured, and how
- **Workload.** 64 conversations sampled (seed 0) from ShareGPT V3: real user questions with the length of the real answer as `max_tokens`. Prompts are chat-templated and tokenized once, so every system sees identical token ids. Generation ignores EOS, so every system produces exactly the same number of tokens.
- **Same interface for everyone.** Kiln and both Hugging Face baselines (`bench/hf_server.py`) sit behind the same streaming `/v1/completions` endpoint. The client measures from outside, so HTTP and tokenization overhead are counted for all three.
- **Open-loop load.** Requests arrive as a Poisson process at 0.5, 1 and 2 per second, then all at once. Open-loop means arrivals don't wait for earlier answers. That's what real traffic does, and it's what exposes queueing. A closed loop (send the next request when one finishes) hides queueing delay and flatters slow systems.
- **Metrics.** Output throughput, time to first token (TTFT), inter-token latency (ITL), end-to-end latency. Medians in the README, p99 in `bench/results/serving.json`.

### Reading the results
- **TTFT is where continuous batching shows most.** HF static batching can't return anything until the whole padded batch is done, so its TTFT equals its end-to-end time. HF sequential makes each request wait for every earlier one. Kiln admits a new request at the next step (~30–130 ms later) and prefills it in chunks alongside running decodes.
- **Throughput saturates differently.** Sequential is flat at ~55 tok/s, one stream's decode speed. Static batching rises to ~130 tok/s but wastes compute on padding and on finished sequences waiting for the longest one. Kiln reaches 306 tok/s because finished requests leave and new ones join every step, so the batch stays full.
- **Memory.** Paging allocates 16 tokens at a time, so a request only holds memory for tokens it actually has. Reserving each request's worst case (prompt + max_tokens) fits 2.4× fewer requests in the same budget. Reserving the context window fits ~30× fewer.
- **Where Kiln loses.** On a fixed batch Hugging Face is ~1.4–2× faster per step. Its attention reads one contiguous K/V tensor. Kiln's torch-only paged attention gathers each sequence's blocks into a temporary tensor per layer per step, and adds Python bookkeeping per step. Continuous batching more than makes up for it under real load, but the gap is real and is the motivation for M8's kernel.

### A bug the benchmark found in the baseline
The first run of the static-batching baseline hung for hours with the GPU idle. Its batcher waits up to 50 ms for more requests using `asyncio.wait_for` and caught `TimeoutError`. On Python 3.10, `asyncio.wait_for` raises `asyncio.TimeoutError`, a different class (they were unified in 3.11). The first partial batch killed the batcher task silently, and every later request awaited a future nobody would resolve. A regression test (`tests/bench/test_hf_server_batcher.py`) sends fewer requests than a full batch and requires them to finish.

### Questions to be ready for
- Why open-loop load rather than closed-loop? What would a closed loop hide?
- Why does static batching's TTFT equal its end-to-end latency?
- Kiln is slower than Hugging Face on a fixed batch. Why does it still win the serving benchmark?
- Why must every system produce exactly the same number of output tokens for the comparison to be fair?

## M6: Prefix caching

### The waste
An assistant product sends the same system prompt in front of every user message. Without caching, every request recomputes K/V for those identical tokens. In the M6 benchmark that's 428 of a mean 490 prompt tokens: 87% of all prefill is repeated work, and it sits right in front of the first token.

### Content-addressed blocks
K/V for a token depends on that token *and every token before it*. So a block can only be reused by a request whose entire prefix up to that block is identical, not just the block's own 16 tokens. Kiln gives every full, computed block a key:

```
key(block i) = sha256( key(block i-1) ‖ tokens of block i )      # key(block -1) = empty
```

Chaining the parent's key in makes the key stand for the whole prefix. A new request computes the same keys for its prompt and follows the chain until the first miss. Matching blocks are shared (refcount + 1). The request's `num_computed` starts at the cached length, so the scheduler only sends the rest through the model.

A hit is also checked against the stored parent key and the stored tokens. The hash must be cryptographic. With Python's built-in 64-bit `hash()`, someone could craft a prompt whose first block collides with another prompt's first-block key, then register a second block whose "parent key" matches. Later requests would reuse that block's K/V, which was computed after a different first block. An automated security review flagged this, and SHA-256 makes it infeasible.

### Keeping freed blocks around
When the last request using a cached block finishes, the block isn't wiped. It joins an LRU of *evictable* blocks: still holding valid K/V, still findable, but counted as free. Allocation takes truly free blocks first, then evicts the least recently used cached block, dropping its key. So the cache costs no memory that anything else needed, and a popular system prompt stays resident as long as there's room.

### Edge cases the tests pin down
- **One token is always left to compute.** The first output token is sampled from the last prompt position's logits, so a fully cached prompt still recomputes its last block.
- **Only full blocks are cached**, and a request writes only after its cached prefix, so shared cached blocks are read-only and need no copy-on-write.
- **Rollback** (used by M7's speculative decoding) can make a cached block partial again. If this sequence is its only owner, the next write will overwrite it in place, so it's removed from the cache first. If it's shared, the write triggers copy-on-write and the cached copy stays valid.
- **Preemption** frees a request's blocks, but its full blocks stay cached, so re-admission usually finds most of its own K/V still there.
- **A Hypothesis state machine** draws prompts from a tiny alphabet so prefixes repeat constantly, interleaves allocate, register, append, fork, rollback and free, and checks after every step that refcounts match block tables, every unused block is free *xor* evictable, and every cached block's stored tokens match what the sequence holding it put there.

### Results
Prefill computed fell 83% (31,368 → 5,304 tokens) and mean time to first token fell 4.8× (10.3 s → 2.2 s), with identical outputs. Requests admitted in the very first step don't benefit, because nothing is cached until the step that computes the system prompt finishes.

### Questions to be ready for
- Why must the key include the parent block's key rather than just the block's own tokens?
- What happens on a hash collision?
- Why is at least one prompt token always recomputed?
- Why do cached blocks never need copy-on-write, but rolled-back ones need special handling?
- Evictable blocks count as free. Why is that safe, and what happens when one is needed?
- Block-level hashing vs a radix tree (SGLang): what does each give up?
- Why must the block key be a cryptographic hash? What could an attacker do with a 64-bit one?
- How can a prefix cache leak information between users, and what are the mitigations?

## M7: Speculative decoding

### Why it can help
At batch 1, a decode step is limited by reading the weights and by per-layer overhead, not by arithmetic. Scoring 5 tokens costs about the same as scoring 1 (62 vs 55 ms for Qwen2.5-1.5B on the M2). So if a cheap model can guess the next few tokens, the big model can check them all in one pass and keep the ones it agrees with.

### The algorithm (`kiln/spec/rejection.py`)
For each draft token d_i, sampled from the draft's distribution q_i, with the target's distribution p_i computed in the verification pass:
- accept d_i with probability min(1, p_i(d_i) / q_i(d_i));
- at the first rejection, sample a replacement from norm(max(0, p_i − q_i)) and stop;
- if all k are accepted, sample one bonus token from p_k, which the target computed anyway.

Every emitted token is then distributed exactly as the target alone would have sampled it. The accepted part contributes min(p, q) and the residual contributes p − min(p, q), which together give p. Greedy decoding is the special case where p and q are one-hot: accept while the draft equals the target's argmax. p and q must be the *warped* distributions (temperature, top-k, top-p), the ones actually sampled from, so `sampling.probs()` is shared by the sampler and the verifier.

### Inside the continuous batch (`LLMEngine._speculative_step`)
- Kiln speculates only on steps where every scheduled request is decoding. A step with any prefill runs normally, and the draft processes the same tokens so its cache stays in sync.
- For each request, k is capped so that k accepted tokens plus the bonus token can't exceed `max_tokens`, and it drops to 0 if the pool can't reserve k more slots.
- The draft runs k passes. The first may be ragged, because the draft can trail by a token (below).
- The target runs **one** pass over [last token, d1 … dk] for every request.
- After verification, `bm.rollback(k − j)` returns the rejected slots in both caches. Emitted tokens go through the normal stop/EOS/`max_tokens` checks one at a time, so a stop in the middle of an accepted run discards the rest.

### Draft lag
When all k drafts are accepted, d_k was never fed to the draft, so the draft's cache is one token behind. `Sequence.draft_computed` tracks this, and every draft forward feeds `tokens[draft_computed:]`. A block joins the prefix cache only once *both* caches hold it, at `min(num_computed, draft_computed)`.

### Testing it
- Greedy speculative output must equal target-only output. That's tested with an identical draft (everything accepted), a slightly perturbed draft (partial acceptance) and an unrelated draft (almost nothing accepted), for k = 1, 3 and 5, and together with preemption, chunked prefill, prefix caching and mid-acceptance stops.
- Chi-square tests over 20,000 trials check that the first *and* second emitted tokens follow p. They include drafts that put zero probability on tokens the target wants, and targets that put zero probability on tokens the draft proposes.
- A deliberate break (removing the rollback) has to fail the tests. At first it didn't: the tiny random test model repeated one token whatever its context, so wrong K/V never showed in the output. Scaling its weights gave varied, context-sensitive output, and the break was then caught.

### What it gave on this machine
Correctness held: 12/12 identical on SmolLM2 135M → 360M in fp32. Speed only modestly improved: 1.28× at k = 2, batch 1, and slower at batch 4. The draft costs about 0.45× of a target step, because both are overhead-bound at batch 1, so the ceiling is roughly tokens-per-forward / (0.45k + 1). On top of that, two models that don't both fit in 8 GB of RAM swap. The README gives the numbers and the profile.

### Questions to be ready for
- Why does accepting with probability min(1, p/q) and resampling from max(0, p − q) give exactly p?
- Why must q be the distribution after temperature/top-k/top-p?
- Why is the bonus token free?
- When does speculative decoding stop paying off? (Draft cost ratio, acceptance rate, batch size.)
- Why can't the draft's cache simply be a second block manager?
- What happens to the KV cache when 2 of 4 drafts are rejected?

## M8: A Metal paged-attention kernel

### Why attention, and why decode
In a decode step, every request contributes one query and has to read its whole K/V history. The PyTorch path gathers that history out of scattered blocks (`k_pool[slots]`) into a temporary tensor padded to the longest sequence in the batch, in every layer. Under serving load that copy dominated the step: 51 of 93 ms at about 700 tokens of context.

### The kernel (`kiln/attention/metal.py`)
- **Grid.** One threadgroup of 128 threads (4 SIMD groups of 32) per (sequence, query head, context partition). Thread *i* of a SIMD group owns dimensions [i·D/32, (i+1)·D/32) of the head.
- **Dot products.** For each token the kernel looks up its slot through the block table, then each thread multiplies its slice of q and k. `simd_sum` adds the 32 partial sums in hardware. The 32 threads read one K vector together, so the loads are contiguous.
- **Online softmax.** Each SIMD group keeps a running max m, a running denominator l and a running weighted sum of V. When a new score raises the max, everything accumulated so far is multiplied by exp(m_old − m_new). The scores never exist as a tensor.
- **Merging.** The 4 SIMD groups merge through threadgroup memory, each rescaled to the common max.
- **Split-K.** The first version had one threadgroup per (sequence, head), which is 14 threadgroups for a single request: far too few to fill the GPU. It was *slower* than PyTorch at long contexts. Splitting the context into 256-token partitions, each producing an unnormalised (m, l, acc), plus a small second kernel to merge partitions, made it faster everywhere. This is the FlashDecoding idea.
- GQA is just `kv_head = h / (H / Hkv)`, so no K/V is copied per query head.

### Results
- **Attention alone:** 1.7–2.8× faster.
- **Whole decode step:** up to 1.5× faster (batch 32 at 1,024 tokens: 129.5 → 86.5 ms).
- **Serving:** 311 → 569 tok/s with all 64 requests at once, because the kernel never pays for padding.

At batch 1 the step barely changes: attention is under 1 ms of a roughly 27 ms step.

### Things found on the way
- **Hundreds of dtype round trips per token.** RoPE cast q and k to fp32 and back in every layer, and each cast is a separate GPU kernel: 302 casts per decode step. Computing cos/sin once per forward in the model's dtype, as Hugging Face does, halved that.
- **Why Hugging Face's `generate` is still faster at batch 1.** Both models launch a similar number of operations, and a Kiln forward pass is in fact a little quicker than a direct Hugging Face forward call. But `generate` defers its stop check, so it never waits for the GPU between steps, and CPU dispatch overlaps GPU execution. A continuous-batching scheduler needs each step's tokens before it can plan the next step. Overlapping the two (async scheduling) is the next step.

### Questions to be ready for
- What does the online softmax store, and why is it numerically safe?
- Why is the q·k product a `simd_sum`, and what makes the K loads coalesced?
- Why was the first kernel slower than PyTorch for one long request, and how does split-K fix it?
- Why is the gain bigger under serving load than in a fixed batch?
- What would you do next for speed? (`simdgroup_matrix`, `half4` loads, several query heads per threadgroup for GQA, async scheduling.)

## M9: The live KV-cache view

`kiln serve --enable-viz` serves `/viz`, a single page in plain HTML, CSS and JS with no build step, plus `/kv` (one snapshot) and `/kv/stream` (Server-Sent Events).
- **Snapshots.** After a step, the engine thread takes a snapshot of every block's state (free, used by one request, shared, cached-and-evictable) and owner, at most 10 times a second and always when a request finishes. HTTP handlers only read the latest snapshot under a lock, so watching never stalls the engine.
- **Colour encodes state, not owner.** 64 requests can't get distinct colours, so the owner is shown on hover and in the request list, and hovering a request outlines its blocks.
- **Opt-in.** It's off by default. It reveals which requests share which prefixes, which is exactly the cross-user side channel ADR 0004 warns about. An automated security review flagged it, and the fix was to make it opt-in.
