# 0006: A hand-written Metal kernel for paged decode attention, with split-K

**Status:** accepted (M8)

**Context.** In a decode step every request contributes one query token, and attention has to read that request's whole K/V history. The PyTorch backend gathers the history out of the paged pool (`k_pool[slots]`) into a temporary tensor, then calls SDPA with a padding mask. That's a full copy of every sequence's K/V, in every layer, on every step, padded to the longest sequence in the batch. On the M5 serving benchmark the gather was the single biggest cost at long contexts (profiled: 51 of 93 ms per step at about 700 tokens of context).

**Decision.** A Metal compute kernel (`kiln/attention/metal.py`, compiled at runtime with `torch.mps.compile_shader`) reads K/V straight from the pool through each sequence's block table. Prefill and mixed steps keep the PyTorch path, which is the correctness reference and is used by tests. The kernel is opt-in with `--attention-backend metal`.

**Kernel shape.**
- One threadgroup of 128 threads (4 SIMD groups of 32) per (sequence, query head, context partition).
- Each thread owns D/32 dimensions of the head. A token's q·k is a `simd_sum` across the SIMD group, so the 32 threads of a SIMD group read one contiguous K vector together (coalesced).
- The 4 SIMD groups take interleaved tokens and keep an **online softmax** (running max, running denominator, running weighted sum of V), so scores are never written to memory. Their results are rescaled to a common max and merged in threadgroup memory.
- Grouped-query attention is just `kv_head = h / (H / Hkv)`. Nothing is copied.
- **Split-K (FlashDecoding).** The first version used one threadgroup per (sequence, head). With a single request that's only 14 threadgroups, too few to fill the GPU, so it was *slower* than PyTorch at long contexts (0.87x at 2,048 tokens). Splitting the context into 256-token partitions, each written as an unnormalised partial (m, l, acc) and merged by a second small kernel, made it faster than PyTorch in every configuration measured.

**Why not more.** The kernel reads K/V one token at a time per SIMD group and doesn't use `simdgroup_matrix` (Apple's matrix units) or vectorised `half4` loads. Those are the next steps if attention becomes the bottleneck again. At batch 1 it isn't: the whole decode step is about 27 ms, and attention is under 1 ms of it.

**Verification.** Randomised tests against the PyTorch reference: fp16 and fp32, four head layouts including GQA, sequence lengths 1–250 across shuffled blocks, and partition sizes that force 0, some, and many splits (including partitions past the end of shorter sequences). Mutation checks: a wrong KV-head mapping, a dropped last token, and an unweighted merge each fail the tests. Engine greedy output is identical with the torch and metal backends, including under preemption and speculative decoding.
