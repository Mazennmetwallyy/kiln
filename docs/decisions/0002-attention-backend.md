# 0002: Attention goes through an AttentionContext

**Status:** accepted (M2). Refines ADR 0001.

**Context.** In M1, `Attention.forward` wrote to the cache, read it back and built the causal mask itself, using `positions[-1] + 1` as the sequence length. That only works for one sequence, and `int(positions[-1])` forces a GPU→CPU sync in every layer of every step.

**Decision.** After RoPE, attention calls `ctx.attend(layer, q, k, v)`. The context object (the third argument to `Transformer.forward`) owns K/V storage and the attention computation. It carries a per-step `AttentionMetadata` with per-sequence lengths as **Python ints**, plus precomputed gather indices built once per step and reused by all layers.

**Consequences.**
- Contiguous (`ContiguousKVCache`) and paged (`PagedContext`) storage are interchangeable. M8's Metal kernel is a third implementation.
- Sequence boundaries in a ragged batch live only in the metadata, which is the promise ADR 0001 made.
- No device syncs inside the layers. Single-stream decode on MPS went from 23.5 to 32.0 tokens/s with no other change.
