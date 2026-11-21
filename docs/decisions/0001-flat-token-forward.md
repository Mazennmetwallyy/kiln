# 0001: The model's forward pass takes a flat 1-D token stream

**Status:** accepted (M1)

**Context.** Most reference code runs a forward pass on `[batch, seq_len]` tensors. A batch of requests at different lengths then has to be padded to the longest one, which wastes compute and memory. Inference servers batch requests that are at very different stages: one request is processing a 2,000-token prompt while 30 others each add one token.

**Decision.** `Transformer.forward(input_ids, positions, cache)` takes `input_ids` and `positions` as 1-D tensors of shape `[T]`, with no batch dimension. Each token carries its own position. In M1 all T tokens belong to one sequence. From M3 on, tokens from many sequences are concatenated ("ragged batching"), and metadata such as sequence lengths and block tables says which token belongs to which sequence.

**Consequences.**
- No padding, ever. The matrix multiplies (q/k/v/o projections, MLP, lm_head) treat all T tokens as one big batch, which is where most FLOPs are.
- Only attention needs to know where one sequence ends and the next begins. That keeps the batching logic in one place (the attention backend), not spread through the model.
- Positions come in explicitly, so prefill (many tokens) and decode (one token) are the same code path.
- Cost: attention is more complex than a padded batched SDPA call. M3 handles it.
