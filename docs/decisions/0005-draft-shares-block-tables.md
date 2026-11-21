# 0005: The draft model's KV cache shares the target's block tables

**Status:** accepted (M7)

**Context.** Speculative decoding runs two models over the same sequences, and each needs its own K/V (different layer counts, head sizes and weights). Each sequence's draft cache has to be allocated, grown, copied on write, rolled back after rejections, freed on preemption and, with prefix caching, reused on a hit, all in step with the target's.

**Decision.** The draft gets its own `PagedKVCache` with the same number of blocks and block size as the target, and it's addressed through the *same* block manager, block tables and slots. Block 37 means "these 16 positions" in both pools. Every block-manager operation therefore applies to both caches at once. The only extra work is applying copy-on-write copies to both pools.

**Why.**
- No second allocator to keep consistent. Preemption, rollback, abort and prefix caching needed no draft-specific code.
- A prefix-cache hit is valid for both models at once, provided a block is only registered after both caches hold its K/V. Registration uses `min(num_computed, draft_computed)`.
- Memory is sized together: `kiln serve --draft-model` divides `--kv-memory-gb` by the combined bytes per block.

**Costs.** A block's slot in the draft pool is reserved even before the draft has computed it, so a little memory is held early. Both pools must have the same number of blocks, so the smaller model's pool can't be sized independently.

**The one asymmetry: draft lag.** When all k drafts are accepted, the last draft token d_k was never fed to the draft, so its K/V is missing from the draft pool. `Sequence.draft_computed` records how far the draft's cache goes. Every draft forward feeds `tokens[draft_computed:]`, so the draft catches up in its next pass, which may then be ragged (one or two tokens per sequence).
