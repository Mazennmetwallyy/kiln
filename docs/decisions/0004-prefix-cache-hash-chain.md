# 0004: Prefix caching with a block hash chain and an LRU of evictable blocks

**Status:** accepted (M6)

**Context.** Many requests start with the same tokens: an assistant's system prompt, few-shot examples, a chat's earlier turns. Recomputing that prefix's K/V for every request wastes prefill compute and delays the first token. vLLM caches at block granularity with hashes. SGLang keeps a radix tree of token sequences.

**Decision.** Each *full*, computed block gets a key `hash((parent block's key, its 16 token ids))`, so a key identifies the entire prefix up to and including that block. A new request walks its prompt block by block, looking up keys, and reuses the longest chain of cached blocks (refcount + 1). Each hit is verified against the stored parent key and tokens, so a hash collision can only cause a miss, never wrong K/V. When a cached block's last user finishes, it isn't wiped. It moves to an LRU of *evictable* blocks, which still count as free memory and are reclaimed oldest-first only when no truly free block is left.

**Why not a radix tree?** Kiln's memory is already managed in blocks, so a block-level hash map reuses the existing allocator, refcounts and copy-on-write with about 100 lines of change. A radix tree can share at token granularity rather than block granularity, but that gain (at most 15 tokens per request) doesn't justify a second data structure.

**Details that matter.**
- At least one prompt token is always left to compute, because the first output token is sampled from the last prompt position's logits.
- Only full blocks are cached, and a sequence writes only past its cached prefix, so a shared cached block is never written in place and needs no copy-on-write.
- `rollback` into a cached block that this sequence alone owns removes it from the cache, because the block is about to be overwritten. A test that broke this deliberately is what found the missing case.
- Requests admitted in the same step don't hit each other: blocks are registered after the step that computes them. That matches vLLM.

**Consequences.** Outputs are unchanged (tested exactly against caching off on both real models). Freed memory stays useful until it's needed. `kiln serve --no-prefix-caching` turns it off. `/metrics` exports query and hit token counts.
