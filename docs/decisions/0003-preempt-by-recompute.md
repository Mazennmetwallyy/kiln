# 0003: Preempt by recomputation, not by swapping

**Status:** accepted (M3)

**Context.** When the KV pool is full and a running request needs another block, something has to give. vLLM supports two options: *swap* the victim's KV blocks out to CPU memory and back later, or *recompute* them by dropping them and redoing the victim's prefill later.

**Decision.** Kiln preempts by recomputation. The most recently admitted running request is the victim. Its blocks are freed and it goes back to the **front** of the queue with its prompt *and* the tokens it has generated so far, so resuming is a single prefill.

**Why.**
- On Apple Silicon, CPU and GPU share the same physical memory. "Swapping to CPU" would just move bytes between two parts of the same RAM, using the very memory that ran out.
- Prefill is compute-bound and efficient (one big batched forward), so recomputing a few hundred tokens is cheap next to decode.
- It's much simpler: no second pool and no async copy streams.

**Consequences.** Preempted requests pay for their prefill again, and the engine counts preemptions so benchmarks can show this. A preempted request resumes with token-for-token identical output; the tests check this with a deliberately tiny pool.
