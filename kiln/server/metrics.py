"""Prometheus metrics. Each app gets its own registry (so tests can build several apps)."""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

from kiln.engine.engine import EngineStats, RequestOutput

_LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60)


class Metrics:
    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        r = self.registry
        self.requests = Counter("kiln_requests", "Requests accepted", registry=r)
        self.prompt_tokens = Counter("kiln_prompt_tokens", "Prompt tokens of accepted requests", registry=r)
        self.generation_tokens = Counter("kiln_generation_tokens", "Tokens generated", registry=r)
        self.preemptions = Gauge("kiln_preemptions", "Preemptions since start", registry=r)
        self.running = Gauge("kiln_running_requests", "Requests being processed", registry=r)
        self.waiting = Gauge("kiln_waiting_requests", "Requests queued", registry=r)
        self.blocks_used = Gauge("kiln_kv_blocks_used", "KV cache blocks in use", registry=r)
        self.blocks_total = Gauge("kiln_kv_blocks_total", "KV cache blocks in the pool", registry=r)
        self.blocks_cached = Gauge("kiln_kv_blocks_cached", "Blocks holding reusable prefix K/V", registry=r)
        self.prefix_query = Gauge("kiln_prefix_cache_query_tokens", "Prompt tokens looked up in the prefix cache",
                                  registry=r)
        self.prefix_hit = Gauge("kiln_prefix_cache_hit_tokens", "Prompt tokens served from the prefix cache",
                                registry=r)
        self.spec_proposed = Gauge("kiln_spec_proposed_tokens", "Draft tokens proposed", registry=r)
        self.spec_accepted = Gauge("kiln_spec_accepted_tokens", "Draft tokens accepted by the target", registry=r)
        self.ttft = Histogram("kiln_time_to_first_token_seconds", "Time to first token", registry=r,
                              buckets=_LATENCY_BUCKETS)
        self.itl = Histogram("kiln_inter_token_latency_seconds", "Time between streamed outputs", registry=r,
                             buckets=_LATENCY_BUCKETS)
        self.e2e = Histogram("kiln_e2e_request_latency_seconds", "Request latency", registry=r,
                             buckets=_LATENCY_BUCKETS)

    def on_step(self, outputs: list[RequestOutput], stats: EngineStats) -> None:
        """Called on the engine thread after every step."""
        self.generation_tokens.inc(sum(len(o.new_token_ids) for o in outputs))
        self.running.set(stats.num_running)
        self.waiting.set(stats.num_waiting)
        self.blocks_used.set(stats.num_blocks - stats.num_free_blocks)
        self.blocks_total.set(stats.num_blocks)
        self.preemptions.set(stats.num_preemptions_total)
        self.blocks_cached.set(stats.num_cached_blocks)
        self.prefix_query.set(stats.prefix_cache_query_tokens)
        self.prefix_hit.set(stats.prefix_cache_hit_tokens)
        self.spec_proposed.set(stats.spec_proposed_tokens)
        self.spec_accepted.set(stats.spec_accepted_tokens)
