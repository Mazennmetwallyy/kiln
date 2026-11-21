# Kiln

An LLM inference engine built from scratch in Python and PyTorch: my own transformer forward pass, a paged KV cache, continuous batching, prefix caching, speculative decoding and a hand-written Metal attention kernel, served behind an OpenAI-compatible API.

> Work in progress. Milestone 1 (the model) is under way.

## How this was built

I designed and built Kiln with an AI coding assistant (Claude). Every commit says so. I write up how each component works in [`docs/how-it-works.md`](docs/how-it-works.md), and I can explain every part of it.
