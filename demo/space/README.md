---
title: Kiln
emoji: 🔥
colorFrom: blue
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
short_description: An LLM inference engine from scratch, with a live KV cache
---

# Kiln live demo

[Kiln](https://github.com/Mazennmetwallyy/kiln) is an LLM inference engine built from scratch: paged KV cache, continuous batching, prefix caching, speculative decoding and a Metal attention kernel.

This Space runs Qwen2.5-0.5B-Instruct on 2 free CPU cores, so it is much slower than the Apple-GPU numbers in the repo. Chat with it, or press **Send a burst** to fire several requests at once and watch them share and fill the KV cache on the right.

The OpenAI-compatible API is live too: `POST /v1/chat/completions` (base URL `https://<this space>.hf.space/v1`).

Everyone using the Space sees the same KV cache view. It shows block usage and token counts, never text, but don't type anything private.
