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

This demo runs Qwen2.5-0.5B-Instruct on 2 free CPU cores, so it is much slower than the Apple-GPU numbers in the repo. Chat with it, or press **Send a burst** to fire several requests at once and watch them share and fill the KV cache on the right.

The OpenAI-compatible API is live too: base URL `https://mazennmetwallyy--kiln-demo-web.us-east.modal.run/v1`.

Deploy: `modal deploy demo/modal_app.py` (free tier, scales to zero). The Dockerfile here runs the same app on any Docker host.

Everyone using the Space sees the same KV cache view. It shows block usage and token counts, never text, but don't type anything private.
