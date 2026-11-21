"""`kiln generate`: stream a completion from a local or Hugging Face Hub model."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

from kiln.generate import generate_stream
from kiln.model.loader import load_model
from kiln.sampling import SamplingParams
from kiln.tokenizer import Tokenizer

_DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def resolve_model_dir(model: str) -> Path:
    if Path(model).is_dir():
        return Path(model)
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(model, allow_patterns=["*.json", "*.safetensors", "*.jinja"]))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="kiln")
    sub = parser.add_subparsers(dest="command", required=True)
    g = sub.add_parser("generate", help="generate text from a prompt")
    g.add_argument("--model", required=True, help="Hugging Face repo id or local directory")
    g.add_argument("--prompt", required=True)
    g.add_argument("--chat", action="store_true", help="wrap the prompt in the model's chat template")
    g.add_argument("--max-tokens", type=int, default=256)
    g.add_argument("--temperature", type=float, default=0.7)
    g.add_argument("--top-p", type=float, default=1.0)
    g.add_argument("--top-k", type=int, default=0)
    g.add_argument("--seed", type=int, default=None)
    g.add_argument("--device", default="cpu", choices=["cpu", "mps"])
    g.add_argument("--dtype", default="fp32", choices=list(_DTYPES))
    args = parser.parse_args(argv)

    path = resolve_model_dir(args.model)
    model, config = load_model(path, dtype=_DTYPES[args.dtype], device=args.device)
    tok = Tokenizer.from_hf_dir(path)
    if args.chat:
        text = tok.apply_chat_template([{"role": "user", "content": args.prompt}])
        prompt_ids = tok.encode(text, add_special_tokens=False)
    else:
        prompt_ids = tok.encode(args.prompt)
    params = SamplingParams(temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
                            seed=args.seed, max_tokens=args.max_tokens)

    try:
        stream = generate_stream(model, config, prompt_ids, params, tok)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    n, finish = 0, None
    start = time.perf_counter()
    for out in stream:
        sys.stdout.write(out.new_text)
        sys.stdout.flush()
        n += len(out.new_token_ids)
        finish = out.finish_reason or finish
    elapsed = time.perf_counter() - start
    print(f"\n[{n} tokens in {elapsed:.2f}s ({n / elapsed:.1f} tokens/s end to end, incl. prefill), "
          f"finish={finish}]", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
