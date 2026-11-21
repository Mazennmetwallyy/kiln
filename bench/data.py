"""Build the benchmark workload from ShareGPT: (prompt token ids, output length) pairs.

    python -m bench.data --n 64 --model Qwen/Qwen2.5-0.5B-Instruct
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

from kiln.tokenizer import Tokenizer

DATA_DIR = Path(__file__).parent / "data"
MAX_PROMPT, MIN_OUTPUT, MAX_OUTPUT = 512, 4, 256


def build(n: int, model: str, seed: int = 0) -> Path:
    raw = hf_hub_download("anon8231489123/ShareGPT_Vicuna_unfiltered", "ShareGPT_V3_unfiltered_cleaned_split.json",
                          repo_type="dataset")
    tok = Tokenizer.from_hf_dir(snapshot_download(model, allow_patterns=["*.json"]))
    convs = [c["conversations"] for c in json.loads(Path(raw).read_text())]
    pairs = [(c[0]["value"], c[1]["value"]) for c in convs
             if len(c) >= 2 and c[0]["from"] == "human" and c[1]["from"] == "gpt"]
    random.Random(seed).shuffle(pairs)

    out = []
    for prompt, reply in pairs:
        ids = tok.encode(tok.apply_chat_template([{"role": "user", "content": prompt}]), add_special_tokens=False)
        out_len = len(tok.encode(reply, add_special_tokens=False))
        if len(ids) <= MAX_PROMPT and MIN_OUTPUT <= out_len <= MAX_OUTPUT:
            out.append({"prompt_ids": ids, "output_len": out_len})
        if len(out) == n:
            break
    DATA_DIR.mkdir(exist_ok=True)
    path = DATA_DIR / f"sharegpt_{n}_seed{seed}.json"
    path.write_text(json.dumps({"model": model, "seed": seed, "filters": {"max_prompt": MAX_PROMPT,
                                "output": [MIN_OUTPUT, MAX_OUTPUT]}, "requests": out}))
    return path


def load(path: Path) -> list[dict]:
    return json.loads(Path(path).read_text())["requests"]


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    print(build(a.n, a.model, a.seed))
