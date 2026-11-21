"""Experiment A: online serving under Poisson load, Kiln vs Hugging Face baselines.

    python -m bench.run_serving --data bench/data/sharegpt_64_seed0.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import httpx

from bench import loadgen
from bench.data import load
from bench.env import describe

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
RESULTS = Path(__file__).parent / "results" / "serving.json"


def command(system: str, port: int) -> list[str]:
    py = sys.executable
    if system == "kiln":
        return [py, "-m", "kiln.cli", "serve", "--model", MODEL, "--port", str(port), "--device", "mps",
                "--dtype", "fp16", "--kv-memory-gb", "1", "--max-num-seqs", "64", "--max-batched-tokens", "512"]
    if system == "hf-sequential":
        return [py, "-m", "bench.hf_server", "--mode", "sequential", "--model", MODEL, "--port", str(port)]
    if system == "hf-static":
        return [py, "-m", "bench.hf_server", "--mode", "static", "--max-batch", "16", "--model", MODEL,
                "--port", str(port)]
    raise ValueError(system)


def wait_healthy(url: str, proc: subprocess.Popen, timeout: float = 300) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited with {proc.returncode}")
        try:
            if httpx.get(f"{url}/health", timeout=2).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(1)
    raise TimeoutError(url)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--systems", nargs="+", default=["kiln", "hf-static", "hf-sequential"])
    p.add_argument("--rates", nargs="+", type=float, default=[0.5, 1.0, 2.0, math.inf])
    p.add_argument("--port", type=int, default=8077)
    a = p.parse_args()
    requests = load(Path(a.data))

    results = json.loads(RESULTS.read_text()) if RESULTS.exists() else {"runs": {}}
    results["meta"] = {"env": describe(), "model": MODEL, "dtype": "fp16", "device": "mps",
                       "dataset": Path(a.data).name, "num_requests": len(requests),
                       "mean_prompt_tokens": sum(len(r["prompt_ids"]) for r in requests) / len(requests),
                       "mean_output_tokens": sum(r["output_len"] for r in requests) / len(requests)}
    url = f"http://127.0.0.1:{a.port}"
    for system in a.systems:
        log = open(RESULTS.parent / f"{system}.log", "w")  # noqa: SIM115
        proc = subprocess.Popen(command(system, a.port), stdout=log, stderr=subprocess.STDOUT)
        try:
            wait_healthy(url, proc)
            asyncio.run(loadgen.run(url, requests[:2], math.inf))  # warm-up (MPS kernels compile lazily)
            runs = []
            for rate in a.rates:
                summary = asyncio.run(loadgen.run(url, requests, rate, seed=0))
                summary["rate"] = rate
                runs.append(summary)
                print(f"{system:14s} rate={rate:>4}: {summary['output_throughput']:7.1f} tok/s, "
                      f"TTFT p50 {summary['ttft_p50']:.2f}s, ITL p50 {summary['itl_p50'] * 1000:.0f}ms, "
                      f"e2e p50 {summary['e2e_p50']:.1f}s, failed {summary['failed']}", flush=True)
            results["runs"][system] = runs
            RESULTS.write_text(json.dumps(results, indent=2, default=str))
        finally:
            proc.terminate()
            proc.wait(timeout=30)
            log.close()


if __name__ == "__main__":
    main()
