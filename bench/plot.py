"""Render the benchmark charts for the README from bench/results/*.json.

    python -m bench.plot
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = Path(__file__).parent / "results"
OUT = Path(__file__).parents[1] / "docs" / "img"

# Categorical slots in fixed order: identity follows the system, never its rank.
SERIES = {"kiln": ("Kiln", "#2a78d6", "o"), "hf-static": ("HF static batching", "#eb6834", "s"),
          "hf-sequential": ("HF sequential", "#1baf7a", "^"), "kiln-metal": ("Kiln + Metal kernel", "#eda100", "D")}
INK, INK_2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 10, "axes.edgecolor": AXIS, "axes.labelcolor": INK_2,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlecolor": INK, "axes.titlesize": 11,
    "axes.titleweight": "bold", "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False,
})


def _label_end(ax, x, y, text: str, color: str) -> None:
    ax.annotate(text, (x, y), xytext=(6, 0), textcoords="offset points", va="center", fontsize=9, color=INK_2)
    ax.plot([], [], color=color)  # keeps identity tied to the line; text stays in ink, not series colour


def serving() -> None:
    data = json.loads((RESULTS / "serving.json").read_text())
    runs, meta = data["runs"], data["meta"]
    panels = [("output_throughput", "Output throughput (tokens/s)", False),
              ("ttft_p50", "Time to first token, p50 (s)", True),
              ("e2e_p50", "End-to-end latency, p50 (s)", True)]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, (key, title, log) in zip(axes, panels, strict=True):
        for system, (name, color, marker) in SERIES.items():
            if system not in runs:
                continue
            xs = list(range(len(runs[system])))
            ys = [r[key] for r in runs[system]]
            ax.plot(xs, ys, color=color, lw=2, marker=marker, ms=8, mec=SURFACE, mew=1.5, label=name)
            _label_end(ax, xs[-1], ys[-1], name, color)
        rates = [r["rate"] for r in next(iter(runs.values()))]
        ax.set_xticks(range(len(rates)), ["all at once" if math.isinf(r) else f"{r:g}/s" for r in rates])
        ax.set_xlabel("Request arrival rate")
        ax.set_title(title, loc="left")
        ax.set_xlim(-0.3, len(rates) - 1 + 1.6)
        if log:
            ax.set_yscale("log")
    axes[0].legend(loc="upper left", fontsize=9)
    env = meta["env"]
    fig.text(0.01, -0.02, f"{meta['num_requests']} ShareGPT requests (mean {meta['mean_prompt_tokens']:.0f} prompt / "
             f"{meta['mean_output_tokens']:.0f} output tokens), {meta['model']} fp16, {env['cpu']} "
             f"{env['memory_gb']} GB, commit {env['commit']}", fontsize=8, color=MUTED)
    fig.tight_layout()
    fig.savefig(OUT / "serving.png", dpi=160, bbox_inches="tight")


def decode_scaling() -> None:
    data = json.loads((RESULTS / "decode_scaling.json").read_text())
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for key, system in (("kiln", "kiln"), ("kiln_metal", "kiln-metal"), ("hf_static", "hf-static")):
        if key not in data:
            continue
        name, color, marker = SERIES[system]
        xs = [int(b) for b in data[key]]
        ys = list(data[key].values())
        ax.plot(xs, ys, color=color, lw=2, marker=marker, ms=8, mec=SURFACE, mew=1.5, label=name)
        _label_end(ax, xs[-1], ys[-1], f"{name}: {ys[-1]:.0f}", color)
    ax.set_xscale("log", base=2)
    ax.set_xticks(xs, [str(x) for x in xs])
    ax.set_xlim(0.8, 260)
    ax.set_xlabel("Concurrent requests (batch size)")
    ax.set_title("Generation throughput (tokens/s), 128-token prompts, 64 new tokens", loc="left")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "decode_scaling.png", dpi=160, bbox_inches="tight")


def memory() -> None:
    d = json.loads((RESULTS / "memory.json").read_text())
    rows = [(f"Contiguous: reserve {d['meta']['context_reserved']}-token context", d["contiguous_reserve_context"]),
            ("Contiguous: reserve prompt + max_tokens", d["contiguous_reserve_request_max"]),
            ("Kiln paged KV cache (measured)", d["kiln_paged"]["peak_concurrent"])]
    fig, ax = plt.subplots(figsize=(7, 2.6))
    ys = range(len(rows))
    ax.barh(list(ys), [v for _, v in rows], height=0.55, color=SERIES["kiln"][1])
    for y, (_, v) in zip(ys, rows, strict=True):
        ax.annotate(str(v), (v, y), xytext=(5, 0), textcoords="offset points", va="center", color=INK, fontsize=10)
    ax.set_yticks(list(ys), [n for n, _ in rows], color=INK_2)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Requests in flight at once")
    ax.set_title(f"Requests that fit in a {d['meta']['budget_mb']} MB KV cache", loc="left")
    fig.tight_layout()
    fig.savefig(OUT / "memory.png", dpi=160, bbox_inches="tight")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for f, name in ((serving, "serving.json"), (decode_scaling, "decode_scaling.json"), (memory, "memory.json")):
        if (RESULTS / name).exists():
            f()
            print("wrote", name.replace(".json", ".png"))
