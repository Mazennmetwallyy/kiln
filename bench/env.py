"""Hardware/software description recorded with every result file."""

from __future__ import annotations

import platform
import subprocess

import torch


def describe() -> dict[str, str]:
    def sh(*cmd: str) -> str:
        try:
            return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
        except Exception:
            return "unknown"

    mem = sh("sysctl", "-n", "hw.memsize")
    return {
        "cpu": sh("sysctl", "-n", "machdep.cpu.brand_string"),
        "memory_gb": f"{int(mem) / 2**30:.0f}" if mem.isdigit() else mem,
        "os": f"{platform.system()} {platform.mac_ver()[0] or platform.release()}",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "commit": sh("git", "rev-parse", "--short", "HEAD"),
    }
