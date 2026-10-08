"""
Compare regenerated outputs with the committed reference versions.

`git status` flags any byte change, but PNG bytes depend on the installed image libraries
even when the pixels are identical, and the appendix diagnostics carry floating-point noise
across hardware. This script compares what matters: pixels for figures, numbers for
.json/.csv/.tex/.md files.

Usage:
    .venv/bin/python reproduce/check_outputs.py results/                 # main paper
    .venv/bin/python reproduce/check_outputs.py outputs/diagnostics results/trajectory_counterfactuals
"""
from __future__ import annotations

import io
import re
import subprocess
import sys

import numpy as np
from PIL import Image

NUM = re.compile(r"-?\d+\.?\d*(?:[eE][-+]?\d+)?")
TEXT_SUFFIXES = (".json", ".csv", ".tex", ".md", ".txt")


def git(*args: str) -> bytes:
    return subprocess.run(["git", *args], capture_output=True, check=True).stdout


def compare_png(path: str) -> str | None:
    ref = np.asarray(Image.open(io.BytesIO(git("show", f"HEAD:{path}"))).convert("RGB"))
    new = np.asarray(Image.open(path).convert("RGB"))
    if ref.shape != new.shape:
        return f"size {ref.shape[:2]} -> {new.shape[:2]}"
    differing = (ref != new).any(-1).mean()
    return f"{differing:.4%} of pixels differ" if differing else None


def compare_text(path: str) -> tuple[str | None, float]:
    ref = git("show", f"HEAD:{path}").decode().replace("\r\n", "\n")
    new = open(path, encoding="utf-8").read().replace("\r\n", "\n")
    if ref == new:
        return None, 0.0
    a, b = NUM.findall(ref), NUM.findall(new)
    if len(a) != len(b) or NUM.sub("#", ref) != NUM.sub("#", new):
        return "text or layout changed", float("inf")
    worst = max((abs(float(x) - float(y)) for x, y in zip(a, b)), default=0.0)
    return f"numbers differ by at most {worst:.1e}", worst


def main() -> None:
    paths = sys.argv[1:] or ["results/"]
    changed = git("diff", "--name-only", "--", *paths).decode().split()
    new_files = git("ls-files", "--others", "--exclude-standard", "--", *paths).decode().split()
    problems, noise = [], []
    for path in changed:
        if path.endswith(".png"):
            msg = compare_png(path)
            if msg:
                problems.append(f"{path}: {msg}")
        elif path.endswith(TEXT_SUFFIXES):
            msg, worst = compare_text(path)
            if msg:
                (noise if worst <= 1e-4 else problems).append(f"{path}: {msg}")
        else:
            problems.append(f"{path}: binary file changed")
    for path in new_files:
        problems.append(f"{path}: new file not in the reference outputs")

    print(f"Checked {len(changed)} changed file(s) under {' '.join(paths)}.")
    if noise:
        print(f"\n{len(noise)} file(s) differ only by floating-point noise (<= 1e-4):")
        print("\n".join(f"  {n}" for n in noise))
    if problems:
        print(f"\n{len(problems)} file(s) differ from the reference outputs:")
        print("\n".join(f"  {p}" for p in problems))
        sys.exit(1)
    print("\nAll regenerated outputs match the reference outputs "
          "(figures pixel-identical, numbers equal" + (" up to floating-point noise)." if noise else ")."))


if __name__ == "__main__":
    main()
