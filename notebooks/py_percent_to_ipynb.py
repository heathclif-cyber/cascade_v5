"""Convert cascade_v5_jupyter.py (# %%) to Cascade_v5_Jupyter.ipynb tanpa jupytext."""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).parent
SRC = ROOT / "cascade_v5_jupyter.py"
DST = ROOT / "Cascade_v5_Jupyter.ipynb"

CELL_RE = re.compile(
    r"^# %%(?:\s*\[([^\]]+)\])?\s*$",
    re.MULTILINE,
)


def main() -> None:
    text = SRC.read_text(encoding="utf-8")
    parts = CELL_RE.split(text)
    # parts[0] may be preamble before first %%
    cells = []
    i = 1
    while i < len(parts):
        lang = (parts[i] or "code").strip().lower()
        body = parts[i + 1] if i + 1 < len(parts) else ""
        i += 2
        body = body.strip("\n")
        if lang == "markdown":
            cells.append({
                "cell_type": "markdown",
                "metadata": {},
                "source": [line + "\n" for line in body.split("\n")] or [""],
            })
        else:
            lines = body.split("\n")
            cells.append({
                "cell_type": "code",
                "metadata": {},
                "source": [ln + "\n" for ln in lines],
                "outputs": [],
                "execution_count": None,
            })

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.12.0"},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    DST.write_text(json.dumps(nb, indent=2), encoding="utf-8")
    print(f"Wrote {len(cells)} cells -> {DST}")


if __name__ == "__main__":
    main()