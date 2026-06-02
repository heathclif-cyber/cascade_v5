"""Convert cascade_v5_jupyter.py -> Cascade_v5_Jupyter.ipynb (butuh jupytext)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = ROOT / "cascade_v5_jupyter.py"
IPYNB = ROOT / "Cascade_v5_Jupyter.ipynb"


def main() -> int:
    try:
        import jupytext  # noqa: F401
    except ImportError:
        print("Install: pip install jupytext")
        return 1
    subprocess.check_call(
        [sys.executable, "-m", "jupytext", "--to", "notebook", str(PY), "-o", str(IPYNB)]
    )
    print("Written:", IPYNB)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())