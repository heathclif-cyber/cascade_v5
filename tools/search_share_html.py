"""Search saved Grok share HTML for architecture keywords."""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
t = (ROOT / "agent-tools" / "grok-share-55e525f4.html").read_text(encoding="utf-8", errors="ignore")

keywords = [
    "Focused", "Residual", "Exhaustion", "H4 Primary", "H4 PRIMARY",
    "56 fitur", "10 fitur", "noise", "LGBM", "Guardian", "mlfinlab",
    "vectorbt", "finrl", "backtrader", "LightGBM", "purged",
    "MODAL", "5.0", "github.com", "Cascade v5", "Komprehensif",
    "trajectory", "momentum_boost", "0.68", "0.55",
]

for kw in keywords:
    idx = t.find(kw)
    if idx >= 0:
        print(f"--- {kw}")
        snippet = t[max(0, idx - 100) : idx + 300]
        snippet = snippet.replace("\\n", "\n").replace("\\u0026", "&")
        print(snippet[:500])
        print()

# decode \u escapes in meta alt
for m in re.finditer(r"Komprehensif Cascade v5.{0,500}", t):
    print("META:", m.group(0)[:300])