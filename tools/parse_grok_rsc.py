"""Parse Next.js RSC payloads from saved Grok share HTML."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    path = ROOT / "agent-tools" / "grok-share-55e525f4.html"
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
    text = path.read_text(encoding="utf-8", errors="ignore")

    chunks = re.findall(r'self\.__next_f\.push\(\[1,"((?:\\.|[^"\\])*)"\]\)', text)
    print(f"RSC chunks: {len(chunks)}")

    keywords = (
        "Cascade", "H4", "H1", "LGBM", "LSTM", "github", "LightGBM",
        "mlfinlab", "vectorbt", "fusion", "Guardian", "MODAL", "56",
        "Exhaustion", "Residual", "Momentum", "purged", "5.0",
    )
    hits: list[str] = []
    for raw in chunks:
        try:
            decoded = json.loads(f'"{raw}"')
        except json.JSONDecodeError:
            decoded = raw.replace("\\n", "\n").replace('\\"', '"')
        low = decoded.lower()
        if any(k.lower() in low for k in keywords):
            if len(decoded) > 80:
                hits.append(decoded)

    out = ROOT / "agent-tools" / "grok-share-rsc-extract.txt"
    out.write_text("\n\n---CHUNK---\n\n".join(hits[:200]), encoding="utf-8")
    print(f"Wrote {len(hits)} keyword chunks -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())