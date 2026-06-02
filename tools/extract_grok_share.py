"""Extract meta / github links from saved Grok share HTML."""
from __future__ import annotations

import html
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    path = ROOT / "agent-tools" / "grok-share-55e525f4.html"
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
    text = path.read_text(encoding="utf-8", errors="ignore")

    for name in ("og:description", "twitter:description", "twitter:image:alt", "title"):
        for m in re.finditer(
            rf'(?:name|property)="{re.escape(name)}"[^>]*content="([^"]{{20,2000}})"',
            text,
        ):
            print(f"--- {name}")
            print(html.unescape(m.group(1))[:1200])
            print()

    urls = sorted(set(re.findall(r"https://github\.com/[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+", text)))
    print(f"github urls: {len(urls)}")
    for u in urls:
        print(u)

    # RSC / embedded strings mentioning Cascade
    for kw in ("Cascade v5", "H4 PRIMARY", "56 fitur", "LightGBM", "mlfinlab", "vectorbt"):
        if kw.lower() in text.lower():
            print(f"found keyword: {kw}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())