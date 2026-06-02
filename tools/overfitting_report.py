"""
overfitting_report.py — Generate overfitting detection report (Markdown + JSON).

Jalankan setelah training (04, 05c, 06) dan opsional holdout (07):

  python tools/overfitting_report.py
  python tools/overfitting_report.py --holdout-run models/runs/holdout_20260602_120000
  python tools/overfitting_report.py --out reports/my_overfit_check.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import MODEL_DIR, REPORT_DIR
from core.overfitting import build_report, render_markdown
from core.utils import setup_logger

logger = setup_logger("overfitting_report")


def main() -> int:
    parser = argparse.ArgumentParser(description="Cascade v5 overfitting report")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=MODEL_DIR,
        help="Folder models/ (default: config.MODEL_DIR)",
    )
    parser.add_argument(
        "--holdout-run",
        type=Path,
        default=None,
        help="Path ke models/runs/{run_id} (holdout_results.json)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output markdown path (default: reports/overfitting_YYYYMMDD_HHMMSS.md)",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Output JSON path (default: same stem as --out)",
    )
    args = parser.parse_args()

    report = build_report(args.model_dir, REPORT_DIR, args.holdout_run)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    if args.out is None:
        stamp = report.generated_at[:19].replace(":", "").replace("-", "")
        md_path = REPORT_DIR / f"overfitting_{stamp}.md"
    else:
        md_path = args.out
        md_path.parent.mkdir(parents=True, exist_ok=True)

    json_path = args.json_out or md_path.with_suffix(".json")

    md_text = render_markdown(report)
    md_path.write_text(md_text, encoding="utf-8")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2)

    logger.info(f"Overall status: {report.overall}")
    logger.info(f"Markdown: {md_path}")
    logger.info(f"JSON:     {json_path}")

    for c in report.checks:
        logger.info(f"  [{c.status:4}] {c.name}: {c.detail}")

    return 1 if report.overall == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())