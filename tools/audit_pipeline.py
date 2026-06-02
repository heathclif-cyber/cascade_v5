"""
audit_pipeline.py — Audit data + kesiapan pipeline Cascade v5.

  python tools/audit_pipeline.py
  python tools/audit_pipeline.py --coins SOLUSDT ETHUSDT BNBUSDT
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import (
    RAW_DIR,
    PROC_DIR,
    LABEL_DIR,
    SEQ_DIR,
    MODEL_DIR,
    TRAINING_COINS,
    LGBM_FEATURE_COLS,
    LSTM_SEQUENCE_COLS,
)

FAIL = "FAIL"
WARN = "WARN"
PASS = "PASS"
SKIP = "SKIP"

rows: list[tuple[str, str, str]] = []


def row(cat: str, status: str, detail: str) -> None:
    rows.append((cat, status, detail))


def check_raw(coins: list[str]) -> None:
    row("RAW", PASS if RAW_DIR.exists() else FAIL, f"dir {RAW_DIR}")
    for sym in coins:
        for tf in ("1h", "4h"):
            p = RAW_DIR / "klines" / sym / f"{tf}_all.parquet"
            if p.exists() and p.stat().st_size > 1000:
                row("RAW", PASS, f"{sym} {tf} ({p.stat().st_size // 1024} KB)")
            else:
                row("RAW", FAIL, f"{sym} {tf} missing/empty: {p}")
        for aux in ("funding_rate", "open_interest"):
            p = RAW_DIR / aux / f"{sym}_{'8h' if aux == 'funding_rate' else '1h'}.parquet"
            if p.exists():
                row("RAW", PASS, f"{sym} {aux}")
            else:
                row("RAW", WARN, f"{sym} {aux} optional missing: {p.name}")
    for macro in ("fear_greed_index.parquet", "btc_dominance.parquet"):
        p = RAW_DIR / "macro" / macro
        row("RAW", PASS if p.exists() else WARN, f"macro {macro}")


def check_processed(coins: list[str]) -> None:
    for sym in coins:
        for tf in ("h1", "h4"):
            p = PROC_DIR / f"{sym}_{tf}_clean.parquet"
            if p.exists() and p.stat().st_size > 5000:
                row("PROC", PASS, f"{sym}_{tf}_clean ({p.stat().st_size // 1024} KB)")
            else:
                row("PROC", FAIL, f"missing: {p}")


def check_labeled(coins: list[str]) -> None:
    import pandas as pd

    for sym in coins:
        p = LABEL_DIR / f"{sym}_h4_lgbm.parquet"
        if not p.exists():
            row("LABEL", FAIL, f"missing {p.name}")
            continue
        try:
            df = pd.read_parquet(p)
            n = len(df)
            has_lbl = "label_lgbm" in df.columns
            n_feat = sum(c in df.columns for c in LGBM_FEATURE_COLS)
            row("LABEL", PASS if n > 100 and has_lbl else WARN,
                f"{sym}_h4_lgbm rows={n} label={has_lbl} lgbm_feats={n_feat}/{len(LGBM_FEATURE_COLS)}")
        except Exception as e:
            row("LABEL", FAIL, f"{sym}_h4_lgbm read error: {e}")

        p2 = LABEL_DIR / f"{sym}_h4_lstm.parquet"
        if p2.exists():
            row("LABEL", PASS, f"{sym}_h4_lstm OK")
        else:
            row("LABEL", FAIL, f"missing {sym}_h4_lstm.parquet")


def check_models() -> None:
    artifacts = [
        "lgbm_tabular.pkl",
        "lgbm_oof_predictions.npz",
        "lstm_momentum_expert.pt",
        "guardian_best.pkl",
    ]
    for name in artifacts:
        p = MODEL_DIR / name
        row("MODEL", PASS if p.exists() else SKIP, name)


def check_repo_patches() -> None:
    shared = (ROOT / "pipeline" / "shared.py").read_text(encoding="utf-8")
    if "_build_purged_folds_ordinal" in shared:
        row("PATCH", PASS, "shared.py purged CV fix")
    else:
        row("PATCH", FAIL, "shared.py LAMA — pull repo terbaru")

    cfg = (ROOT / "config.py").read_text(encoding="utf-8")
    if "_colab_lgbm_cpu" in cfg:
        row("PATCH", PASS, "config.py Colab LGBM CPU")
    else:
        row("PATCH", WARN, "config.py tanpa auto Colab CPU")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coins", nargs="+", default=["SOLUSDT", "ETHUSDT", "BNBUSDT"])
    args = parser.parse_args()
    coins = [c.upper() for c in args.coins]

    print("=" * 70)
    print("CASCADE V5 PIPELINE AUDIT")
    print(f"ROOT: {ROOT}")
    print(f"COINS: {coins}")
    print("=" * 70)

    check_repo_patches()
    check_raw(coins)
    check_processed(coins)
    check_labeled(coins)
    check_models()

    n_fail = sum(1 for _, s, _ in rows if s == FAIL)
    n_warn = sum(1 for _, s, _ in rows if s == WARN)
    for cat, st, det in rows:
        print(f"  [{st:4}] {cat:<6} {det}")

    print("-" * 70)
    print(f"FAIL={n_fail}  WARN={n_warn}  (PASS/SKIP tidak dihitung)")
    if n_fail:
        print("\nPerbaikan umum:")
        print("  1. python -u pipeline/01_fetch.py --coins ...   (tunggu log SELESAI)")
        print("  2. python -u pipeline/02_clean.py --coins ...")
        print("  3. python -u pipeline/03_engineer.py --coins ...")
        print("  Colab: pakai python -u atau notebook terbaru (run in-process).")
    print("=" * 70)
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())