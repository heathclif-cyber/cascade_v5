"""
pipeline/03_engineer.py — Feature Engineering Split (Cascade v5)

H4 adalah PRIMARY timeframe. Output tiga file per koin:
  data/labeled/{symbol}_h4_lgbm.parquet   → H4 tabular features + label_lgbm (5-class)
  data/labeled/{symbol}_h4_lstm.parquet   → H4 trajectory features (10 cols) + exhaustion_score
  data/labeled/{symbol}_h1_conf.parquet   → H1 confirmation signals (3 cols, untuk gate saja)

Jalankan:
  python pipeline/03_engineer.py --all
  python pipeline/03_engineer.py --coins SOLUSDT ETHUSDT
  python pipeline/03_engineer.py --all --holdout
"""

import argparse, sys, warnings, traceback
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from config import (
    TRAINING_COINS, PROC_DIR, LABEL_DIR,
    LGBM_FEATURE_COLS, LSTM_SEQUENCE_COLS, H1_CONFIRMATION_COLS,
)
from core.features import engineer_features_v5
from core.utils import setup_logger, ensure_utc_index

logger = setup_logger("03_engineer")

HOLDOUT_PROC  = ROOT / "data" / "holdout" / "processed"
HOLDOUT_LABEL = ROOT / "data" / "holdout" / "labeled"


def _save(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df, preserve_index=True)
    pq.write_table(table, str(path), compression="snappy")


def engineer_symbol(symbol: str, proc_dir: Path, label_dir: Path) -> bool:
    h1_path = proc_dir / f"{symbol}_h1_clean.parquet"
    h4_path = proc_dir / f"{symbol}_h4_clean.parquet"

    if not h1_path.exists():
        logger.error(f"[{symbol}] H1 clean tidak ditemukan: {h1_path}")
        return False
    if not h4_path.exists():
        logger.error(f"[{symbol}] H4 clean tidak ditemukan: {h4_path}")
        return False

    try:
        df_h1 = pd.read_parquet(h1_path)
        df_h4 = pd.read_parquet(h4_path)
        df_h1 = ensure_utc_index(df_h1)
        df_h4 = ensure_utc_index(df_h4)

        symbol_id = list(TRAINING_COINS).index(symbol) if symbol in TRAINING_COINS else -1

        # engineer_features_v5 returns (h4_lgbm, h4_lstm, h1_conf)
        h4_lgbm, h4_lstm, h1_conf = engineer_features_v5(df_h1, df_h4, symbol, symbol_id)

        # ── H4 LGBM: tabular features + label ────────────────────────────────
        for col in LGBM_FEATURE_COLS:
            if col not in h4_lgbm.columns:
                h4_lgbm[col] = 0.0

        h4_lgbm_out = h4_lgbm[LGBM_FEATURE_COLS + ["label_lgbm"]].copy()
        h4_lgbm_out = h4_lgbm_out.dropna(subset=["close"])

        # ── H4 LSTM: trajectory features + exhaustion ─────────────────────────
        for col in LSTM_SEQUENCE_COLS:
            if col not in h4_lstm.columns:
                logger.warning(f"  [{symbol}] LSTM col missing: {col}")
                h4_lstm[col] = 0.0

        extra_cols = ["exhaustion_score", "close", "atr_14_h4",
                      "h4_swing_high" if "h4_swing_high" in h4_lstm.columns else "last_swing_high_price",
                      "h4_swing_low"  if "h4_swing_low"  in h4_lstm.columns else "last_swing_low_price"]
        extra_cols = [c for c in extra_cols if c in h4_lstm.columns]

        h4_lstm_out = h4_lstm[LSTM_SEQUENCE_COLS + extra_cols].copy()
        h4_lstm_out = h4_lstm_out.dropna(subset=["close"])

        # ── H1 Confirmation: 3 lightweight signals ────────────────────────────
        h1_conf_out = h1_conf.copy()

        # ── Save ──────────────────────────────────────────────────────────────
        _save(h4_lgbm_out, label_dir / f"{symbol}_h4_lgbm.parquet")
        _save(h4_lstm_out, label_dir / f"{symbol}_h4_lstm.parquet")
        _save(h1_conf_out, label_dir / f"{symbol}_h1_conf.parquet")

        # Label distribution
        lc = h4_lgbm_out["label_lgbm"].value_counts().sort_index()
        lc_str = " | ".join([f"c{k}={v}" for k, v in lc.items()])

        logger.info(
            f"[{symbol}] H4_LGBM={len(h4_lgbm_out)} | "
            f"H4_LSTM={len(h4_lstm_out)} | "
            f"H1_CONF={len(h1_conf_out)} | "
            f"Labels: {lc_str}"
        )
        return True

    except Exception as e:
        logger.error(f"[{symbol}] Error: {e}")
        logger.error(traceback.format_exc())
        return False


def main():
    parser = argparse.ArgumentParser(description="Feature Engineering v5 (H4 Primary)")
    parser.add_argument("--all",     action="store_true")
    parser.add_argument("--coins",   nargs="+")
    parser.add_argument("--holdout", action="store_true",
                        help="Engineer holdout data dari data/holdout/processed/")
    args = parser.parse_args()

    if args.all:
        coins = TRAINING_COINS
    elif args.coins:
        coins = [c.upper() for c in args.coins]
    else:
        coins = TRAINING_COINS[:3]

    if args.holdout:
        proc_dir  = HOLDOUT_PROC
        label_dir = HOLDOUT_LABEL
        logger.info("Mode: HOLDOUT")
    else:
        proc_dir  = PROC_DIR
        label_dir = LABEL_DIR
        logger.info("Mode: TRAINING")

    label_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Engineering {len(coins)} coins (H4 primary)...")

    success = sum(engineer_symbol(c, proc_dir, label_dir) for c in coins)
    logger.info(f"Done: {success}/{len(coins)} coins OK")
    if success == 0:
        logger.error("Engineer gagal semua koin — cek 02_clean / raw data")
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
