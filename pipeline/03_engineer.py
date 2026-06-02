"""
pipeline/03_engineer.py — Feature Engineering Split (Cascade v5)

Menghasilkan dua dataset terpisah per koin:
  data/labeled/{symbol}_h4_lgbm.parquet   → H4 tabular features + label_lgbm (5-class)
  data/labeled/{symbol}_h1_lstm.parquet   → H1 trajectory features (10 cols) + exhaustion_score

Jalankan:
  python pipeline/03_engineer.py --all
  python pipeline/03_engineer.py --coins SOLUSDT ETHUSDT
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
    LGBM_FEATURE_COLS, LSTM_SEQUENCE_COLS,
    TRAIN_CUTOFF_DATE,
)
from core.features import engineer_features_v5
from core.utils import setup_logger, ensure_utc_index

logger = setup_logger("03_engineer")


def _save(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df, preserve_index=True)
    pq.write_table(table, str(path), compression="snappy")


def engineer_symbol(symbol: str) -> bool:
    proc_h1 = PROC_DIR / f"{symbol}_h1_clean.parquet"
    proc_h4 = PROC_DIR / f"{symbol}_h4_clean.parquet"

    if not proc_h1.exists():
        logger.error(f"[{symbol}] H1 clean tidak ditemukan: {proc_h1}")
        return False
    if not proc_h4.exists():
        logger.error(f"[{symbol}] H4 clean tidak ditemukan: {proc_h4}")
        return False

    try:
        df_h1 = pd.read_parquet(proc_h1)
        df_h4 = pd.read_parquet(proc_h4)
        df_h1 = ensure_utc_index(df_h1)
        df_h4 = ensure_utc_index(df_h4)

        symbol_id = list(TRAINING_COINS).index(symbol) if symbol in TRAINING_COINS else -1

        h4_feat, h1_feat = engineer_features_v5(df_h1, df_h4, symbol, symbol_id)

        # ── H4: pilih LGBM cols + label ───────────────────────────────────────
        h4_cols = [c for c in LGBM_FEATURE_COLS if c in h4_feat.columns]
        missing_h4 = [c for c in LGBM_FEATURE_COLS if c not in h4_feat.columns]
        if missing_h4:
            logger.warning(f"[{symbol}] H4 missing cols ({len(missing_h4)}): {missing_h4[:5]}...")

        # Tambah kolom yang belum ada dengan 0
        for col in missing_h4:
            h4_feat[col] = 0.0

        h4_out = h4_feat[LGBM_FEATURE_COLS + ["label_lgbm"]].copy()
        h4_out = h4_out.dropna(subset=["close", "atr_14_h4"] if "atr_14_h4" in h4_out.columns else ["close"])

        # ── H1: pilih LSTM cols + exhaustion ─────────────────────────────────
        h1_cols = [c for c in LSTM_SEQUENCE_COLS if c in h1_feat.columns]
        missing_h1 = [c for c in LSTM_SEQUENCE_COLS if c not in h1_feat.columns]
        if missing_h1:
            logger.warning(f"[{symbol}] H1 missing cols ({len(missing_h1)}): {missing_h1}")
            for col in missing_h1:
                h1_feat[col] = 0.0

        h1_out = h1_feat[LSTM_SEQUENCE_COLS + ["exhaustion_score", "close", "atr_14_h1"]].copy()
        h1_out = h1_out.dropna(subset=["close"])

        # ── Save ──────────────────────────────────────────────────────────────
        out_h4 = LABEL_DIR / f"{symbol}_h4_lgbm.parquet"
        out_h1 = LABEL_DIR / f"{symbol}_h1_lstm.parquet"
        _save(h4_out, out_h4)
        _save(h1_out, out_h1)

        # Label distribution H4
        lc = h4_out["label_lgbm"].value_counts().sort_index()
        label_str = " | ".join([f"class{k}={v}" for k, v in lc.items()])
        logger.info(
            f"[{symbol}] H4={len(h4_out)} bars | H1={len(h1_out)} bars | "
            f"Labels: {label_str}"
        )
        return True

    except Exception as e:
        logger.error(f"[{symbol}] Error: {e}")
        logger.error(traceback.format_exc())
        return False


def main():
    parser = argparse.ArgumentParser(description="Feature Engineering v5")
    parser.add_argument("--all",   action="store_true", help="Semua coins")
    parser.add_argument("--coins", nargs="+",           help="Coin spesifik")
    args = parser.parse_args()

    coins = TRAINING_COINS if args.all else (args.coins or TRAINING_COINS[:3])
    logger.info(f"Engineering {len(coins)} coins...")

    success = sum(engineer_symbol(c) for c in coins)
    logger.info(f"Done: {success}/{len(coins)} coins OK")


if __name__ == "__main__":
    main()
