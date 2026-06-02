"""
pipeline/05d_oof_residuals.py — OOF Residual Computation (Cascade v5)

Menghitung residual dari LGBM OOF predictions untuk LSTM residual head training.

residual[i] = actual_return_normalized[i] - lgbm_predicted_return[i]

Dimana:
  actual_return_normalized = future_return_atr (ATR-normalized, dari label H4)
  lgbm_predicted_return    = weighted_sum(lgbm_oof_proba × class_values)

Protocol anti-leakage:
  - LGBM OOF predictions sudah disimpan saat 04_train_lgbm.py
  - Tidak ada overlap antara train dan val saat OOF computation
  - Residual per sequence endpoint = bar H4 (bar_indices dari 05b)

Jalankan (setelah 04_train_lgbm.py):
  python pipeline/05d_oof_residuals.py --all
"""

import argparse, sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from config import (
    TRAINING_COINS, LABEL_DIR, SEQ_DIR, MODEL_DIR,
    LGBM_LABEL_MAP_INV, LGBM_NUM_CLASSES,
    LGBM_STRONG_THR, LGBM_WEAK_THR,
    TRAIN_CUTOFF_DATE,
)
from core.utils import setup_logger, ensure_utc_index

logger = setup_logger("05d_oof_residuals")

# Nilai numerik tiap class (-2, -1, 0, +1, +2) sebagai "predicted return" proxy
CLASS_VALUES = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)  # index 0..4


def compute_residuals() -> dict[str, np.ndarray]:
    """
    Hitung residual per sequence endpoint (bar H4) untuk semua koin.
    Returns dict: {symbol: residual_array (N_sequences,)}
    """
    oof_path = MODEL_DIR / "lgbm_oof_predictions.npz"
    if not oof_path.exists():
        raise FileNotFoundError(
            "lgbm_oof_predictions.npz tidak ditemukan. Jalankan 04_train_lgbm.py dulu."
        )

    oof_data = np.load(oof_path)
    oof_proba = oof_data["oof_proba"]   # (N_total, 5)
    y_true    = oof_data["y_true"]      # (N_total,) — class index 0..4

    # Expected return dari OOF proba
    oof_expected = (oof_proba * CLASS_VALUES).sum(axis=1)  # (N_total,)

    # Actual return (dari label)
    actual_return = np.array([CLASS_VALUES[int(yi)] for yi in y_true], dtype=np.float32)

    # Residual = actual - predicted
    residual_global = actual_return - oof_expected

    logger.info(f"OOF residual — mean={residual_global.mean():.4f}  std={residual_global.std():.4f}")

    # Simpan per koin ke sequences dir
    results = {}
    offset  = 0

    for symbol in TRAINING_COINS:
        h4_path  = LABEL_DIR / f"{symbol}_h4_lgbm.parquet"
        seq_path = SEQ_DIR   / f"{symbol}_sequences.npz"

        if not h4_path.exists() or not seq_path.exists():
            continue

        df_h4 = pd.read_parquet(h4_path)
        df_h4 = df_h4[df_h4.index < TRAIN_CUTOFF_DATE]
        n_h4  = len(df_h4)

        seq_data  = np.load(seq_path, allow_pickle=True)
        n_seq     = len(seq_data["X_seq"])

        # Slice residual untuk koin ini
        coin_residual_h4 = residual_global[offset: offset + n_h4]
        offset += n_h4

        # bar_indices = index bar H4 pada endpoint sequence (dari 05b)
        bar_indices = seq_data["bar_indices"].astype(int)
        bar_indices = np.minimum(bar_indices, len(coin_residual_h4) - 1)
        seq_residuals = coin_residual_h4[bar_indices].astype(np.float32)
        results[symbol] = seq_residuals

        # Update sequences.npz dengan residual
        X_seq    = seq_data["X_seq"]
        mom_labs = seq_data["momentum_labels"]
        exh_labs = seq_data["exhaustion_labels"]
        dir_labs = seq_data["direction_labels"]

        np.savez_compressed(
            seq_path,
            X_seq              = X_seq,
            momentum_labels    = mom_labs,
            exhaustion_labels  = exh_labs,
            direction_labels   = dir_labs,
            residual_labels    = seq_residuals,
            bar_indices        = bar_indices,
            feature_cols       = seq_data["feature_cols"],
        )
        logger.info(
            f"[{symbol}] residual mean={seq_residuals.mean():.4f}  "
            f"std={seq_residuals.std():.4f}  n={n_seq}"
        )

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    logger.info("Computing OOF residuals...")
    results = compute_residuals()
    logger.info(f"Done: {len(results)} coins updated")


if __name__ == "__main__":
    main()
