"""
pipeline/05b_build_sequences.py — Build LSTM Sequence Datasets

Membangun sequence dataset (N, seq_len, n_features) untuk AttentionLSTM.
Hanya menggunakan LSTM_SEQUENCE_COLS (10 trajectory features H4).

Jalankan:
  python pipeline/05b_build_sequences.py --all
"""

import argparse, sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from config import (
    TRAINING_COINS, LABEL_DIR, SEQ_DIR,
    LSTM_SEQUENCE_COLS, LSTM_SEQ_LEN,
    TRAIN_CUTOFF_DATE,
)
from core.utils import setup_logger, ensure_utc_index

logger = setup_logger("05b_build_sequences")


def build_sequences_for_coin(symbol: str) -> bool:
    # H4 PRIMARY — baca dari h4_lstm (bukan h1_lstm)
    feat_path  = LABEL_DIR / f"{symbol}_h4_lstm.parquet"
    label_path = SEQ_DIR   / f"{symbol}_momentum_labels.parquet"

    if not feat_path.exists():
        logger.warning(f"[{symbol}] H4 LSTM features tidak ditemukan — jalankan 03_engineer dulu")
        return False
    if not label_path.exists():
        logger.warning(f"[{symbol}] momentum_labels tidak ditemukan — jalankan 05a dulu")
        return False

    df     = pd.read_parquet(feat_path)
    labels = pd.read_parquet(label_path)
    df     = ensure_utc_index(df)
    labels = ensure_utc_index(labels)
    df     = df[df.index < TRAIN_CUTOFF_DATE]

    # Align labels ke features
    labels = labels.reindex(df.index)

    # Pastikan semua LSTM_SEQUENCE_COLS ada
    avail_cols = [c for c in LSTM_SEQUENCE_COLS if c in df.columns]
    missing    = [c for c in LSTM_SEQUENCE_COLS if c not in df.columns]
    if missing:
        logger.warning(f"[{symbol}] Missing LSTM cols: {missing} — zero-filled")

    # Build feature matrix
    n = len(df)
    X = np.zeros((n, len(LSTM_SEQUENCE_COLS)), dtype=np.float32)
    for idx, col in enumerate(LSTM_SEQUENCE_COLS):
        if col in df.columns:
            X[:, idx] = df[col].ffill().fillna(0).values.astype(np.float32)

    # Scale tiap fitur (StandardScaler per fitur, fit pada training data saja)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X      = scaler.fit_transform(X).astype(np.float32)

    # Build sequences: (N - seq_len + 1, seq_len, n_features)
    seqs, idxs = [], []
    for i in range(LSTM_SEQ_LEN - 1, n):
        seq = X[i - LSTM_SEQ_LEN + 1: i + 1]  # (seq_len, n_features)
        seqs.append(seq)
        idxs.append(i)

    if len(seqs) == 0:
        logger.warning(f"[{symbol}] Tidak cukup data untuk sequence — skip")
        return False

    X_seq = np.stack(seqs, axis=0)  # (N_seq, seq_len, n_features)

    # Labels untuk setiap sequence endpoint
    momentum_labels   = labels["momentum_label"].values[idxs].astype(np.float32)
    exhaustion_labels = labels["exhaustion_label"].values[idxs].astype(np.float32)
    direction_labels  = labels["direction_label"].values[idxs].astype(np.int64)

    # Simpan
    out_path = SEQ_DIR / f"{symbol}_sequences.npz"
    np.savez_compressed(
        out_path,
        X_seq              = X_seq,
        momentum_labels    = momentum_labels,
        exhaustion_labels  = exhaustion_labels,
        direction_labels   = direction_labels,
        bar_indices        = np.array(idxs),
        feature_cols       = np.array(LSTM_SEQUENCE_COLS),
    )

    # Simpan scaler
    import joblib
    scaler_path = SEQ_DIR / f"{symbol}_seq_scaler.pkl"
    joblib.dump(scaler, scaler_path)

    logger.info(
        f"[{symbol}] Sequences built: {X_seq.shape} | "
        f"momentum mean={momentum_labels.mean():.3f} | "
        f"exhaustion rate={exhaustion_labels.mean():.3f}"
    )
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all",   action="store_true")
    parser.add_argument("--coins", nargs="+", default=[])
    args = parser.parse_args()

    coins = TRAINING_COINS if args.all else (args.coins or TRAINING_COINS[:3])

    success = 0
    for coin in coins:
        if build_sequences_for_coin(coin):
            success += 1

    logger.info(f"Done: {success}/{len(coins)} coins processed")


if __name__ == "__main__":
    main()
