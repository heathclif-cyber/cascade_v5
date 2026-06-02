"""
pipeline/shared.py — Shared utilities untuk pipeline training dan backtest
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from config import (
    LSTM_SEQ_LEN, LSTM_SEQUENCE_COLS, LGBM_FEATURE_COLS,
    N_FOLDS, PURGE_GAP_BARS,
)


def _build_purged_folds_ordinal(n_rows: int, n_folds: int, purge: int) -> list:
    """Expanding-window CV by row order (sequences / fallback)."""
    splits = np.array_split(np.arange(n_rows), n_folds + 1)
    folds = []
    for k in range(1, n_folds + 1):
        train_idx = np.concatenate(splits[:k])
        test_idx = splits[k]
        if len(train_idx) > purge:
            train_idx = train_idx[:-purge]
        if len(test_idx) > purge:
            test_idx = test_idx[purge:]
        if len(train_idx) > 0 and len(test_idx) > 0:
            folds.append((train_idx, test_idx))
    return folds


def build_purged_folds(
    time_index: pd.DatetimeIndex | pd.Index | np.ndarray,
    n_folds: int = N_FOLDS,
    purge: int = PURGE_GAP_BARS,
) -> list:
    """
    Build expanding-window folds with purging.

    - DatetimeIndex / datetime Index: purge di ruang timestamp (multi-coin H4).
    - ndarray integer / arange: purge di urutan baris (LSTM sequences).
    """
    if isinstance(time_index, np.ndarray):
        if time_index.dtype.kind in "iu" or (
            time_index.size > 0 and isinstance(time_index.flat[0], (int, np.integer))
        ):
            return _build_purged_folds_ordinal(len(time_index), n_folds, purge)

    if not isinstance(time_index, (pd.DatetimeIndex, pd.Index)):
        time_index = pd.Index(time_index)

    unique_ts = np.sort(time_index.unique())
    splits_ts = np.array_split(unique_ts, n_folds + 1)
    
    row_indices = np.arange(len(time_index))
    ts_to_idx = pd.Series(row_indices, index=time_index)
    
    folds = []
    for k in range(1, n_folds + 1):
        train_ts = np.concatenate(splits_ts[:k])
        test_ts = splits_ts[k]
        
        train_ts_purged = train_ts[:-purge] if len(train_ts) > purge else train_ts
        test_ts_purged = test_ts[purge:] if len(test_ts) > purge else test_ts
        
        # Ambil seluruh baris yang cocok dengan timestamp yang sudah dipurge
        train_idx = ts_to_idx.loc[train_ts_purged].values
        test_idx = ts_to_idx.loc[test_ts_purged].values
        
        if isinstance(train_idx, (int, np.integer)):
            train_idx = np.array([train_idx])
        elif len(train_idx.shape) > 1:
            train_idx = train_idx.flatten()
            
        if isinstance(test_idx, (int, np.integer)):
            test_idx = np.array([test_idx])
        elif len(test_idx.shape) > 1:
            test_idx = test_idx.flatten()
            
        folds.append((train_idx, test_idx))
    return folds


class SequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int = LSTM_SEQ_LEN):
        self.X       = torch.from_numpy(X.astype(np.float32))
        self.y       = torch.from_numpy(y.astype(np.int64))
        self.seq_len = seq_len
        self.indices = list(range(seq_len - 1, len(X)))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        end = self.indices[idx]
        return self.X[end - self.seq_len + 1: end + 1], self.y[end]

    def get_labels(self):
        return self.y[self.indices].numpy()


def upsample_h4_to_h1(
    values: np.ndarray,
    h4_index: pd.DatetimeIndex,
    h1_index: pd.DatetimeIndex,
) -> np.ndarray:
    """Forward-fill nilai H4 ke grid H1 (anti look-ahead sudah di 02_clean)."""
    s = pd.Series(np.asarray(values, dtype=np.float64), index=h4_index)
    return s.reindex(h1_index, method="ffill").fillna(0.0).values


def load_coin_simulation_data(
    symbol: str,
    label_dir: Path,
    proc_dir: Path,
) -> dict | None:
    """
    Muat frame H4 (LGBM/LSTM) + H1 (OHLC + konfirmasi gate).
    Returns None jika file wajib tidak ada.
    """
    from core.utils import ensure_utc_index

    h4_lgbm_path = label_dir / f"{symbol}_h4_lgbm.parquet"
    h4_lstm_path = label_dir / f"{symbol}_h4_lstm.parquet"
    h1_conf_path = label_dir / f"{symbol}_h1_conf.parquet"
    h1_proc_path = proc_dir / f"{symbol}_h1_clean.parquet"

    for p in (h4_lgbm_path, h4_lstm_path, h1_conf_path, h1_proc_path):
        if not p.exists():
            return None

    h4_lgbm = ensure_utc_index(pd.read_parquet(h4_lgbm_path))
    h4_lstm = ensure_utc_index(pd.read_parquet(h4_lstm_path))
    h1_conf = ensure_utc_index(pd.read_parquet(h1_conf_path))
    h1_proc = ensure_utc_index(pd.read_parquet(h1_proc_path))

    h1_index = h1_proc.index.intersection(h1_conf.index)
    if len(h1_index) == 0:
        h1_index = h1_proc.index
    h1_proc = h1_proc.loc[h1_index]
    h1_conf = h1_conf.reindex(h1_index).ffill().fillna(0)

    return {
        "h4_lgbm": h4_lgbm,
        "h4_lstm": h4_lstm,
        "h1_conf": h1_conf,
        "h1_proc": h1_proc,
        "h1_index": h1_index,
        "h4_index": h4_lgbm.index,
    }


def build_lgbm_feature_matrix(df: pd.DataFrame, feature_cols: list[str] | None = None) -> np.ndarray:
    """Matrix (N, n_features) dengan kolom hilang diisi 0."""
    cols = feature_cols or LGBM_FEATURE_COLS
    n = len(df)
    X = np.zeros((n, len(cols)), dtype=np.float64)
    for idx, col in enumerate(cols):
        if col in df.columns:
            X[:, idx] = df[col].ffill().fillna(0).values
    return X


def lgbm_proba_h4_to_h1(
    proba_h4: np.ndarray,
    h4_index: pd.DatetimeIndex,
    h1_index: pd.DatetimeIndex,
) -> np.ndarray:
    """Collapse 5-class H4 proba ke (N_h1, 3): [p_short, p_flat, p_long]."""
    p_short = upsample_h4_to_h1(proba_h4[:, 0] + proba_h4[:, 1], h4_index, h1_index)
    p_flat  = upsample_h4_to_h1(proba_h4[:, 2], h4_index, h1_index)
    p_long  = upsample_h4_to_h1(proba_h4[:, 3] + proba_h4[:, 4], h4_index, h1_index)
    out = np.stack([p_short, p_flat, p_long], axis=1)
    return np.clip(out, 0.0, 1.0)


def build_lstm_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    n = len(df)
    X = np.zeros((n, len(LSTM_SEQUENCE_COLS)), dtype=np.float32)
    for idx, col in enumerate(LSTM_SEQUENCE_COLS):
        if col in df.columns:
            X[:, idx] = df[col].ffill().fillna(0).values.astype(np.float32)
    return X


def infer_lstm_on_h4(
    lstm_model,
    df_h4_lstm: pd.DataFrame,
    seq_scaler=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Inference AttentionLSTM pada bar H4.
    Returns momentum, exhaustion, residual — masing-masing (N_h4,).
    """
    from core.models import infer_momentum_expert

    n = len(df_h4_lstm)
    X = build_lstm_feature_matrix(df_h4_lstm)
    if seq_scaler is not None:
        X = seq_scaler.transform(X).astype(np.float32)

    seq_len = LSTM_SEQ_LEN
    X_seq = np.zeros((n, seq_len, len(LSTM_SEQUENCE_COLS)), dtype=np.float32)
    for i in range(seq_len - 1, n):
        X_seq[i] = X[i - seq_len + 1: i + 1]

    mom, exh, res = infer_momentum_expert(lstm_model, X_seq)
    out_mom = np.full(n, 0.5, dtype=np.float32)
    out_exh = np.zeros(n, dtype=np.float32)
    out_res = np.zeros(n, dtype=np.float32)
    out_mom[seq_len - 1:] = mom[seq_len - 1:].astype(np.float32)
    out_exh[seq_len - 1:] = exh[seq_len - 1:].astype(np.float32)
    out_res[seq_len - 1:] = res[seq_len - 1:].astype(np.float32)
    return out_mom, out_exh, out_res


def build_guardian_static_matrix(
    df_h4_lgbm: pd.DataFrame,
    h4_index: pd.DatetimeIndex,
    h1_index: pd.DatetimeIndex,
) -> np.ndarray:
    """Static Guardian features = LGBM tabular upsampled H4 -> H1."""
    n_h1 = len(h1_index)
    X = np.zeros((n_h1, len(LGBM_FEATURE_COLS)), dtype=np.float64)
    for idx, col in enumerate(LGBM_FEATURE_COLS):
        if col in df_h4_lgbm.columns:
            X[:, idx] = upsample_h4_to_h1(
                df_h4_lgbm[col].ffill().fillna(0).values, h4_index, h1_index
            )
    return X


def h4_swing_levels_on_h1(
    df_h4_lstm: pd.DataFrame,
    h4_index: pd.DatetimeIndex,
    h1_index: pd.DatetimeIndex,
) -> tuple[np.ndarray, np.ndarray]:
    """Swing high/low H4 di-forward-fill ke H1 untuk simulasi TP/SL."""
    sh_col = "last_swing_high_price" if "last_swing_high_price" in df_h4_lstm.columns else None
    sl_col = "last_swing_low_price" if "last_swing_low_price" in df_h4_lstm.columns else None
    n = len(h1_index)
    if sh_col is None or sl_col is None:
        return np.full(n, np.nan), np.full(n, np.nan)
    sh = upsample_h4_to_h1(df_h4_lstm[sh_col].values, h4_index, h1_index)
    sl = upsample_h4_to_h1(df_h4_lstm[sl_col].values, h4_index, h1_index)
    return sh, sl
