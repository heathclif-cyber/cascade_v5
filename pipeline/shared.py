"""
pipeline/shared.py — Shared utilities untuk pipeline training dan backtest
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from config import LSTM_SEQ_LEN, N_FOLDS, PURGE_GAP_BARS


def build_purged_folds(df_index: pd.DatetimeIndex, n_folds: int = N_FOLDS, purge: int = PURGE_GAP_BARS) -> list:
    """
    Build expanding-window folds with purging in timestamp space.
    Pemisahan dilakukan pada level unique timestamps agar tidak terjadi
    overlap waktu antar koin pada batas fold.
    """
    unique_ts = np.sort(df_index.unique())
    splits_ts = np.array_split(unique_ts, n_folds + 1)
    
    row_indices = np.arange(len(df_index))
    ts_to_idx = pd.Series(row_indices, index=df_index)
    
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
