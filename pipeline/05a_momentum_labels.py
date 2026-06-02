"""
pipeline/05a_momentum_labels.py — Momentum + Exhaustion Labels (Cascade v5)

Menghasilkan labels untuk LSTM training:
  momentum_label    : float 0–1 (kekuatan momentum saat ini, backward-only)
  exhaustion_label  : float 0–1 (max exhaustion long/short — share rencana)
  exhaustion_long_label / exhaustion_short_label : rule directional (3.5 ATR + accel)
  direction_label   : int 0/1/2 (SHORT/FLAT/LONG dari swing labeling)

ANTI-LEAKAGE:
  - Semua label dihitung dari data saat ini ke belakang
  - Tidak ada shift(-n) dalam label computation
  - exhaustion_label: distance > 3.5 ATR + acceleration sign change (rule-based)

Jalankan:
  python pipeline/05a_momentum_labels.py --all
  python pipeline/05a_momentum_labels.py --coins SOLUSDT ETHUSDT
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
    TRAINING_COINS, LABEL_DIR, SEQ_DIR,
    LSTM_SEQUENCE_COLS,
    EXHAUSTION_SWING_ATR_THR,
    LABEL_MAP, TRAIN_CUTOFF_DATE,
)
from core.features import compute_exhaustion_directional
from core.utils import setup_logger, ensure_utc_index

logger = setup_logger("05a_momentum_labels")


def compute_momentum_label(
    close:   np.ndarray,
    volume:  np.ndarray,
    atr:     np.ndarray,
    window:  int = 5,
) -> np.ndarray:
    """
    Momentum strength 0–1 per bar.
    Kombinasi: price acceleration (60%) + volume confirmation (40%).
    Backward-only — tidak ada look-ahead.
    """
    n   = len(close)
    out = np.zeros(n, dtype=np.float32)

    for i in range(window, n):
        atr_i = atr[i]
        if atr_i <= 0 or np.isnan(atr_i):
            continue

        # Price acceleration
        price_move  = abs(close[i] - close[i - window])
        price_accel = min(price_move / (atr_i * window), 1.0)

        # Volume confirmation
        vol_mean    = np.mean(volume[i - window:i])
        vol_confirm = min(max((volume[i] / vol_mean - 1.0), 0.0), 1.0) if vol_mean > 0 else 0.0

        out[i] = 0.6 * price_accel + 0.4 * vol_confirm

    return out


def process_coin(symbol: str) -> bool:
    # H4 adalah primary — baca dari h4_lstm (trajectory features H4)
    h4_path = LABEL_DIR / f"{symbol}_h4_lstm.parquet"
    if not h4_path.exists():
        logger.warning(f"[{symbol}] H4 LSTM file tidak ditemukan — jalankan 03_engineer dulu")
        return False

    df = pd.read_parquet(h4_path)
    df = ensure_utc_index(df)
    df = df[df.index < TRAIN_CUTOFF_DATE]

    if len(df) < 100:
        logger.warning(f"[{symbol}] Data terlalu sedikit ({len(df)}) — skip")
        return False

    close  = df["close"].values.astype(np.float64)
    atr    = df["atr_14_h1"].values.astype(np.float64) if "atr_14_h1" in df.columns else np.ones(len(df))
    volume = df.get("volume", pd.Series(1.0, index=df.index)).values.astype(np.float64)

    # ── Momentum label ────────────────────────────────────────────────────────
    momentum = compute_momentum_label(close, volume, atr, window=5)

    # ── Exhaustion label — menggunakan compute_exhaustion_score dari core/features.py
    # Butuh distance dari swing — ambil dari LSTM sequence cols
    dist_sh = (
        df["distance_from_recent_swing_high_atr"]
        if "distance_from_recent_swing_high_atr" in df.columns
        else pd.Series(0.0, index=df.index)
    )
    dist_sl = (
        df["distance_from_recent_swing_low_atr"]
        if "distance_from_recent_swing_low_atr" in df.columns
        else pd.Series(0.0, index=df.index)
    )
    accel = (
        df["acceleration_sign_change"]
        if "acceleration_sign_change" in df.columns
        else pd.Series(0.0, index=df.index)
    )

    n = len(df)
    directional = compute_exhaustion_directional(
        dist_sh, dist_sl, accel, swing_atr_thr=EXHAUSTION_SWING_ATR_THR
    )
    exh_long = directional["exhaustion_long_score"].values
    exh_short = directional["exhaustion_short_score"].values
    # Label LSTM: agregat arah (max long/short) — selaras unified exhaustion_head
    exh = np.maximum(exh_long, exh_short)

    # ── Direction label (3-class dari swing labeling) ─────────────────────────
    # Ambil dari label yang ada di h4 (downsampled) atau pakai FLAT default
    dir_label = np.ones(n, dtype=np.int64)   # default FLAT

    # ── Save ──────────────────────────────────────────────────────────────────
    SEQ_DIR.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame({
        "momentum_label":          momentum,
        "exhaustion_label":        exh,
        "exhaustion_long_label":   exh_long,
        "exhaustion_short_label":  exh_short,
        "direction_label":         dir_label,
    }, index=df.index)

    out_path = SEQ_DIR / f"{symbol}_momentum_labels.parquet"
    table = pa.Table.from_pandas(out, preserve_index=True)
    pq.write_table(table, str(out_path), compression="snappy")

    high_mom = int((momentum > 0.6).sum())
    high_exh = int((exh > 0.5).sum())
    logger.info(
        f"[{symbol}] {n} bars | momentum_high={high_mom} ({high_mom/n:.1%}) | "
        f"exhaustion_high={high_exh} ({high_exh/n:.1%})"
    )
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all",   action="store_true")
    parser.add_argument("--coins", nargs="+", default=[])
    args = parser.parse_args()

    coins = TRAINING_COINS if args.all else (args.coins or TRAINING_COINS[:3])
    success = sum(process_coin(c) for c in coins)
    logger.info(f"Done: {success}/{len(coins)} coins")


if __name__ == "__main__":
    main()
