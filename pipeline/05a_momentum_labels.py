"""
pipeline/05a_momentum_labels.py — Generate Momentum + Exhaustion Labels

Labels ini RULE-BASED, tidak ada forward scan → tidak ada circular leakage.

Label yang dihasilkan per bar:
  momentum_label    : float 0–1 (seberapa kuat momentum saat ini)
  exhaustion_label  : int 0/1 (apakah bar ini menunjukkan exhaustion signal)
  direction_label   : int 0/1/2 (SHORT/FLAT/LONG — dari swing labeling biasa)

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
    MOMENTUM_WINDOW, MOMENTUM_ATR_THRESH, MOMENTUM_VOL_THRESH,
    EXHAUSTION_VOL_DROP, EXHAUSTION_PRICE_ATR, EXHAUSTION_WICK_RATIO,
    LABEL_MAP, TRAIN_CUTOFF_DATE,
)
from core.utils import setup_logger, ensure_utc_index

logger = setup_logger("05a_momentum_labels")


def compute_momentum_label(
    close:   np.ndarray,
    volume:  np.ndarray,
    atr:     np.ndarray,
    window:  int = 5,
) -> np.ndarray:
    """
    Momentum strength label per bar — rule-based, backward-looking only.

    momentum = 0.5 * price_acceleration + 0.5 * volume_confirmation
    Tidak ada forward scan.
    """
    n = len(close)
    momentum = np.zeros(n, dtype=np.float32)

    for i in range(window, n):
        atr_i = atr[i]
        if atr_i == 0 or np.isnan(atr_i):
            continue

        # Price acceleration: seberapa cepat harga bergerak dalam window bar
        price_move = abs(close[i] - close[i - window])
        price_accel = min(price_move / (atr_i * window), 1.0)

        # Volume confirmation: apakah volume naik bersamaan dengan harga
        vol_mean = np.mean(volume[i - window:i])
        vol_ratio = volume[i] / vol_mean if vol_mean > 0 else 1.0
        vol_confirm = min(max((vol_ratio - 1.0) / 1.0, 0.0), 1.0)

        momentum[i] = 0.6 * price_accel + 0.4 * vol_confirm

    return momentum


def compute_exhaustion_label(
    close:       np.ndarray,
    high:        np.ndarray,
    low:         np.ndarray,
    volume:      np.ndarray,
    atr:         np.ndarray,
    swing_high:  np.ndarray,
    swing_low:   np.ndarray,
) -> np.ndarray:
    """
    Exhaustion score per bar — rule-based, tidak forward scan.

    Exhaustion = price sudah bergerak jauh + volume melemah + wick besar.
    Sinyal bahwa momentum sedang kehabisan tenaga.
    """
    n = len(close)
    exhaustion = np.zeros(n, dtype=np.float32)

    for i in range(1, n):
        atr_i = atr[i]
        if atr_i == 0 or np.isnan(atr_i):
            continue

        score = 0.0
        signals = 0

        # Signal 1: Volume melemah (volume bar ini < rata-rata 5 bar lalu)
        if i >= 5:
            vol_mean = np.mean(volume[i-5:i])
            if vol_mean > 0 and volume[i] / vol_mean < EXHAUSTION_VOL_DROP:
                score += 1.0
            signals += 1

        # Signal 2: Harga sudah jauh dari swing level
        sh = swing_high[i]
        sl = swing_low[i]
        if not np.isnan(sh) and not np.isnan(sl):
            dist_from_swing_high = (sh - close[i]) / atr_i
            dist_from_swing_low  = (close[i] - sl) / atr_i
            # Sangat dekat ke swing high (< 0.3 ATR) = over-extended untuk LONG
            if dist_from_swing_high < 0.3 or dist_from_swing_low < 0.3:
                score += 1.0
            signals += 1

        # Signal 3: Wick besar = ragu-ragu / rejection
        candle_range = high[i] - low[i]
        candle_body  = abs(close[i] - (close[i-1] if i > 0 else close[i]))
        if candle_range > 0:
            wick_ratio = 1.0 - (candle_body / candle_range)
            if wick_ratio > EXHAUSTION_WICK_RATIO:
                score += 1.0
        signals += 1

        # Signal 4: Price move sangat besar dalam 1 bar (spike exhaustion)
        single_bar_move = abs(close[i] - close[i-1]) / atr_i if i > 0 else 0
        if single_bar_move > EXHAUSTION_PRICE_ATR:
            score += 0.5
        signals += 1

        exhaustion[i] = min(score / signals, 1.0) if signals > 0 else 0.0

    return exhaustion


def process_coin(symbol: str) -> bool:
    path = LABEL_DIR / f"{symbol}_features_v3.parquet"
    if not path.exists():
        logger.warning(f"[{symbol}] features_v3.parquet tidak ditemukan — skip")
        return False

    df = pd.read_parquet(path)
    df = ensure_utc_index(df)
    df = df[df.index < TRAIN_CUTOFF_DATE]

    if len(df) < 100:
        logger.warning(f"[{symbol}] Data terlalu sedikit ({len(df)}) — skip")
        return False

    close  = df["close"].values.astype(np.float64)
    high   = df["high"].values.astype(np.float64)
    low    = df["low"].values.astype(np.float64)
    volume = df["volume"].values.astype(np.float64)
    atr    = df["atr_14_h1"].values.astype(np.float64) if "atr_14_h1" in df.columns else np.ones(len(df))
    sh     = df["h4_swing_high"].values if "h4_swing_high" in df.columns else np.full(len(df), np.nan)
    sl     = df["h4_swing_low"].values  if "h4_swing_low"  in df.columns else np.full(len(df), np.nan)

    # Compute labels
    momentum   = compute_momentum_label(close, volume, atr, MOMENTUM_WINDOW)
    exhaustion = compute_exhaustion_label(close, high, low, volume, atr, sh, sl)

    # Direction label dari swing-based labeling yang sudah ada
    direction = df["label"].map(LABEL_MAP).fillna(1).values.astype(np.int64)

    # Buat output DataFrame
    out = pd.DataFrame({
        "momentum_label":   momentum,
        "exhaustion_label": exhaustion,
        "direction_label":  direction,
    }, index=df.index)

    out_path = SEQ_DIR / f"{symbol}_momentum_labels.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(out, preserve_index=True)
    pq.write_table(table, str(out_path), compression="snappy")

    n_exhaustion = int((exhaustion > 0.5).sum())
    n_momentum_h = int((momentum > 0.6).sum())
    logger.info(
        f"[{symbol}] Labels saved — {len(df)} bars | "
        f"high_momentum={n_momentum_h} ({n_momentum_h/len(df):.1%}) | "
        f"exhaustion={n_exhaustion} ({n_exhaustion/len(df):.1%})"
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
        if process_coin(coin):
            success += 1

    logger.info(f"Done: {success}/{len(coins)} coins processed")


if __name__ == "__main__":
    main()
