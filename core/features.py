"""
core/features.py — Cascade v5 Feature Engineering

Dua set fitur yang dipisah secara eksplisit:
  LGBM_FEATURE_COLS  : ~60 tabular + structure features (snapshot ok)
  LSTM_SEQUENCE_COLS : 10 trajectory features H4

Prinsip anti-leakage:
  - Semua fitur backward-looking (tidak ada shift negatif)
  - Swing high/low hanya pakai rolling max/min ke belakang (tidak center=True)
  - Tidak ada fitur yang menggunakan future price
"""

import numpy as np
import pandas as pd
from typing import Optional


# ─── Swing High / Low (Backward-Only, No Look-Ahead) ─────────────────────────

def compute_swing_levels(
    high:      pd.Series,
    low:       pd.Series,
    close:     pd.Series,
    atr:       pd.Series,
    left_bars: int = 5,
) -> pd.DataFrame:
    """
    Hitung swing high/low menggunakan hanya data historis.

    Pivot swing high: bar saat ini adalah max dari window (left_bars + 1) bar ke belakang.
    Tidak menggunakan center=True atau right bars — tidak ada look-ahead.

    Returns DataFrame dengan kolom:
      last_swing_high_price, last_swing_low_price,
      distance_from_recent_swing_high_atr, distance_from_recent_swing_low_atr
    """
    window = left_bars + 1   # bar saat ini + left_bars ke belakang

    # Rolling max/min — hanya ke belakang
    rolling_high = high.rolling(window, min_periods=window).max()
    rolling_low  = low.rolling(window,  min_periods=window).min()

    # Pivot: bar ini adalah local max/min dalam window ke belakang
    is_pivot_high = (high == rolling_high)
    is_pivot_low  = (low  == rolling_low)

    # Forward-fill last confirmed pivot level
    last_swing_high = high.where(is_pivot_high).ffill()
    last_swing_low  = low.where(is_pivot_low).ffill()

    atr_safe = atr.replace(0, np.nan).ffill().fillna(1.0)

    dist_high = (close - last_swing_high) / atr_safe   # negatif = price di bawah swing high
    dist_low  = (last_swing_low - close)  / atr_safe   # negatif = price di atas swing low

    return pd.DataFrame({
        "last_swing_high_price":                last_swing_high,
        "last_swing_low_price":                 last_swing_low,
        "distance_from_recent_swing_high_atr":  dist_high,
        "distance_from_recent_swing_low_atr":   dist_low,
    })


# ─── Trajectory Features untuk LSTM ──────────────────────────────────────────

def compute_trajectory_features(
    close:        pd.Series,
    high:         pd.Series,
    low:          pd.Series,
    volume:       pd.Series,
    atr:          pd.Series,
    funding_rate: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """
    Compute 10 trajectory features untuk LSTM.
    Semua backward-looking — tidak ada shift negatif.
    """
    feats = {}

    log_ret = np.log(close / close.shift(1)).fillna(0)

    # 1 & 2: Distance dari swing levels (pakai atr dari close)
    swing = compute_swing_levels(high, low, close, atr)
    feats["distance_from_recent_swing_high_atr"] = swing["distance_from_recent_swing_high_atr"]
    feats["distance_from_recent_swing_low_atr"]  = swing["distance_from_recent_swing_low_atr"]

    # 3: Run length up bars — berapa bar berturut-turut harga naik
    up_mask = (close > close.shift(1)).astype(int)
    run_len = up_mask.copy().astype(float)
    for i in range(1, len(run_len)):
        if up_mask.iloc[i] == 1:
            run_len.iloc[i] = run_len.iloc[i-1] + 1
        else:
            run_len.iloc[i] = 0.0
    feats["run_length_up_bars"] = run_len

    # 4: Acceleration sign change — apakah momentum baru berbalik arah
    momentum_5  = log_ret.rolling(5).sum()
    momentum_10 = log_ret.rolling(10).sum()
    prev_mom    = momentum_5.shift(3)
    sign_change = np.sign(momentum_5) != np.sign(prev_mom)
    feats["acceleration_sign_change"] = sign_change.astype(float).fillna(0)

    # 5: Log return acceleration dalam 5 bar
    ret_5     = log_ret.rolling(5).sum()
    ret_5_lag = ret_5.shift(5)
    feats["log_return_acceleration_5"] = (ret_5 - ret_5_lag).fillna(0)

    # 6: Momentum delta 5 — perubahan momentum 5 bar (2nd derivative of price)
    feats["momentum_delta_5"] = (momentum_5 - momentum_5.shift(5)).fillna(0)

    # 7: Funding extreme z-score
    if funding_rate is not None and not funding_rate.isna().all():
        fr = funding_rate.ffill().fillna(0)
        fr_mean = fr.rolling(168).mean()   # rolling 7-hari
        fr_std  = fr.rolling(168).std().replace(0, np.nan)
        feats["funding_extreme_zscore"] = ((fr - fr_mean) / fr_std).fillna(0)
    else:
        feats["funding_extreme_zscore"] = pd.Series(0.0, index=close.index)

    # 8: Price-funding divergence
    if funding_rate is not None and not funding_rate.isna().all():
        fr_sign    = np.sign(funding_rate.ffill().fillna(0))
        price_sign = np.sign(log_ret)
        feats["price_funding_divergence"] = (price_sign != fr_sign).astype(float)
    else:
        feats["price_funding_divergence"] = pd.Series(0.0, index=close.index)

    # 9: Volume acceleration — percepatan volume relatif ke rata-rata
    vol_mean_10 = volume.rolling(10).mean().replace(0, np.nan)
    vol_ratio   = (volume / vol_mean_10).fillna(1.0)
    feats["volume_acceleration"] = (vol_ratio - vol_ratio.shift(3)).fillna(0)

    # 10: Candle body size acceleration
    body_size     = (close - close.shift(1)).abs() / atr.replace(0, np.nan)
    body_mean_5   = body_size.rolling(5).mean()
    feats["candle_body_size_acceleration"] = (body_mean_5 - body_mean_5.shift(5)).fillna(0)

    return pd.DataFrame(feats, index=close.index)


# ─── H4 Structure Break Strength ─────────────────────────────────────────────

def compute_h4_structure_break_strength(
    close:     pd.Series,
    high:      pd.Series,
    low:       pd.Series,
    atr:       pd.Series,
    lookback:  int = 20,
) -> pd.Series:
    """
    Kekuatan break struktur H4 — seberapa jauh close menembus level sebelumnya.
    Backward-looking: bandingkan close saat ini vs highest high / lowest low N bar lalu.
    """
    prev_high = high.shift(1).rolling(lookback).max()
    prev_low  = low.shift(1).rolling(lookback).min()
    atr_safe  = atr.replace(0, np.nan).ffill().fillna(1.0)

    # Break up: close melebihi previous high
    break_up   = ((close - prev_high) / atr_safe).clip(lower=0)
    # Break down: close di bawah previous low
    break_down = ((prev_low - close) / atr_safe).clip(lower=0)

    # Strength = max dari keduanya, arah disesuaikan
    strength = break_up - break_down
    return strength.fillna(0).rename("H4_structure_break_strength")


# ─── Exhaustion Score (untuk label 05a dan Guardian) ─────────────────────────

def compute_exhaustion_score(
    close:      pd.Series,
    high:       pd.Series,
    low:        pd.Series,
    volume:     pd.Series,
    atr:        pd.Series,
    swing_high: pd.Series,
    swing_low:  pd.Series,
    dist_swing_high_atr: pd.Series,
    dist_swing_low_atr:  pd.Series,
    swing_atr_thr:  float = 3.5,
    vol_drop_thr:   float = 0.7,
    wick_ratio_thr: float = 0.5,
) -> pd.Series:
    """
    Rule-based exhaustion score per bar (0–1).
    Tidak ada forward scan — semua backward-looking.

    Komponen:
      1. Price over-extended (distance dari swing > threshold ATR)
      2. Volume melemah (vol ratio < threshold)
      3. Wick besar (rejection candle)
      4. Acceleration sign change (momentum berbalik)
    """
    n     = len(close)
    score = np.zeros(n, dtype=np.float32)
    atr_s = atr.values
    vol   = volume.values
    c     = close.values
    h     = high.values
    l     = low.values
    dsh   = dist_swing_high_atr.values
    dsl   = dist_swing_low_atr.values

    for i in range(10, n):
        if atr_s[i] == 0 or np.isnan(atr_s[i]):
            continue

        signals = 0
        fired   = 0.0

        # Signal 1: Over-extended dari swing
        if abs(dsh[i]) > swing_atr_thr or abs(dsl[i]) > swing_atr_thr:
            fired   += 1.0
        signals += 1

        # Signal 2: Volume melemah
        vol_mean = np.mean(vol[i-5:i]) if i >= 5 else vol[i]
        if vol_mean > 0 and vol[i] / vol_mean < vol_drop_thr:
            fired += 1.0
        signals += 1

        # Signal 3: Rejection wick besar
        candle_range = h[i] - l[i]
        candle_body  = abs(c[i] - c[i-1])
        if candle_range > 0 and (1.0 - candle_body / candle_range) > wick_ratio_thr:
            fired += 1.0
        signals += 1

        # Signal 4: Acceleration sign change
        if i >= 5:
            mom_now  = c[i]   - c[i-3]
            mom_prev = c[i-3] - c[i-6] if i >= 6 else 0.0
            if np.sign(mom_now) != np.sign(mom_prev) and mom_prev != 0:
                fired += 1.0
        signals += 1

        score[i] = min(fired / signals, 1.0)

    return pd.Series(score, index=close.index, name="exhaustion_score")


def compute_exhaustion_directional(
    dist_swing_high_atr: pd.Series,
    dist_swing_low_atr: pd.Series,
    acceleration_sign_change: pd.Series,
    swing_atr_thr: float = 3.5,
) -> pd.DataFrame:
    """
    Skor exhaustion arah (rencana share):
      exhaustion_long_score  — overextended ke swing high + percepatan berbalik turun
      exhaustion_short_score — overextended ke swing low  + percepatan berbalik naik

    LSTM tetap satu head `exhaustion_score` (agregat); kolom ini untuk label/audit.
    """
    asc = acceleration_sign_change.fillna(0)
    long_score = (
        (dist_swing_high_atr > swing_atr_thr) & (asc < 0)
    ).astype(np.float32)
    short_score = (
        (dist_swing_low_atr > swing_atr_thr) & (asc > 0)
    ).astype(np.float32)
    return pd.DataFrame(
        {
            "exhaustion_long_score":  long_score,
            "exhaustion_short_score": short_score,
        },
        index=dist_swing_high_atr.index,
    )


# ─── ATR-Based Return Label (LGBM 5-class, H4) ───────────────────────────────

def compute_lgbm_labels_h4(
    close:    pd.Series,
    atr:      pd.Series,
    horizon:  int   = 18,
    strong_thr: float = 1.8,
    weak_thr:   float = 0.75,
) -> pd.Series:
    """
    5-class ATR-normalized return label untuk LGBM (H4 timeframe).

    future_return_atr = (close[t+horizon] - close[t]) / ATR[t]

    Classes:
      -2: strong short (return < -strong_thr)
      -1: weak short   (-strong_thr <= return < -weak_thr)
       0: flat          (-weak_thr <= return <= weak_thr)
      +1: weak long    (weak_thr < return <= strong_thr)
      +2: strong long  (return > strong_thr)

    PENTING: shift(-horizon) digunakan HANYA untuk label generation,
    bukan sebagai input feature. Label hanya tersedia sampai TRAIN_CUTOFF_DATE.
    Bar dalam radius `horizon` dari cutoff dibuang (purge).
    """
    atr_safe = atr.replace(0, np.nan).ffill().fillna(close * 0.01)

    future_ret_atr = (close.shift(-horizon) - close) / atr_safe

    label = pd.Series(0, index=close.index, dtype=np.int64)
    label[future_ret_atr >  strong_thr] =  2
    label[(future_ret_atr >  weak_thr) & (future_ret_atr <=  strong_thr)] =  1
    label[(future_ret_atr < -strong_thr)] = -2
    label[(future_ret_atr < -weak_thr)  & (future_ret_atr >= -strong_thr)] = -1

    # Paksa FLAT untuk bar terakhir yang tidak punya forward window
    label.iloc[-horizon:] = 0

    return label.rename("label_lgbm")


# ─── H1 Confirmation Signals (Smart Entry Gate only) ─────────────────────────

def compute_h1_confirmation(
    close: pd.Series,
    high:  pd.Series,
    low:   pd.Series,
    atr:   pd.Series,
    funding_rate: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """
    Lightweight H1 signals untuk Smart Entry Gate — bukan input LSTM.
    Rule-based, backward-looking.

    Output kolom:
      h1_rsi         : RSI(14) H1
      h1_log_ret_1   : log return H1 bar terbaru
      h1_accel_sign  : +1 jika momentum H1 membangun, -1 jika melemah
    """
    # RSI H1
    delta  = close.diff()
    gain   = delta.clip(lower=0).ewm(span=14, adjust=False).mean()
    loss   = (-delta.clip(upper=0)).ewm(span=14, adjust=False).mean()
    rs     = gain / loss.replace(0, np.nan)
    h1_rsi = (100 - 100 / (1 + rs)).fillna(50).rename("h1_rsi")

    # Log return bar terbaru
    h1_ret = np.log(close / close.shift(1)).fillna(0).rename("h1_log_ret_1")

    # Acceleration sign: positif jika momentum 3 bar > momentum 3 bar sebelumnya
    mom_now  = (close - close.shift(3)).fillna(0)
    mom_prev = (close.shift(3) - close.shift(6)).fillna(0)
    h1_accel = np.sign(mom_now - mom_prev).rename("h1_accel_sign")

    return pd.DataFrame({"h1_rsi": h1_rsi, "h1_log_ret_1": h1_ret, "h1_accel_sign": h1_accel})


# ─── Main Engineer Function ───────────────────────────────────────────────────

def engineer_features_v5(
    df_h1:      pd.DataFrame,
    df_h4:      pd.DataFrame,
    symbol:     str,
    symbol_id:  int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Feature engineering split untuk Cascade v5.

    H4 adalah PRIMARY timeframe untuk semua model.
    H1 hanya menghasilkan confirmation signals untuk Smart Entry Gate.

    Returns:
      h4_lgbm : H4 DataFrame — LGBM_FEATURE_COLS + label_lgbm
      h4_lstm : H4 DataFrame — LSTM_SEQUENCE_COLS + exhaustion_score (untuk sequence builder)
      h1_conf : H1 DataFrame — H1_CONFIRMATION_COLS (lightweight, untuk gate saja)
    """
    from config import LGBM_FEATURE_COLS, LSTM_SEQUENCE_COLS, SWING_LEFT_BARS

    # ── Hitung ATR H4 ─────────────────────────────────────────────────────────
    h4 = df_h4.copy()
    if "atr_14_h4" not in h4.columns:
        tr = pd.concat([
            h4["high"] - h4["low"],
            (h4["high"] - h4["close"].shift(1)).abs(),
            (h4["low"]  - h4["close"].shift(1)).abs(),
        ], axis=1).max(axis=1)
        h4["atr_14_h4"] = tr.ewm(span=14, adjust=False).mean()

    atr_h4   = h4["atr_14_h4"]
    funding  = h4.get("funding_rate", None)

    # ── Swing levels H4 (backward-only) ──────────────────────────────────────
    swing_h4 = compute_swing_levels(h4["high"], h4["low"], h4["close"], atr_h4, SWING_LEFT_BARS)
    h4 = pd.concat([h4, swing_h4], axis=1)
    h4["dist_swing_high"] = h4["distance_from_recent_swing_high_atr"]
    h4["dist_swing_low"]  = h4["distance_from_recent_swing_low_atr"]

    # ── Structure break strength H4 ───────────────────────────────────────────
    h4["H4_structure_break_strength"] = compute_h4_structure_break_strength(
        h4["close"], h4["high"], h4["low"], atr_h4
    )

    # ── LGBM label (5-class, forward-looking — hanya untuk training) ──────────
    h4["label_lgbm"] = compute_lgbm_labels_h4(h4["close"], atr_h4)

    # ══════════════════════════════════════════════════════════════════════════
    # H4 LSTM: trajectory features dari H4 (bukan H1)
    # ══════════════════════════════════════════════════════════════════════════
    traj_h4 = compute_trajectory_features(
        close        = h4["close"],
        high         = h4["high"],
        low          = h4["low"],
        volume       = h4["volume"],
        atr          = atr_h4,
        funding_rate = funding,
    )
    h4_lstm = pd.concat([h4, traj_h4], axis=1)

    # Exhaustion score H4 (dipakai LSTM + Guardian)
    h4_lstm["exhaustion_score"] = compute_exhaustion_score(
        close               = h4["close"],
        high                = h4["high"],
        low                 = h4["low"],
        volume              = h4["volume"],
        atr                 = atr_h4,
        swing_high          = h4["last_swing_high_price"],
        swing_low           = h4["last_swing_low_price"],
        dist_swing_high_atr = h4["distance_from_recent_swing_high_atr"],
        dist_swing_low_atr  = h4["distance_from_recent_swing_low_atr"],
    )

    # ══════════════════════════════════════════════════════════════════════════
    # H1 CONFIRMATION: lightweight signals untuk Smart Entry Gate saja
    # Tidak dipakai sebagai input LSTM atau LGBM
    # ══════════════════════════════════════════════════════════════════════════
    h1 = df_h1.copy()
    if "atr_14_h1" not in h1.columns:
        tr = pd.concat([
            h1["high"] - h1["low"],
            (h1["high"] - h1["close"].shift(1)).abs(),
            (h1["low"]  - h1["close"].shift(1)).abs(),
        ], axis=1).max(axis=1)
        h1["atr_14_h1"] = tr.ewm(span=14, adjust=False).mean()

    h1_conf = compute_h1_confirmation(
        close        = h1["close"],
        high         = h1["high"],
        low          = h1["low"],
        atr          = h1["atr_14_h1"],
        funding_rate = h1.get("funding_rate", None),
    )
    h1_conf.index = h1.index

    return h4, h4_lstm, h1_conf
