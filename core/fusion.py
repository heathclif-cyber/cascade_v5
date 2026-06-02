"""
core/fusion.py — Dynamic Fusion Layer + Smart Entry Gate

Menggabungkan output LGBM dan Momentum Expert menjadi final decision.
Menggantikan hard_consensus scalar hack dari Cascade v4.
"""

import numpy as np
from config import (
    LGBM_THRESHOLD_LONG, LGBM_THRESHOLD_SHORT,
    FUSION_LGBM_WEIGHT, FUSION_RESIDUAL_WEIGHT,
    FUSION_EXHAUSTION_PENALTY, FUSION_MOMENTUM_BOOST_THR,
    FUSION_MOMENTUM_BOOST_VAL,
    ENTRY_FINAL_PROB_THR, ENTRY_MOMENTUM_STR_THR,
    ENTRY_EXHAUSTION_MAX, ENTRY_HIGH_CONV_MOMENTUM_THR,
    ENTRY_HIGH_CONV_PROB_THR,
    H1_CONFIRMATION_ENABLED, H1_CONF_RSI_LONG_MIN,
    H1_CONF_RSI_SHORT_MAX, H1_CONF_RET_DIRECTION,
)


def dynamic_fusion(
    lgbm_long_prob:   float,
    lgbm_short_prob:  float,
    momentum_strength: float,
    exhaustion_score:  float,
    residual_correction: float,
) -> tuple[int, float, dict]:
    """
    Fusi output LGBM + Momentum Expert menjadi final direction & confidence.

    Returns:
        direction  : 0=SHORT, 1=FLAT, 2=LONG
        final_prob : confidence (0–1)
        debug      : dict untuk logging/analysis
    """
    # Momentum boost jika momentum sangat kuat
    momentum_boost = FUSION_MOMENTUM_BOOST_VAL if momentum_strength > FUSION_MOMENTUM_BOOST_THR else 0.0

    # Fusi LONG
    final_long = (
        FUSION_LGBM_WEIGHT    * lgbm_long_prob
        + FUSION_RESIDUAL_WEIGHT  * max(0.0, residual_correction)  # residual positif = LGBM underestimate LONG
        - FUSION_EXHAUSTION_PENALTY * exhaustion_score
        + momentum_boost
    )

    # Fusi SHORT
    final_short = (
        FUSION_LGBM_WEIGHT    * lgbm_short_prob
        + FUSION_RESIDUAL_WEIGHT  * max(0.0, -residual_correction)  # residual negatif = LGBM underestimate SHORT
        - FUSION_EXHAUSTION_PENALTY * exhaustion_score
        + momentum_boost
    )

    final_long  = float(np.clip(final_long,  0.0, 1.0))
    final_short = float(np.clip(final_short, 0.0, 1.0))

    debug = {
        "lgbm_long":   lgbm_long_prob,
        "lgbm_short":  lgbm_short_prob,
        "momentum":    momentum_strength,
        "exhaustion":  exhaustion_score,
        "residual":    residual_correction,
        "boost":       momentum_boost,
        "final_long":  final_long,
        "final_short": final_short,
    }

    # Determine direction
    if final_long >= final_short:
        return 2, final_long, debug
    else:
        return 0, final_short, debug


def smart_entry_gate(
    direction:         int,
    final_prob:        float,
    momentum_strength: float,
    exhaustion_score:  float,
    h1_rsi:            float = 50.0,
    h1_log_ret:        float = 0.0,
    h1_accel_sign:     float = 0.0,
) -> bool:
    """
    Smart Entry Gate — apakah trade boleh dieksekusi?

    H4 primary decision + H1 selective confirmation:
      Normal path    : final_prob > 0.68 AND momentum > 0.55 AND exhaustion < 0.60
      High-conviction: momentum > 0.85 AND exhaustion rendah → prob threshold lebih longgar
      H1 confirmation: RSI H1 dan arah return H1 harus align dengan sinyal H4

    Returns True jika trade boleh dieksekusi.
    """
    if direction == 1:  # FLAT
        return False

    # ── H1 Selective Confirmation ─────────────────────────────────────────────
    if H1_CONFIRMATION_ENABLED:
        if direction == 2:  # LONG
            # RSI H1 tidak boleh terlalu rendah (oversold → mungkin rebound tapi lemah)
            if h1_rsi < H1_CONF_RSI_LONG_MIN:
                return False
            # Return H1 terbaru searah (opsional, lebih longgar)
            if H1_CONF_RET_DIRECTION and h1_log_ret < -0.005:  # drop > 0.5% H1 = bad timing
                return False
        elif direction == 0:  # SHORT
            if h1_rsi > H1_CONF_RSI_SHORT_MAX:
                return False
            if H1_CONF_RET_DIRECTION and h1_log_ret > 0.005:  # pump > 0.5% H1 = bad timing
                return False

    # ── H4 Primary Gate ───────────────────────────────────────────────────────
    is_high_conviction = (
        momentum_strength > ENTRY_HIGH_CONV_MOMENTUM_THR
        and exhaustion_score < ENTRY_EXHAUSTION_MAX * 0.7
    )

    if is_high_conviction:
        return final_prob > ENTRY_HIGH_CONV_PROB_THR

    return (
        final_prob        > ENTRY_FINAL_PROB_THR
        and momentum_strength > ENTRY_MOMENTUM_STR_THR
        and exhaustion_score  < ENTRY_EXHAUSTION_MAX
    )


def batch_predict(
    lgbm_proba:          np.ndarray,           # (N, 3) — [p_short, p_flat, p_long]
    momentum_strength:   np.ndarray,           # (N,)
    exhaustion_score:    np.ndarray,           # (N,)
    residual_correction: np.ndarray,           # (N,)
    h1_rsi:              np.ndarray = None,    # (N,) — H1 confirmation (optional)
    h1_log_ret:          np.ndarray = None,    # (N,)
    h1_accel_sign:       np.ndarray = None,    # (N,)
) -> tuple[np.ndarray, np.ndarray]:
    """
    Batch fusion untuk seluruh H4 dataset.
    H1 confirmation arrays bersifat optional — di-upsample dari H1 ke H4 sebelum dipanggil.
    Returns: y_pred (N,), confidence (N,)
    """
    n = len(lgbm_proba)
    y_pred     = np.ones(n, dtype=np.int64)
    confidence = np.zeros(n, dtype=np.float64)

    _h1_rsi   = h1_rsi   if h1_rsi   is not None else np.full(n, 50.0)
    _h1_ret   = h1_log_ret if h1_log_ret is not None else np.zeros(n)
    _h1_accel = h1_accel_sign if h1_accel_sign is not None else np.zeros(n)

    for i in range(n):
        direction, final_prob, _ = dynamic_fusion(
            lgbm_long_prob      = float(lgbm_proba[i, 2]),
            lgbm_short_prob     = float(lgbm_proba[i, 0]),
            momentum_strength   = float(momentum_strength[i]),
            exhaustion_score    = float(exhaustion_score[i]),
            residual_correction = float(residual_correction[i]),
        )

        if smart_entry_gate(
            direction, final_prob,
            float(momentum_strength[i]),
            float(exhaustion_score[i]),
            h1_rsi       = float(_h1_rsi[i]),
            h1_log_ret   = float(_h1_ret[i]),
            h1_accel_sign= float(_h1_accel[i]),
        ):
            y_pred[i]     = direction
            confidence[i] = final_prob

    return y_pred, confidence
