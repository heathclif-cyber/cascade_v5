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
) -> bool:
    """
    Smart Entry Gate — apakah trade boleh dieksekusi?

    Normal path    : final_prob > 0.68 AND momentum > 0.55 AND exhaustion < 0.60
    High-conviction: momentum > 0.85 AND exhaustion rendah → prob threshold lebih longgar

    Returns True jika trade boleh dieksekusi.
    """
    if direction == 1:  # FLAT
        return False

    # High-conviction path: momentum sangat kuat, exhaustion rendah
    is_high_conviction = (
        momentum_strength > ENTRY_HIGH_CONV_MOMENTUM_THR
        and exhaustion_score < ENTRY_EXHAUSTION_MAX * 0.7  # exhaustion lebih ketat
    )

    if is_high_conviction:
        return final_prob > ENTRY_HIGH_CONV_PROB_THR

    # Normal path
    return (
        final_prob        > ENTRY_FINAL_PROB_THR
        and momentum_strength > ENTRY_MOMENTUM_STR_THR
        and exhaustion_score  < ENTRY_EXHAUSTION_MAX
    )


def batch_predict(
    lgbm_proba:          np.ndarray,  # (N, 3) — [p_short, p_flat, p_long]
    momentum_strength:   np.ndarray,  # (N,)
    exhaustion_score:    np.ndarray,  # (N,)
    residual_correction: np.ndarray,  # (N,)
) -> tuple[np.ndarray, np.ndarray]:
    """
    Batch fusion untuk seluruh dataset.
    Returns: y_pred (N,), confidence (N,)
    """
    n = len(lgbm_proba)
    y_pred     = np.ones(n, dtype=np.int64)    # default FLAT
    confidence = np.zeros(n, dtype=np.float64)

    for i in range(n):
        direction, final_prob, _ = dynamic_fusion(
            lgbm_long_prob       = float(lgbm_proba[i, 2]),
            lgbm_short_prob      = float(lgbm_proba[i, 0]),
            momentum_strength    = float(momentum_strength[i]),
            exhaustion_score     = float(exhaustion_score[i]),
            residual_correction  = float(residual_correction[i]),
        )

        if smart_entry_gate(direction, final_prob, float(momentum_strength[i]), float(exhaustion_score[i])):
            y_pred[i]     = direction
            confidence[i] = final_prob

    return y_pred, confidence
