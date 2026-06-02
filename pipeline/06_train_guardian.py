"""
pipeline/06_train_guardian.py — Guardian v3.5 Training (Cascade v5)

Multiclass LGBM: 0=HOLD, 1=PARTIAL_EXIT, 2=FULL_EXIT.
Fitur: static (dari LGBM_FEATURE_COLS) + dynamic (per-bar trade context) +
       exhaustion_score + momentum_strength (baru di v3.5).

Label generation:
  - Jalankan cascade (LGBM + LSTM) untuk mendapat entry signals
  - Simulate trades untuk dapat trade timelines
  - Per in-trade bar: assign HOLD/PARTIAL/FULL berdasarkan future PnL

Jalankan:
  python pipeline/06_train_guardian.py --all
"""

import argparse, json, sys, warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from config import (
    TRAINING_COINS, LABEL_DIR, MODEL_DIR, PROC_DIR,
    LGBM_FEATURE_COLS,
    GUARDIAN_DYNAMIC_FEATURES, GUARDIAN_EXIT_THRESHOLD,
    GUARDIAN_N_FOLDS, GUARDIAN_PURGE_GAP_BARS, GUARDIAN_EARLY_STOPPING,
    GUARDIAN_PARTIAL_EXIT_RATIO,
    MODAL_PER_TRADE, LEVERAGE_SIM, FEE_PER_SIDE, SLIPPAGE_PER_SIDE,
    MAX_HOLDING_BARS, SWING_LABEL_MIN_RR, SWING_LABEL_MIN_TP, SWING_LABEL_MAX_SL,
    TP_SL_FALLBACK_TP, TP_SL_FALLBACK_SL,
    TRAIN_CUTOFF_DATE,
)
from core.evaluator import simulate_trades_swing
from core.utils import setup_logger
from pipeline.shared import (
    build_purged_folds,
    load_coin_simulation_data,
    build_lgbm_feature_matrix,
    lgbm_proba_h4_to_h1,
    upsample_h4_to_h1,
    h4_swing_levels_on_h1,
    build_guardian_static_matrix,
)

logger = setup_logger("06_train_guardian")

GUARDIAN_LGBM_PARAMS = {
    "objective":         "multiclass",
    "num_class":         3,
    "n_estimators":      500,
    "learning_rate":     0.05,
    "max_depth":         6,
    "num_leaves":        31,
    "min_child_samples": 50,
    "subsample":         0.8,
    "colsample_bytree":  0.8,
    "verbose":           -1,
    "n_jobs":            -1,
    "random_state":      42,
    "device_type":       "gpu",
    "gpu_platform_id":   0,
    "gpu_device_id":     0,
}

from config import apply_colab_settings, _colab_lgbm_cpu

if _colab_lgbm_cpu():
    apply_colab_settings()
    GUARDIAN_LGBM_PARAMS["device_type"] = "cpu"
    GUARDIAN_LGBM_PARAMS.pop("gpu_platform_id", None)
    GUARDIAN_LGBM_PARAMS.pop("gpu_device_id", None)
    GUARDIAN_LGBM_PARAMS["n_jobs"] = 2


def load_models():
    lgbm = joblib.load(MODEL_DIR / "lgbm_tabular.pkl")
    with open(MODEL_DIR / "lgbm_feature_cols.json") as f:
        feat_cols = json.load(f)
    return lgbm, feat_cols


def _compute_guardian_dynamic(
    bars_held: int,
    entry_price: float,
    current_price: float,
    direction: int,   # 2=LONG, 0=SHORT
    atr_entry: float,
    mfe_pnl: float,
    exhaustion: float = 0.0,
    momentum: float   = 0.0,
) -> np.ndarray:
    """Compute 9 dynamic features per bar (7 original + 2 v3.5)."""
    atr_pct       = atr_entry / entry_price if entry_price > 0 else 0.01
    current_pnl   = ((current_price - entry_price) / entry_price
                     if direction == 2
                     else (entry_price - current_price) / entry_price)
    dd_from_peak  = (mfe_pnl - current_pnl) / mfe_pnl if mfe_pnl > 0.001 else 0.0
    entry_ratio   = entry_price / current_price if current_price > 0 else 1.0

    return np.array([
        bars_held / MAX_HOLDING_BARS,    # bars_held_norm
        current_pnl,                     # current_pnl_pct
        current_pnl / atr_pct if atr_pct > 0 else 0.0,  # current_pnl_atr
        mfe_pnl,                         # max_favorable_pnl_pct
        dd_from_peak,                    # drawdown_from_peak_pct
        1.0 if direction == 2 else 0.0,  # direction
        entry_ratio,                     # entry_price_ratio
        exhaustion,                      # exhaustion_score (v3.5)
        momentum,                        # momentum_strength (v3.5)
    ], dtype=np.float64)


def generate_labels_for_coin(
    symbol: str,
    lgbm_model,
    feat_cols: list[str],
) -> list[dict]:
    """
    Simulate trades pada satu koin → hasilkan per-bar labeled samples.
    Label: 0=HOLD, 1=PARTIAL_EXIT, 2=FULL_EXIT

    LGBM inference di H4; simulasi trade di H1 (OHLC dari processed).
    """
    frames = load_coin_simulation_data(symbol, LABEL_DIR, PROC_DIR)
    if frames is None:
        logger.warning(f"[{symbol}] Data tidak ditemukan — jalankan 02_clean + 03_engineer")
        return []

    df_h4   = frames["h4_lgbm"][frames["h4_lgbm"].index < TRAIN_CUTOFF_DATE]
    df_h4l  = frames["h4_lstm"].reindex(df_h4.index).ffill()
    h1_proc = frames["h1_proc"][frames["h1_proc"].index < TRAIN_CUTOFF_DATE]
    h1_index = h1_proc.index
    h4_index = df_h4.index

    if len(df_h4) < 50 or len(h1_proc) < 100:
        return []

    proba_h4 = lgbm_model.predict(build_lgbm_feature_matrix(df_h4, feat_cols))
    lgbm_proba_3 = lgbm_proba_h4_to_h1(proba_h4, h4_index, h1_index)
    p_long  = lgbm_proba_3[:, 2]
    p_short = lgbm_proba_3[:, 0]
    n       = len(h1_index)
    y_pred  = np.ones(n, dtype=np.int64)
    y_pred[p_long > 0.50]  = 2
    y_pred[p_short > 0.50] = 0
    confidence = np.maximum(p_long, p_short)

    close = h1_proc["close"].values
    high  = h1_proc["high"].values if "high" in h1_proc.columns else close
    low   = h1_proc["low"].values  if "low" in h1_proc.columns else close
    atr   = (
        h1_proc["atr_14_h1"].values
        if "atr_14_h1" in h1_proc.columns
        else np.ones(n)
    )
    sh_arr, sl_arr = h4_swing_levels_on_h1(df_h4l, h4_index, h1_index)
    if "exhaustion_score" in df_h4l.columns:
        exh = upsample_h4_to_h1(
            df_h4l["exhaustion_score"].values, h4_index, h1_index
        )
    else:
        exh = np.zeros(n)

    result = simulate_trades_swing(
        y_pred=y_pred, close=close, high=high, low=low, atr=atr,
        h4_swing_highs=sh_arr, h4_swing_lows=sl_arr,
        modal=MODAL_PER_TRADE, leverage=LEVERAGE_SIM[0],
        fee_per_side=FEE_PER_SIDE, slippage=SLIPPAGE_PER_SIDE,
        max_hold=MAX_HOLDING_BARS,
        min_rr=SWING_LABEL_MIN_RR, min_tp_atr=SWING_LABEL_MIN_TP,
        max_sl_atr=SWING_LABEL_MAX_SL,
        tp_fallback_atr=TP_SL_FALLBACK_TP, sl_fallback_atr=TP_SL_FALLBACK_SL,
        confidence=confidence, guardian_enabled=False,
        exhaustion_series=exh,
        momentum_series=np.zeros(n, dtype=np.float32),
    )

    trades = result.get("trades", [])
    if not trades:
        return []

    X_static = build_guardian_static_matrix(df_h4, h4_index, h1_index)

    samples = []
    for t_rec in trades:
        bar_in    = t_rec["bar_in"]
        bar_out   = t_rec["bar_out"]
        direction = 2 if t_rec["direction"] == "LONG" else 0
        entry_px  = t_rec["entry"]
        atr_entry = atr[bar_in] if bar_in < len(atr) else 1.0

        if bar_out <= bar_in + 1:
            continue

        mfe_sofar = 0.0
        for j in range(bar_in + 1, min(bar_out, n)):
            if np.isnan(close[j]):
                continue

            bars_held = j - bar_in

            # Update MFE
            if direction == 2:
                mfe_sofar = max(mfe_sofar, (close[j] - entry_px) / entry_px)
            else:
                mfe_sofar = max(mfe_sofar, (entry_px - close[j]) / entry_px)

            # Current PnL
            if direction == 2:
                current_pnl = (close[j] - entry_px) / entry_px
            else:
                current_pnl = (entry_px - close[j]) / entry_px

            # Best future PnL dari j+1 ke bar_out
            best_future = 0.0
            for k in range(j + 1, min(bar_out, n)):
                if np.isnan(close[k]):
                    continue
                fp = ((close[k] - entry_px) / entry_px
                      if direction == 2
                      else (entry_px - close[k]) / entry_px)
                best_future = max(best_future, fp)

            atr_pct    = atr_entry / entry_px if entry_px > 0 else 0.01
            upside_rat = (best_future - current_pnl) / best_future if best_future > 0.001 else 0.0

            # Label assignment (sama dengan Guardian v3)
            if bars_held < 3:
                label = 0
            elif current_pnl < -1.0 * atr_pct:
                label = 2
            elif mfe_sofar > 0.015 and current_pnl < mfe_sofar * 0.25:
                label = 2
            elif current_pnl >= best_future * 0.95:
                label = 2
            elif mfe_sofar > 0.015 and current_pnl < mfe_sofar * 0.55:
                label = 1
            elif current_pnl > 0.008 and upside_rat < 0.03:
                label = 1
            elif best_future > current_pnl * 1.05:
                label = 0
            else:
                continue

            dynamic = _compute_guardian_dynamic(
                bars_held, entry_px, close[j], direction, atr_entry, mfe_sofar,
                exhaustion=float(exh[j]) if j < len(exh) else 0.0,
                momentum=0.0,   # momentum_strength dari LSTM (0 saat training)
            )

            sample = {
                **{f"static_{i}": X_static[j, i] for i in range(len(LGBM_FEATURE_COLS))},
                **{GUARDIAN_DYNAMIC_FEATURES[i]: dynamic[i] for i in range(len(dynamic))},
                "label": label,
            }
            samples.append(sample)

    return samples


def train_guardian(samples_df: pd.DataFrame):
    """Train Guardian v3.5 LGBM multiclass."""
    static_cols  = [c for c in samples_df.columns if c.startswith("static_")]
    dynamic_cols = [c for c in GUARDIAN_DYNAMIC_FEATURES if c in samples_df.columns]
    all_feat     = static_cols + dynamic_cols

    X = samples_df[all_feat].values.astype(np.float64)
    y = samples_df["label"].values.astype(np.int64)

    n_hold    = int((y == 0).sum())
    n_partial = int((y == 1).sum())
    n_full    = int((y == 2).sum())
    logger.info(f"Training data: {len(samples_df)} | HOLD={n_hold} PARTIAL={n_partial} FULL={n_full}")

    if len(samples_df) < 200:
        raise RuntimeError("Tidak cukup samples untuk Guardian training")

    # Scale
    scaler = StandardScaler()
    X_s    = scaler.fit_transform(X)

    # Purged CV
    t     = np.arange(len(X_s))
    folds = build_purged_folds(t, GUARDIAN_N_FOLDS, GUARDIAN_PURGE_GAP_BARS)

    cv_folds = []
    for fold_idx, (tr_idx, val_idx) in enumerate(folds, 1):
        model = lgb.LGBMClassifier(**GUARDIAN_LGBM_PARAMS)
        model.fit(
            X_s[tr_idx], y[tr_idx],
            eval_set=[(X_s[val_idx], y[val_idx])],
            callbacks=[
                lgb.early_stopping(GUARDIAN_EARLY_STOPPING, verbose=False),
                lgb.log_evaluation(period=-1),
            ],
        )
        f1_val = f1_score(
            y[val_idx], model.predict(X_s[val_idx]),
            average="macro", zero_division=0,
        )
        f1_tr = f1_score(
            y[tr_idx], model.predict(X_s[tr_idx]),
            average="macro", zero_division=0,
        )
        cv_folds.append({
            "fold": fold_idx,
            "f1_macro": float(f1_val),
            "f1_macro_train": float(f1_tr),
            "f1_gap_train_minus_val": float(f1_tr - f1_val),
            "n_train": int(len(tr_idx)),
            "n_val": int(len(val_idx)),
        })
        logger.info(f"  Fold {fold_idx}: val_F1={f1_val:.4f} train_F1={f1_tr:.4f}")

    f1_scores = [f["f1_macro"] for f in cv_folds]
    logger.info(f"Guardian CV Mean F1: {np.mean(f1_scores):.4f}")

    with open(MODEL_DIR / "guardian_cv_results.json", "w") as f:
        json.dump({
            "mean_f1": float(np.mean(f1_scores)),
            "std_f1": float(np.std(f1_scores)),
            "folds": cv_folds,
        }, f, indent=2)

    # Final retrain
    final = lgb.LGBMClassifier(**GUARDIAN_LGBM_PARAMS)
    final.fit(X_s, y)

    # Save
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(final,  MODEL_DIR / "guardian_best.pkl")
    joblib.dump(scaler, MODEL_DIR / "guardian_scaler.pkl")

    # Feature name list
    feat_names = static_cols + dynamic_cols
    with open(MODEL_DIR / "guardian_feature_cols.json", "w") as f:
        json.dump(feat_names, f, indent=2)

    logger.info(f"Guardian saved: models/guardian_best.pkl ({len(feat_names)} features)")
    return final, scaler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all",   action="store_true")
    parser.add_argument("--coins", nargs="+", default=[])
    args = parser.parse_args()

    logger.info("Loading models...")
    lgbm_model, feat_cols = load_models()

    coins = TRAINING_COINS if args.all else (args.coins or TRAINING_COINS[:5])

    all_samples = []
    for symbol in coins:
        logger.info(f"Generating labels: {symbol}")
        samples = generate_labels_for_coin(symbol, lgbm_model, feat_cols)
        all_samples.extend(samples)
        logger.info(f"  [{symbol}] {len(samples)} samples")

    if not all_samples:
        logger.error("Tidak ada samples — jalankan 03_engineer dan 04_train_lgbm dulu")
        return

    df_samples = pd.DataFrame(all_samples)
    logger.info(f"Total samples: {len(df_samples)}")

    train_guardian(df_samples)
    logger.info("Guardian v3.5 training selesai.")


if __name__ == "__main__":
    main()
