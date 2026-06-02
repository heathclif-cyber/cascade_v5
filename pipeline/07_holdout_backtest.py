"""
pipeline/07_holdout_backtest.py — Holdout Backtest (Cascade v5)

Genuine Out-of-Sample evaluation: Mei 2025 – Apr 2026.
Menggunakan full Cascade v5 pipeline:
  LGBM (5-class) → Attention LSTM → Dynamic Fusion → Smart Entry Gate → Guardian v3.5

Output:
  models/runs/{run_id}/holdout_results.json
  models/runs/{run_id}/holdout_trade_history.csv
  models/runs/{run_id}/holdout_report.md

Jalankan:
  python pipeline/07_holdout_backtest.py --all
  python pipeline/07_holdout_backtest.py --coins SOLUSDT ETHUSDT --run-id test_v5
"""

import argparse, json, sys, traceback, warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from config import (
    TRAINING_COINS, MODEL_DIR,
    LGBM_FEATURE_COLS, LSTM_SEQUENCE_COLS,
    LGBM_NUM_CLASSES,
    MODAL_PER_TRADE, LEVERAGE_SIM, FEE_PER_SIDE, SLIPPAGE_PER_SIDE,
    MAX_HOLDING_BARS, SWING_LABEL_MIN_RR, SWING_LABEL_MIN_TP, SWING_LABEL_MAX_SL,
    TP_SL_FALLBACK_TP, TP_SL_FALLBACK_SL,
    GUARDIAN_ENABLED, GUARDIAN_EXIT_THRESHOLD,
)
from core.models import load_attention_lstm, infer_momentum_expert
from core.fusion import batch_predict
from core.evaluator import full_trading_report
from core.utils import setup_logger, ensure_utc_index

logger = setup_logger("07_holdout_backtest")

HOLDOUT_START = datetime(2025, 5,  1, tzinfo=timezone.utc)
HOLDOUT_END   = datetime(2026, 4,  1, tzinfo=timezone.utc)
HOLDOUT_DIR   = ROOT / "data" / "holdout"


def load_models():
    lgbm = joblib.load(MODEL_DIR / "lgbm_tabular.pkl")

    lstm_path = MODEL_DIR / "lstm_momentum_expert.pt"
    lstm      = None
    if lstm_path.exists():
        n_feat = len(LSTM_SEQUENCE_COLS)
        lstm   = load_attention_lstm(lstm_path, n_feat, device="cpu")
        logger.info("LSTM momentum expert loaded")
    else:
        logger.warning("LSTM tidak ditemukan — fusion tanpa momentum/exhaustion")

    guardian = guardian_scaler = None
    g_path   = MODEL_DIR / "guardian_best.pkl"
    gs_path  = MODEL_DIR / "guardian_scaler.pkl"
    if GUARDIAN_ENABLED and g_path.exists() and gs_path.exists():
        guardian        = joblib.load(g_path)
        guardian_scaler = joblib.load(gs_path)
        logger.info("Guardian v3.5 loaded")

    return lgbm, lstm, guardian, guardian_scaler


def backtest_coin(
    symbol:          str,
    lgbm_model,
    lstm_model,
    guardian_model,
    guardian_scaler,
    run_dir:         Path,
) -> dict | None:

    h4_path = HOLDOUT_DIR / "labeled" / f"{symbol}_h4_lgbm.parquet"
    h1_path = HOLDOUT_DIR / "labeled" / f"{symbol}_h1_lstm.parquet"

    if not h4_path.exists() or not h1_path.exists():
        logger.warning(f"[{symbol}] Holdout data tidak ada — skip")
        return None

    try:
        df_h4 = pd.read_parquet(h4_path)
        df_h1 = pd.read_parquet(h1_path)
        df_h4 = ensure_utc_index(df_h4)
        df_h1 = ensure_utc_index(df_h1)

        n_h1 = len(df_h1)

        # ── LGBM inference ────────────────────────────────────────────────────
        X_h4 = np.zeros((len(df_h4), len(LGBM_FEATURE_COLS)), dtype=np.float64)
        for idx, col in enumerate(LGBM_FEATURE_COLS):
            if col in df_h4.columns:
                X_h4[:, idx] = df_h4[col].ffill().fillna(0).values

        lgbm_proba_h4 = lgbm_model.predict_proba(X_h4)  # (N_h4, 5)

        # Collapse 5-class → 3 proba: short, flat, long (untuk fusion)
        p_long_h4  = lgbm_proba_h4[:, 3] + lgbm_proba_h4[:, 4]
        p_short_h4 = lgbm_proba_h4[:, 0] + lgbm_proba_h4[:, 1]
        p_flat_h4  = lgbm_proba_h4[:, 2]

        # Upsample H4 → H1 (ffill)
        h4_index = df_h4.index
        h1_index = df_h1.index

        def upsample_h4_to_h1(arr_h4, idx_h4, idx_h1):
            s = pd.Series(arr_h4, index=idx_h4)
            return s.reindex(idx_h1).ffill().fillna(0).values

        lgbm_long_h1  = upsample_h4_to_h1(p_long_h4,  h4_index, h1_index)
        lgbm_short_h1 = upsample_h4_to_h1(p_short_h4, h4_index, h1_index)

        # ── LSTM inference ────────────────────────────────────────────────────
        momentum_strength   = np.full(n_h1, 0.5, dtype=np.float32)
        exhaustion_score    = np.zeros(n_h1, dtype=np.float32)
        residual_correction = np.zeros(n_h1, dtype=np.float32)

        if lstm_model is not None:
            X_lstm = np.zeros((n_h1, len(LSTM_SEQUENCE_COLS)), dtype=np.float32)
            for idx, col in enumerate(LSTM_SEQUENCE_COLS):
                if col in df_h1.columns:
                    X_lstm[:, idx] = df_h1[col].ffill().fillna(0).values

            # Scale
            from sklearn.preprocessing import StandardScaler
            scaler_path = MODEL_DIR / "lstm_seq_scaler.pkl"
            if scaler_path.exists():
                scaler  = joblib.load(scaler_path)
                X_lstm  = scaler.transform(X_lstm)

            # Build sequences
            seq_len = 32
            X_seq   = np.zeros((n_h1, seq_len, len(LSTM_SEQUENCE_COLS)), dtype=np.float32)
            for i in range(seq_len - 1, n_h1):
                X_seq[i] = X_lstm[i - seq_len + 1: i + 1]

            mom, exh, res = infer_momentum_expert(lstm_model, X_seq)
            momentum_strength[seq_len-1:]   = mom[seq_len-1:]
            exhaustion_score[seq_len-1:]    = exh[seq_len-1:]
            residual_correction[seq_len-1:] = res[seq_len-1:]

        # Exhaustion dari feature engineering jika tersedia
        if "exhaustion_score" in df_h1.columns:
            exh_feat = df_h1["exhaustion_score"].ffill().fillna(0).values
            exhaustion_score = np.maximum(exhaustion_score, exh_feat.astype(np.float32))

        # ── Dynamic Fusion ────────────────────────────────────────────────────
        lgbm_proba_3 = np.stack([lgbm_short_h1, 1 - lgbm_long_h1 - lgbm_short_h1, lgbm_long_h1], axis=1)
        lgbm_proba_3 = np.clip(lgbm_proba_3, 0, 1)

        y_pred, confidence = batch_predict(
            lgbm_proba          = lgbm_proba_3,
            momentum_strength   = momentum_strength,
            exhaustion_score    = exhaustion_score,
            residual_correction = residual_correction,
        )

        # ── Guardian pre-compute ──────────────────────────────────────────────
        X_guardian = None
        if guardian_model is not None:
            with open(MODEL_DIR / "guardian_feature_cols.json") as f:
                g_feat_cols = json.load(f)
            g_static_cols = [c for c in g_feat_cols if not c.startswith("static_") and c in df_h1.columns]
            X_guardian = df_h1[g_static_cols].ffill().fillna(0).values if g_static_cols else None

        # ── Simulate trades ───────────────────────────────────────────────────
        close  = df_h1["close"].values  if "close"  in df_h1.columns else np.ones(n_h1)
        high   = df_h1["high"].values   if "high"   in df_h1.columns else close
        low    = df_h1["low"].values    if "low"    in df_h1.columns else close
        atr    = df_h1["atr_14_h1"].values if "atr_14_h1" in df_h1.columns else np.ones(n_h1)
        sh_arr = df_h1["h4_swing_high"].values if "h4_swing_high" in df_h1.columns else None
        sl_arr = df_h1["h4_swing_low"].values  if "h4_swing_low"  in df_h1.columns else None

        report = full_trading_report(
            y_pred         = y_pred,
            y_actual       = np.ones(n_h1, dtype=np.int64),
            atr            = atr,
            close          = close,
            high           = high,
            low            = low,
            h4_swing_highs = sh_arr,
            h4_swing_lows  = sl_arr,
            index          = df_h1.index,
            modal          = MODAL_PER_TRADE,
            leverages      = LEVERAGE_SIM,
            fee_per_side   = FEE_PER_SIDE,
            slippage       = SLIPPAGE_PER_SIDE,
            min_rr         = SWING_LABEL_MIN_RR,
            min_tp_atr     = SWING_LABEL_MIN_TP,
            max_sl_atr     = SWING_LABEL_MAX_SL,
            max_hold       = MAX_HOLDING_BARS,
            symbol         = symbol,
            confidence     = confidence,
            guardian_model  = guardian_model,
            guardian_scaler = guardian_scaler,
            X_guardian      = X_guardian,
            guardian_enabled = guardian_model is not None,
            guardian_exit_threshold = GUARDIAN_EXIT_THRESHOLD,
        )
        return report

    except Exception as e:
        logger.error(f"[{symbol}] Backtest error: {e}")
        logger.error(traceback.format_exc())
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all",    action="store_true")
    parser.add_argument("--coins",  nargs="+", default=[])
    parser.add_argument("--run-id", default=f"holdout_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    args = parser.parse_args()

    coins   = TRAINING_COINS if args.all else (args.coins or TRAINING_COINS[:3])
    run_dir = MODEL_DIR / "runs" / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Run ID: {args.run_id} | Coins: {len(coins)}")
    logger.info("Loading models...")
    lgbm, lstm, guardian, guardian_scaler = load_models()

    all_reports = {}
    for symbol in coins:
        logger.info(f"Backtesting {symbol}...")
        report = backtest_coin(symbol, lgbm, lstm, guardian, guardian_scaler, run_dir)
        if report:
            all_reports[symbol] = report

    if not all_reports:
        logger.error("Tidak ada hasil — cek holdout data di data/holdout/labeled/")
        return

    # ── Aggregate metrics ─────────────────────────────────────────────────────
    wrs    = [r.get("winrate", 0)       for r in all_reports.values()]
    pfs    = [r.get("profit_factor", 0) for r in all_reports.values()]
    sharpe = [r.get("sharpe_ratio", 0)  for r in all_reports.values()]
    tpms   = [r.get("trade_per_month",0) for r in all_reports.values()]
    pnls   = [r.get("pnl_lev5x", 0)    for r in all_reports.values() if "pnl_lev5x" in r]

    lev_key = f"pnl_lev{int(LEVERAGE_SIM[0])}x"
    pnls    = [r.get(lev_key, 0) for r in all_reports.values()]

    logger.info("=" * 60)
    logger.info(f"HOLDOUT RESULTS — {len(all_reports)} coins")
    logger.info(f"  Mean WR     : {np.mean(wrs):.2%}")
    logger.info(f"  Mean PF     : {np.mean(pfs):.2f}")
    logger.info(f"  Mean Sharpe : {np.mean(sharpe):.2f}")
    logger.info(f"  Mean TPM    : {np.mean(tpms):.1f}")
    logger.info(f"  Total PnL   : ${sum(pnls):.2f}")
    logger.info("=" * 60)

    # Save JSON
    summary = {
        "run_id":       args.run_id,
        "coins":        list(all_reports.keys()),
        "mean_wr":      float(np.mean(wrs)),
        "mean_pf":      float(np.mean(pfs)),
        "mean_sharpe":  float(np.mean(sharpe)),
        "total_pnl":    float(sum(pnls)),
        "modal":        MODAL_PER_TRADE,
        "per_coin":     {s: {k: v for k, v in r.items() if k != "trades"}
                         for s, r in all_reports.items()},
    }
    with open(run_dir / "holdout_results.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Save trade history
    all_trades = []
    for symbol, report in all_reports.items():
        for t in report.get("trades", []):
            t["symbol"] = symbol
            all_trades.append(t)

    if all_trades:
        pd.DataFrame(all_trades).to_csv(run_dir / "holdout_trade_history.csv", index=False)

    logger.info(f"Results saved to: {run_dir}")


if __name__ == "__main__":
    main()
