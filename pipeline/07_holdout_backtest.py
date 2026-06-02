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
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from config import (
    TRAINING_COINS, MODEL_DIR,
    LGBM_FEATURE_COLS, LSTM_SEQUENCE_COLS,
    MODAL_PER_TRADE, LEVERAGE_SIM, FEE_PER_SIDE, SLIPPAGE_PER_SIDE,
    MAX_HOLDING_BARS, SWING_LABEL_MIN_RR, SWING_LABEL_MIN_TP, SWING_LABEL_MAX_SL,
    TP_SL_FALLBACK_TP, TP_SL_FALLBACK_SL,
    GUARDIAN_ENABLED, GUARDIAN_EXIT_THRESHOLD,
)
from core.models import load_attention_lstm
from core.fusion import batch_predict
from core.evaluator import full_trading_report
from core.utils import setup_logger
from pipeline.shared import (
    load_coin_simulation_data,
    build_lgbm_feature_matrix,
    lgbm_proba_h4_to_h1,
    infer_lstm_on_h4,
    upsample_h4_to_h1,
    build_guardian_static_matrix,
    h4_swing_levels_on_h1,
)

logger = setup_logger("07_holdout_backtest")

HOLDOUT_START = datetime(2025, 5,  1, tzinfo=timezone.utc)
HOLDOUT_END   = datetime(2026, 4,  1, tzinfo=timezone.utc)
HOLDOUT_DIR   = ROOT / "data" / "holdout"
HOLDOUT_LABEL = HOLDOUT_DIR / "labeled"
HOLDOUT_PROC  = HOLDOUT_DIR / "processed"


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

    frames = load_coin_simulation_data(symbol, HOLDOUT_LABEL, HOLDOUT_PROC)
    if frames is None:
        logger.warning(f"[{symbol}] Holdout data tidak ada — jalankan 01-03 dengan --holdout")
        return None

    try:
        df_h4  = frames["h4_lgbm"]
        df_h4l = frames["h4_lstm"]
        h1_proc = frames["h1_proc"]
        h1_conf = frames["h1_conf"]
        h1_index = frames["h1_index"]
        h4_index = frames["h4_index"]

        mask_h1 = (h1_index >= HOLDOUT_START) & (h1_index < HOLDOUT_END)
        h1_index = h1_index[mask_h1]
        h1_proc  = h1_proc.loc[h1_index]
        h1_conf  = h1_conf.loc[h1_index]

        df_h4  = df_h4[(df_h4.index >= HOLDOUT_START) & (df_h4.index < HOLDOUT_END)]
        df_h4l = df_h4l.reindex(df_h4.index).ffill()
        h4_index = df_h4.index

        n_h1 = len(h1_index)
        if n_h1 < 50:
            logger.warning(f"[{symbol}] Holdout H1 terlalu sedikit — skip")
            return None

        proba_h4 = lgbm_model.predict(build_lgbm_feature_matrix(df_h4))
        lgbm_proba_3 = lgbm_proba_h4_to_h1(proba_h4, h4_index, h1_index)

        momentum_strength   = np.full(n_h1, 0.5, dtype=np.float32)
        exhaustion_score    = np.zeros(n_h1, dtype=np.float32)
        residual_correction = np.zeros(n_h1, dtype=np.float32)

        if lstm_model is not None:
            seq_scaler = None
            scaler_path = MODEL_DIR / "lstm_seq_scaler.pkl"
            if scaler_path.exists():
                seq_scaler = joblib.load(scaler_path)
            mom_h4, exh_h4, res_h4 = infer_lstm_on_h4(lstm_model, df_h4l, seq_scaler)
            momentum_strength = upsample_h4_to_h1(mom_h4, h4_index, h1_index).astype(np.float32)
            exhaustion_score  = upsample_h4_to_h1(exh_h4, h4_index, h1_index).astype(np.float32)
            residual_correction = upsample_h4_to_h1(res_h4, h4_index, h1_index).astype(np.float32)

        if "exhaustion_score" in df_h4l.columns:
            exh_feat = upsample_h4_to_h1(
                df_h4l["exhaustion_score"].values, h4_index, h1_index
            )
            exhaustion_score = np.maximum(exhaustion_score, exh_feat.astype(np.float32))

        h1_rsi   = h1_conf["h1_rsi"].values if "h1_rsi" in h1_conf.columns else None
        h1_ret   = h1_conf["h1_log_ret_1"].values if "h1_log_ret_1" in h1_conf.columns else None
        h1_accel = h1_conf["h1_accel_sign"].values if "h1_accel_sign" in h1_conf.columns else None

        y_pred, confidence = batch_predict(
            lgbm_proba          = lgbm_proba_3,
            momentum_strength   = momentum_strength,
            exhaustion_score    = exhaustion_score,
            residual_correction = residual_correction,
            h1_rsi              = h1_rsi,
            h1_log_ret          = h1_ret,
            h1_accel_sign       = h1_accel,
        )

        X_guardian = None
        if guardian_model is not None:
            X_guardian = build_guardian_static_matrix(df_h4, h4_index, h1_index)

        close = h1_proc["close"].values
        high  = h1_proc["high"].values if "high" in h1_proc.columns else close
        low   = h1_proc["low"].values  if "low" in h1_proc.columns else close
        atr   = (
            h1_proc["atr_14_h1"].values
            if "atr_14_h1" in h1_proc.columns
            else np.ones(n_h1)
        )
        sh_arr, sl_arr = h4_swing_levels_on_h1(df_h4l, h4_index, h1_index)

        report = full_trading_report(
            y_pred         = y_pred,
            y_actual       = np.ones(n_h1, dtype=np.int64),
            atr            = atr,
            close          = close,
            high           = high,
            low            = low,
            h4_swing_highs = sh_arr,
            h4_swing_lows  = sl_arr,
            index          = h1_index,
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
            exhaustion_series = exhaustion_score,
            momentum_series   = momentum_strength,
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
