"""
config.py — Cascade v5: Exhaustion-Aware Residual Momentum Hybrid
Semua parameter terpusat. Edit di sini, jangan duplikasi.
"""

from datetime import datetime, timezone
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT_DIR  = Path(__file__).parent
DATA_DIR  = ROOT_DIR / "data"
RAW_DIR   = DATA_DIR / "raw"
PROC_DIR  = DATA_DIR / "processed"
LABEL_DIR = DATA_DIR / "labeled"
SEQ_DIR   = DATA_DIR / "sequences"
MODEL_DIR = ROOT_DIR / "models"
REPORT_DIR = ROOT_DIR / "reports"

# ─── Koin ─────────────────────────────────────────────────────────────────────
TRAINING_COINS = [
    "SOLUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT",
    "TONUSDT", "ADAUSDT", "TRXUSDT", "1000SHIBUSDT", "AVAXUSDT",
    "LINKUSDT", "DOTUSDT", "SUIUSDT", "POLUSDT", "NEARUSDT",
    "1000PEPEUSDT", "TAOUSDT", "ARBUSDT", "HBARUSDT", "ONDOUSDT",
]
ALL_COINS  = TRAINING_COINS
SYMBOL_MAP = {coin: i for i, coin in enumerate(ALL_COINS)}

# ─── Periode ──────────────────────────────────────────────────────────────────
TRAIN_START      = datetime(2020, 1, 1, tzinfo=timezone.utc)
TRAIN_END        = datetime(2026, 4, 1, tzinfo=timezone.utc)
TRAIN_CUTOFF_DATE = datetime(2025, 5, 1, tzinfo=timezone.utc)  # holdout = setelah ini

# ─── Binance ──────────────────────────────────────────────────────────────────
BINANCE_BASE_URL       = "https://fapi.binance.com"
SLEEP_BETWEEN_REQUESTS = 0.12
SLEEP_ON_RATE_LIMIT    = 60.0
MAX_RETRIES            = 3
RETRY_BACKOFF_BASE     = 2.0
KLINE_LIMIT            = 1000
FUNDING_LIMIT          = 1000
KLINE_INTERVALS        = ["1h", "4h"]

# ─── Simulasi Trading ─────────────────────────────────────────────────────────
MODAL_PER_TRADE  = 5.0          # USD per trade — MUTLAK, jangan diubah
LEVERAGE_SIM     = [5.0]
FEE_PER_SIDE     = 0.0004
SLIPPAGE_PER_SIDE = 0.0005
MAX_HOLDING_BARS = 24           # bar H1 = 24 jam

# ─── TP/SL Hybrid ─────────────────────────────────────────────────────────────
SWING_LABEL_MIN_RR   = 0.5
SWING_LABEL_MIN_TP   = 1.2
SWING_LABEL_MAX_SL   = 4.0
SWING_LABEL_MAX_HOLD = 24
TP_SL_FALLBACK_TP    = 2.0
TP_SL_FALLBACK_SL    = 1.5
SWING_BUMPER_ATR     = 0.5

# ─── Feature Split ────────────────────────────────────────────────────────────
# LGBM Tabular Expert — 60 fitur struktur + regime + value
# Berdasarkan feature_separation analysis: lgbm_importance tinggi, tempvar boleh rendah
LGBM_FEATURE_COLS = [
    # Liquidation levels (top importance)
    "dist_liq_50x_long", "dist_liq_50x_short",
    "dist_liq_20x_long", "dist_liq_20x_short",
    # Macro
    "fear_greed", "btc_dominance",
    # H4 structure
    "atr_percent_h4", "cvd_slope_h4", "ofi_h4_delta",
    "trend_strength", "trend_accel_4h", "cvd_momentum_adv",
    # Price levels
    "dist_from_8h_high", "dist_swing_high", "dist_swing_low",
    "PDH", "PDL", "PWH", "PWL",
    # Volume profile
    "POC", "VAH", "VAL",
    # Liquidity
    "Buy_Liq", "Sell_Liq",
    # Open interest & funding
    "open_interest", "funding_rate",
    # ATR
    "atr_14_h1", "atr_14_h4", "atr_zscore_20d", "atr_percentile_h1",
    # EMA H1 (slow)
    "ema_200_h1",
    # Momentum tabular
    "log_ret_20", "cvd",
    # Time (tabular context)
    "dow_sin", "dow_cos",
    # Structure tabular
    "bars_since_BOS", "absorption_at_swing",
    # Other
    "vwdp_smooth", "relative_strength_z", "whale_retail_divergence",
    "funding_price_div", "relative_strength_momentum",
    "vol_spike_zscore", "spread_to_volume", "vol_efficiency",
    "buy_volume", "volume",
    # OHLC (tabular snapshot)
    "open", "high", "low", "close",
    # Regime soft features
    "range_expansion_h4", "cvd_div_h4",
]

# LSTM Sequence Expert — 20 fitur trajectory (temporal variance tinggi)
# Berdasarkan feature_separation: tempvar > 0.30, berubah setiap H1 bar
LSTM_SEQUENCE_COLS = [
    "time_to_funding_norm",  # 1.016 — berubah tiap jam
    "swing_momentum",        # 0.962
    "ofi_z_score",           # 0.952
    "absorption_z",          # 0.956
    "effort_vs_result",      # 0.951
    "rsi_slope_h4",          # 0.918
    "log_ret_1",             # 0.917
    "rsi_6",                 # 0.909
    "stochrsi_k",            # 0.938
    "stochrsi_d",            # 0.929
    "vol_ratio_20",          # 0.902
    "vol_regime",            # 0.881
    "ema_7_h1",              # 0.887
    "dist_from_8h_high",     # 0.885 (shared)
    "log_ret_5",             # 0.832
    "ofi_acceleration",      # 0.353
    "volume_delta",          # 0.319
    "h4_trend",              # 0.606
    "dist_swing_low",        # 0.672
    "log_ret_20",            # 0.622 (shared)
]

# Features buang — zero importance dan zero temporal variance
DROP_FEATURES = ["FVG_up", "FVG_down", "dynamic_position_pressure",
                 "hidden_divergence", "sell_volume"]

# ─── Label Settings ───────────────────────────────────────────────────────────
LABEL_MAP     = {"SHORT": 0, "FLAT": 1, "LONG": 2}
LABEL_MAP_INV = {v: k for k, v in LABEL_MAP.items()}
NUM_CLASSES   = 3

# Momentum label (05a) — rule-based, tidak forward scan
MOMENTUM_WINDOW     = 5    # bar H1 untuk hitung acceleration
MOMENTUM_ATR_THRESH = 1.5  # price move >= 1.5x ATR = high momentum
MOMENTUM_VOL_THRESH = 1.3  # volume ratio >= 1.3x = volume confirm
N_FORWARD_BARS      = 12   # horizon label momentum (N bar ke depan)
MIN_MOVE_ATR        = 0.4  # minimum move untuk label valid

# Exhaustion label (rule-based, tidak forward scan)
EXHAUSTION_VOL_DROP   = 0.7   # volume ratio < 0.7 = volume melemah
EXHAUSTION_PRICE_ATR  = 2.0   # harga sudah bergerak > 2x ATR dari swing
EXHAUSTION_WICK_RATIO = 0.5   # wick > 50% candle body = exhaustion signal

# ─── LGBM Tabular Expert ──────────────────────────────────────────────────────
LGBM_PARAMS = {
    "objective":         "multiclass",
    "num_class":         3,
    "n_estimators":      1000,
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
LGBM_EARLY_STOPPING  = 50
LGBM_THRESHOLD_LONG  = 0.65
LGBM_THRESHOLD_SHORT = 0.65

# ─── LSTM Momentum + Exhaustion Expert ────────────────────────────────────────
LSTM_SEQ_LEN     = 32    # 32 bar H1 = 32 jam history
LSTM_HIDDEN      = 128
LSTM_LAYERS      = 2
LSTM_DROPOUT     = 0.3
LSTM_EPOCHS      = 100
LSTM_PATIENCE    = 15
LSTM_BATCH_SIZE  = 512
LSTM_LR          = 0.001
LSTM_WEIGHT_DECAY = 1e-4

# Output heads LSTM
LSTM_OUTPUT_MOMENTUM_STRENGTH  = True   # head 1: seberapa kuat momentum
LSTM_OUTPUT_EXHAUSTION_SCORE   = True   # head 2: seberapa dekat exhaustion
LSTM_OUTPUT_RESIDUAL_CORRECTION = True  # head 3: koreksi error LGBM (OOF residual)

# ─── Dynamic Fusion Layer ─────────────────────────────────────────────────────
# Weights akan di-learn via meta-learner (logistic regression pada OOF predictions)
# Nilai awal untuk eksperimen pertama:
FUSION_LGBM_WEIGHT        = 0.55
FUSION_RESIDUAL_WEIGHT    = 0.35
FUSION_EXHAUSTION_PENALTY = 0.25
FUSION_MOMENTUM_BOOST_THR = 0.75   # momentum_strength > ini → beri boost
FUSION_MOMENTUM_BOOST_VAL = 0.05   # besar boost

# ─── Smart Entry Gate ─────────────────────────────────────────────────────────
ENTRY_FINAL_PROB_THR        = 0.68  # threshold normal
ENTRY_MOMENTUM_STR_THR      = 0.55  # momentum_strength minimum
ENTRY_EXHAUSTION_MAX         = 0.60  # reject jika exhaustion > ini
ENTRY_HIGH_CONV_MOMENTUM_THR = 0.85  # high-conviction path: momentum threshold
ENTRY_HIGH_CONV_PROB_THR     = 0.60  # high-conviction path: prob threshold (lebih rendah)

# ─── Training & CV ────────────────────────────────────────────────────────────
N_FOLDS        = 8
PURGE_GAP_BARS = 24   # lebih konservatif dari v4 (24 bars = 1 hari)

# ─── Guardian v3.5 ────────────────────────────────────────────────────────────
GUARDIAN_ENABLED             = True
GUARDIAN_EXIT_THRESHOLD      = 0.60
GUARDIAN_MIN_HOLD_BARS       = 3
GUARDIAN_ACTIVATION_ATR      = 1.0
GUARDIAN_PARTIAL_EXIT_RATIO  = 0.50
GUARDIAN_EARLY_STOPPING      = 50
GUARDIAN_N_FOLDS             = 8
GUARDIAN_PURGE_GAP_BARS      = 24

# Guardian dynamic features (tetap sama dengan v3)
GUARDIAN_DYNAMIC_FEATURES = [
    "bars_held_norm", "current_pnl_pct", "current_pnl_atr",
    "max_favorable_pnl_pct", "drawdown_from_peak_pct",
    "direction", "entry_price_ratio",
]
# Guardian v3.5: tambah exhaustion & momentum dari Expert
GUARDIAN_EXTRA_FEATURES = [
    "exhaustion_score",     # dari Momentum Expert
    "momentum_strength",    # dari Momentum Expert
]

# ─── Trailing Stop (non-ML) ───────────────────────────────────────────────────
TRAILING_STOP_ENABLED   = False
TRAILING_STOP_ATR       = 2.0
TRAILING_STOP_MIN_BARS  = 2
