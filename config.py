"""
config.py — Cascade v5: Exhaustion-Aware Residual Momentum Hybrid
H4 Primary + Attention LSTM + Dynamic Fusion + Guardian v3.5
Semua parameter terpusat. Edit di sini, jangan duplikasi.
"""

from datetime import datetime, timezone
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT_DIR   = Path(__file__).parent
DATA_DIR   = ROOT_DIR / "data"
RAW_DIR    = DATA_DIR / "raw"
PROC_DIR   = DATA_DIR / "processed"
LABEL_DIR  = DATA_DIR / "labeled"
SEQ_DIR    = DATA_DIR / "sequences"
MODEL_DIR  = ROOT_DIR / "models"
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
TRAIN_START       = datetime(2020, 1, 1, tzinfo=timezone.utc)
TRAIN_END         = datetime(2026, 4, 1, tzinfo=timezone.utc)
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
MODAL_PER_TRADE   = 5.0           # USD per trade — MUTLAK, jangan diubah
LEVERAGE_SIM      = [5.0]
FEE_PER_SIDE      = 0.0004
SLIPPAGE_PER_SIDE = 0.0005
MAX_HOLDING_BARS  = 24            # bar H1 = 24 jam

# ─── TP/SL Hybrid (H4 Swing + ATR fallback) ──────────────────────────────────
SWING_LABEL_MIN_RR   = 0.5
SWING_LABEL_MIN_TP   = 1.2
SWING_LABEL_MAX_SL   = 4.0
SWING_LABEL_MAX_HOLD = 24
TP_SL_FALLBACK_TP    = 2.0
TP_SL_FALLBACK_SL    = 1.5
SWING_BUMPER_ATR     = 0.5

# ─── Swing Calculation ────────────────────────────────────────────────────────
# Hanya backward-looking — tidak ada look-ahead leakage
SWING_LEFT_BARS  = 5   # berapa bar ke kiri untuk deteksi swing pivot
# TIDAK ADA right bars — kita pakai rolling max/min ke belakang saja

# ─── Label Classes ────────────────────────────────────────────────────────────
# LGBM 5-class: -2=strong short, -1=weak short, 0=flat, 1=weak long, 2=strong long
LGBM_NUM_CLASSES   = 5
LGBM_LABEL_MAP     = {-2: 0, -1: 1, 0: 2, 1: 3, 2: 4}   # map ke index 0-4
LGBM_LABEL_MAP_INV = {v: k for k, v in LGBM_LABEL_MAP.items()}

# ATR-return thresholds untuk 5-class label (pada H4, horizon 18 bar)
LGBM_LABEL_HORIZON  = 18          # bar H4 ke depan untuk hitung return (~3 hari)
LGBM_STRONG_THR     = 2.5         # |return/ATR| > 2.5 = strong
LGBM_WEAK_THR       = 1.0         # |return/ATR| > 1.0 = weak

# Label lama (3-class) — dipakai di Guardian dan TP/SL simulation
LABEL_MAP     = {"SHORT": 0, "FLAT": 1, "LONG": 2}
LABEL_MAP_INV = {v: k for k, v in LABEL_MAP.items()}
NUM_CLASSES   = 3

# ─── LGBM Tabular Expert ──────────────────────────────────────────────────────
# ~60 fitur — struktur pasar, regime, macro, levels
# Berdasarkan feature_separation analysis: lgbm_importance tinggi, tempvar boleh rendah
LGBM_FEATURE_COLS = [
    # Liquidation levels (top importance dari audit v4)
    "dist_liq_50x_long", "dist_liq_50x_short",
    "dist_liq_20x_long", "dist_liq_20x_short",
    # Macro
    "fear_greed", "btc_dominance",
    # H4 structure & regime
    "atr_percent_h4", "cvd_slope_h4", "ofi_h4_delta",
    "trend_strength", "trend_accel_4h", "cvd_momentum_adv",
    "range_expansion_h4", "cvd_div_h4",
    "H4_structure_break_strength",       # BARU v5
    # Price levels
    "dist_from_8h_high", "dist_swing_high", "dist_swing_low",
    "PDH", "PDL", "PWH", "PWL",
    # Volume profile
    "POC", "VAH", "VAL",
    # Liquidity
    "Buy_Liq", "Sell_Liq",
    # OI & funding
    "open_interest", "funding_rate", "funding_price_div",
    # ATR
    "atr_14_h1", "atr_14_h4", "atr_zscore_20d", "atr_percentile_h1",
    # EMA slow (snapshot — cocok LGBM)
    "ema_200_h1", "ema_50_h4", "ema_200_h4",
    # Returns tabular
    "log_ret_20",
    # CVD tabular
    "cvd",
    # Time
    "dow_sin", "dow_cos",
    # Structure tabular
    "bars_since_BOS", "absorption_at_swing",
    # Other tabular
    "vwdp_smooth", "relative_strength_z", "whale_retail_divergence",
    "relative_strength_momentum", "vol_spike_zscore", "spread_to_volume",
    "vol_efficiency", "buy_volume", "volume",
    # OHLC (snapshot)
    "open", "high", "low", "close",
]

# ─── LSTM Sequence Expert (H4 PRIMARY) ───────────────────────────────────────
# 10 trajectory features — dihitung dari H4 bars (bukan H1)
# H4 lebih smooth, swing/exhaustion lebih meaningful, kurang noise
# Setiap bar = 4 jam → sequence 24 bar = 4 hari context
LSTM_SEQUENCE_COLS = [
    "distance_from_recent_swing_high_atr",  # jarak dari H4 swing high (ATR units)
    "distance_from_recent_swing_low_atr",   # jarak dari H4 swing low (ATR units)
    "run_length_up_bars",                   # H4 bars naik berturut-turut
    "acceleration_sign_change",             # apakah H4 momentum baru berbalik
    "log_return_acceleration_5",            # percepatan return 5 H4 bars (20 jam)
    "momentum_delta_5",                     # delta momentum 5 H4 bars
    "funding_extreme_zscore",               # funding rate extreme z-score
    "price_funding_divergence",             # divergence harga H4 vs funding
    "volume_acceleration",                  # percepatan volume H4
    "candle_body_size_acceleration",        # percepatan ukuran body H4 candle
]

# ─── H1 Selective Confirmation (Smart Entry Gate only) ───────────────────────
# H1 TIDAK dipakai sebagai input LSTM — hanya sebagai filter timing di gate
# Rule-based, tidak membutuhkan model tambahan
H1_CONFIRMATION_ENABLED = True
H1_CONFIRMATION_COLS = [
    "h1_rsi",            # RSI H1 untuk konfirmasi arah momentum di level lebih kecil
    "h1_log_ret_1",      # return H1 bar terbaru (apakah arah sesuai sinyal H4?)
    "h1_accel_sign",     # acceleration sign H1 (momentum H1 membangun atau melemah?)
]
# Threshold untuk H1 confirmation (jika sinyal H4 LONG):
H1_CONF_RSI_LONG_MIN  = 45.0   # RSI H1 minimal 45 untuk LONG (tidak oversold ekstrem)
H1_CONF_RSI_SHORT_MAX = 55.0   # RSI H1 maksimal 55 untuk SHORT
H1_CONF_RET_DIRECTION = True   # H1 return harus searah sinyal H4

# Features yang di-DROP dari v4 (zero importance + zero temporal variance)
DROP_FEATURES = [
    "FVG_up", "FVG_down", "dynamic_position_pressure",
    "hidden_divergence", "sell_volume",
]

# ─── Exhaustion Detection ─────────────────────────────────────────────────────
EXHAUSTION_SWING_ATR_THR  = 3.5   # distance > 3.5 ATR dari swing = over-extended
EXHAUSTION_ACCEL_SIGN_CHG = True  # harus ada acceleration sign change bersamaan
EXHAUSTION_VOL_DROP       = 0.7   # volume ratio < 0.7 = konfirmasi volume melemah
EXHAUSTION_WICK_RATIO     = 0.5   # wick > 50% = rejection candle

# ─── Training & CV ────────────────────────────────────────────────────────────
N_FOLDS          = 8
PURGE_GAP_BARS   = 24   # 24 H1 bars = 1 hari (lebih konservatif dari v4)
PURGE_GAP_H4     = 6    # 6 H4 bars = 1 hari (untuk model H4)

# ─── LGBM Params ──────────────────────────────────────────────────────────────
LGBM_PARAMS = {
    "objective":         "multiclass",
    "num_class":         LGBM_NUM_CLASSES,
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
LGBM_EARLY_STOPPING = 50

# Entry thresholds — mapped dari 5-class ke long/short probability
# class 3 (weak long) + class 4 (strong long) = long signal
# class 0 (strong short) + class 1 (weak short) = short signal
LGBM_LONG_CLASSES  = [3, 4]   # index 3=weak long, 4=strong long
LGBM_SHORT_CLASSES = [0, 1]   # index 0=strong short, 1=weak short
LGBM_THRESHOLD_LONG  = 0.50   # threshold combined p(weak long) + p(strong long)
LGBM_THRESHOLD_SHORT = 0.50

# ─── LSTM / AttentionLSTM ─────────────────────────────────────────────────────
LSTM_SEQ_LEN      = 24    # 24 bar H4 = 4 hari sequence (H4 PRIMARY)
LSTM_HIDDEN       = 128
LSTM_LAYERS       = 2
LSTM_DROPOUT      = 0.3
LSTM_EPOCHS       = 100
LSTM_PATIENCE     = 15
LSTM_BATCH_SIZE   = 512
LSTM_LR           = 0.001
LSTM_WEIGHT_DECAY = 1e-4

# ─── Dynamic Fusion Layer ─────────────────────────────────────────────────────
FUSION_LGBM_WEIGHT          = 0.55
FUSION_RESIDUAL_WEIGHT      = 0.35
FUSION_EXHAUSTION_PENALTY   = 0.25
FUSION_MOMENTUM_BOOST_THR   = 0.75
FUSION_MOMENTUM_BOOST_VAL   = 0.05

# ─── Smart Entry Gate ─────────────────────────────────────────────────────────
ENTRY_FINAL_PROB_THR         = 0.68
ENTRY_MOMENTUM_STR_THR       = 0.55
ENTRY_EXHAUSTION_MAX         = 0.60
ENTRY_HIGH_CONV_MOMENTUM_THR = 0.85
ENTRY_HIGH_CONV_PROB_THR     = 0.60

# ─── Guardian v3.5 ────────────────────────────────────────────────────────────
GUARDIAN_ENABLED            = True
GUARDIAN_EXIT_THRESHOLD     = 0.60
GUARDIAN_MIN_HOLD_BARS      = 3
GUARDIAN_ACTIVATION_ATR     = 1.0
GUARDIAN_PARTIAL_EXIT_RATIO = 0.50
GUARDIAN_EARLY_STOPPING     = 50
GUARDIAN_N_FOLDS            = 8
GUARDIAN_PURGE_GAP_BARS     = 24

GUARDIAN_DYNAMIC_FEATURES = [
    "bars_held_norm", "current_pnl_pct", "current_pnl_atr",
    "max_favorable_pnl_pct", "drawdown_from_peak_pct",
    "direction", "entry_price_ratio",
    # v3.5: tambah exhaustion & momentum
    "exhaustion_score", "momentum_strength",
]

# ─── Trailing Stop ────────────────────────────────────────────────────────────
TRAILING_STOP_ENABLED  = False
TRAILING_STOP_ATR      = 2.0
TRAILING_STOP_MIN_BARS = 2
