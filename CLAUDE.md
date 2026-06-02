# CLAUDE.md — Cascade v5: Exhaustion-Aware Residual Momentum Hybrid

## Overview

Sistem trading kripto berbasis ML untuk Binance Futures.
Arsitektur baru yang mengatasi kelemahan utama Cascade v4:
- LGBM chase pump (masuk saat harga sudah over-extended)
- LSTM F1 ≈ random karena input fitur snapshot yang flat
- Confidence LGBM tidak terkalibrasi (loss punya conf lebih tinggi dari win)

**MODAL_PER_TRADE = 5.0 USD — MUTLAK, tidak boleh diubah**

Periode: 2020-01-01 s/d 2026-04-01. Timeframe: H1 base, H4 context.
TRAIN_CUTOFF_DATE = 2025-05-01

## Arsitektur

```
Raw Data H1 + H4
      |
Feature Engineering Split (03_engineer.py)
      |                    |
LGBM Features (~60)   LSTM Sequence Features (20)
tabular + structure   trajectory + exhaustion
      |                    |
LGBM Tabular      Momentum + Exhaustion Expert
Expert            (Attention LSTM)
3-class           Outputs:
[p_short,          - momentum_strength (0-1)
 p_flat,           - exhaustion_score  (0-1)
 p_long]           - residual_correction
      |                    |
      +----Dynamic Fusion--+
           Layer
           final_prob = 0.55*lgbm + 0.35*residual
                      - 0.25*exhaustion
                      + momentum_boost (if momentum > 0.75)
                |
         Smart Entry Gate
         final_prob > 0.68 AND
         momentum_strength > 0.55 AND
         exhaustion_score < 0.60
         [High-Conviction Path: momentum > 0.85]
                |
           Guardian v3.5
           HOLD / PARTIAL_EXIT / FULL_EXIT
           (+ exhaustion & momentum features)
                |
           TP/SL Hybrid (H4 Swing + ATR)
```

## Perbedaan dari Cascade v4

| Aspek | Cascade v4 | Cascade v5 |
|-------|-----------|-----------|
| Entry model | LGBM H1 (93 fitur, semua campur) | LGBM H1 (60 fitur tabular saja) |
| Sequence model | ManualLSTMCell (F1 ≈ random) | Attention LSTM + Momentum/Exhaustion heads |
| LSTM input | Snapshot H4 features (flat) | Trajectory features (tempvar > 0.30) |
| Fusion | Scalar hard_consensus | Dynamic weighted fusion |
| Exhaustion detection | Tidak ada | Exhaustion score + penalty |
| Modal per trade | $100 | **$5 (mutlak)** |
| Label | Forward scan TP/SL (circular) | OOF residual + rule-based exhaustion |

## Feature Split

### LGBM Tabular Expert (~60 fitur)
- Liquidation distances (top importance dari audit)
- Fear & greed, BTC dominance (macro)
- ATR statistics, OI, funding rate
- Price levels: PDH/PDL/PWH/PWL, Swing high/low
- Volume profile: POC, VAH, VAL
- Market structure tabular: bars_since_BOS, absorption_at_swing

### LSTM Sequence Expert (20 fitur — temporal variance > 0.30)
```
time_to_funding_norm  1.016  ← berubah tiap jam
swing_momentum        0.962
ofi_z_score           0.952
absorption_z          0.956
effort_vs_result      0.951
rsi_slope_h4          0.918
log_ret_1             0.917
rsi_6                 0.909
stochrsi_k            0.938
stochrsi_d            0.929
vol_ratio_20          0.902
vol_regime            0.881
ema_7_h1              0.887
dist_from_8h_high     0.885
log_ret_5             0.832
ofi_acceleration      0.353
volume_delta          0.319
h4_trend              0.606
dist_swing_low        0.672
log_ret_20            0.622
```

### DROP (tidak dipakai)
FVG_up, FVG_down, dynamic_position_pressure, hidden_divergence, sell_volume

## Label Strategy (Anti-Leakage)

### Masalah v4: Circular Leakage
```
Label: "did price reach TP in next 24 bars?" — forward scan
Model: belajar prediksi label
Backtest: evaluasi dengan TP/SL yang sama
Result: WR tinggi by construction, bukan generalisasi
```

### Solusi v5: 3 Label Terpisah

**1. LGBM Label** — tetap swing-based (OOF purge ketat 24 bar)
- Purge gap ditingkatkan dari 5 → 24 bars di TRAIN_CUTOFF_DATE boundary
- Membuang bars dalam radius 24 dari cutoff

**2. Exhaustion Label** — rule-based, tidak forward scan
```python
exhaustion = (
    volume_ratio_3bar < 0.7           # volume melemah
    AND abs(price_move) > 2.0 * ATR  # sudah bergerak jauh
    AND candle_wick_ratio > 0.5       # wick besar = ragu-ragu
)
```

**3. Residual Correction Label** — OOF stacking
```
Stage 1: Train LGBM dengan purged CV → simpan OOF predictions
Stage 2: residual = actual_label - lgbm_oof_proba
         LSTM belajar prediksi residual ini
         (tidak ada leakage karena LGBM tidak pernah lihat OOF bars saat prediksi)
```

## Pipeline

```
01_fetch.py           → Fetch H1 + H4 data dari Binance
02_clean.py           → Clean + resample
03_engineer.py        → Feature engineering (split LGBM / LSTM)
04_train_lgbm.py      → Train LGBM Tabular Expert (60 fitur)
05a_momentum_labels.py → Generate momentum + exhaustion labels (rule-based)
05b_build_sequences.py → Build LSTM sequences (20 trajectory features × 32 bars)
05c_train_momentum_expert.py → Train Attention LSTM (3 output heads)
05d_oof_residuals.py  → Compute OOF residuals dari LGBM untuk LSTM training
06_train_guardian.py  → Guardian v3.5 (+ exhaustion/momentum features)
07_holdout_backtest.py → OOS evaluation (Mei 2025 – Apr 2026)
```

## Key Files

| File | Role |
|------|------|
| `config.py` | Source of truth semua parameter |
| `core/features.py` | Feature engineering (LGBM + LSTM split) |
| `core/models.py` | AttentionLSTM + Momentum/Exhaustion heads |
| `core/evaluator.py` | simulate_trades_swing + Guardian |
| `core/fusion.py` | Dynamic Fusion Layer + Smart Entry Gate |
| `pipeline/shared.py` | build_purged_folds + SequenceDataset |

## Constraints

- Python 3.12, Windows, AMD RX 6600 (DirectML)
- Shell: PowerShell (chain dengan `;`)
- Encoding terminal: cp1252 — hindari unicode di logger
- LGBM: device_type="gpu" OpenCL
- LSTM: train DirectML, infer CPU
- MODAL_PER_TRADE = 5.0 — tidak boleh diubah
- TRAIN_CUTOFF_DATE = 2025-05-01 — tidak boleh bocor ke training

## Known Issues (dari Audit Arsitektur 2026-06-02)

| Severity | Komponen | Masalah |
|---|---|---|
| HIGH | `pipeline/05c_train_momentum_expert.py` | CV fold pakai `np.arange(len(X))` bukan timestamp — purge ordinal bukan temporal |
| LOW | `pipeline/05d_oof_residuals.py` line 115 | Residual bar terakhir di-clip ke `len-1` → re-used untuk trailing sequences |

## Fine-Tuning Agenda

Parameter berikut **belum dikalibrasi dari data** — semua hardcode tanpa validasi holdout.
Detail sweep dan metrik keberhasilan ada di EXPERIMENTS.md.

### Prioritas 1 — Jalankan dulu, cek distribusi output model sebelum sweep apapun

### Prioritas 2 — Fusion Weights (paling berpengaruh ke PnL)
```python
FUSION_LGBM_WEIGHT        = 0.55   # belum tahu bobot ideal vs kualitas LGBM di holdout
FUSION_RESIDUAL_WEIGHT    = 0.35   # bergantung pada residual LSTM MSE
FUSION_EXHAUSTION_PENALTY = 0.25   # mungkin terlalu agresif / terlalu lunak
FUSION_MOMENTUM_BOOST_THR = 0.75   # belum ada distribusi momentum_strength aktual
FUSION_MOMENTUM_BOOST_VAL = 0.05   # mungkin terlalu kecil untuk berdampak
```

### Prioritas 3 — Smart Entry Gate (trade count vs kualitas)
```python
ENTRY_FINAL_PROB_THR         = 0.68   # distribusi final_prob belum diketahui
ENTRY_MOMENTUM_STR_THR       = 0.55   # distribusi momentum_strength belum diketahui
ENTRY_EXHAUSTION_MAX         = 0.60   # berapa % bar yang exhaustion > 0.60?
ENTRY_HIGH_CONV_MOMENTUM_THR = 0.85   # apakah realistis dicapai?
```

### Prioritas 4 — Exhaustion Rules (base rate belum divalidasi)
```python
EXHAUSTION_SWING_ATR_THR = 3.5   # target base rate 10-20% bar H4
EXHAUSTION_VOL_DROP      = 0.7   # cocok untuk semua coin?
EXHAUSTION_WICK_RATIO    = 0.5   # mungkin terlalu banyak false positives
```

### Prioritas 5 — Ablasi (on/off comparison)
- `H1_CONFIRMATION_ENABLED` — apakah H1 gate meningkatkan WR minimal 2%?
- `GUARDIAN_ENABLED` — apakah Guardian meningkatkan Sharpe tanpa mengurangi PnL > 10%?
- `GUARDIAN_EXIT_THRESHOLD = 0.60` — sweep [0.55, 0.60, 0.65, 0.70]
- `LSTM_SEQ_LEN = 24` — ablasi [16, 24, 32 bar H4]

### Target Metrik Holdout (Mei 2025 – Apr 2026)
| Metrik | Target |
|---|---|
| PnL | > 0 USD |
| Win Rate | > 52% |
| Sharpe | > 1.0 |
| Max Drawdown | < 30% modal |
| Trade Count | ≥ 150 total |
| Residual LSTM MSE | < 0.50 |

## Experiments Log

Lihat EXPERIMENTS.md untuk catatan hasil per run dan detail sweep setiap kelompok.

### Konvensi File

- **Model:** overwrite file yang sudah ada di `models/` — jangan buat `model_v2`, `model_final2`, dst.
- **Eksperimen:** tambahkan entry baru di EXPERIMENTS.md (satu file) — jangan buat file eksperimen terpisah
- **Config:** perubahan parameter dicatat di EXPERIMENTS.md, bukan duplikasi config.py
