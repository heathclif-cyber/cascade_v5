# EXPERIMENTS.md — Cascade v5 Logbook

## Template Entry

```
## YYYY-MM-DD — Deskripsi

### Config
| Parameter | Nilai |
...

### Hasil
| Metrik | Nilai |
...

### Temuan
1. ...

### Next
- [ ] ...
```

---

## 2026-06-02 — Inisialisasi Cascade v5

### Latar Belakang
Rebuild dari Cascade v4 karena:
1. LGBM confidence tidak terkalibrasi: loss punya conf lebih tinggi dari win (0.793 vs 0.790)
2. LSTM F1 ≈ random (0.3339 vs 0.333 baseline) — fitur snapshot H4 flat dalam 32 bar
3. Circular leakage: label forward scan → model belajar "setup yang menghasilkan TP" → 
   backtest dengan TP yang sama → WR tinggi by construction
4. LSTM neutral penalty = 0.00 → LGBM bisa decide sendiri tanpa LSTM

### Keputusan Arsitektur
- Feature separation: LGBM tabular (60 fitur), LSTM trajectory (20 fitur tempvar > 0.30)
- AttentionLSTM dengan 3 output heads: momentum_strength, exhaustion_score, residual_correction
- Dynamic Fusion (bukan scalar hard_consensus)
- Smart Entry Gate (normal + high-conviction path)
- MODAL_PER_TRADE = 5.0 USD (mutlak)
- Guardian v3.5 dipertahankan + enhanced dengan exhaustion features

### Feature Selection Results (dari Riset_pemodelan/_analysis/feature_separation.py)
- Total input v4: 104 fitur
- LGBM v5: 60 fitur (drop snapshot fitur yang temporal variance = 0 untuk LGBM)
- LSTM v5: 20 fitur (temporal variance > 0.30 — berubah setiap H1 bar)
- DROP: 5 fitur (FVG_up, FVG_down, dynamic_position_pressure, hidden_divergence, sell_volume)

### Status
- [x] Repo dibuat
- [x] config.py dengan MODAL=5.0
- [x] CLAUDE.md
- [x] core/models.py (AttentionLSTM)
- [x] core/fusion.py (Dynamic Fusion + Smart Entry Gate)
- [x] pipeline/05a (momentum + exhaustion labels)
- [x] pipeline/05b (sequence builder)
- [ ] core/features.py (feature engineering split)
- [ ] pipeline/01-04 (fetch → engineer → LGBM training)
- [ ] pipeline/05c (train AttentionLSTM)
- [ ] pipeline/05d (OOF residuals)
- [ ] pipeline/06 (Guardian v3.5)
- [ ] pipeline/07 (holdout backtest)

### Next Steps
1. Buat core/features.py dengan fitur split (LGBM vs LSTM)
2. Copy dan adapt pipeline 01-04 dari Riset_pemodelan
3. Train LGBM dengan 60 fitur → validate feature importance
4. Generate momentum labels (05a) → cek distribusi
5. Build sequences (05b) → verify temporal variance
6. Train AttentionLSTM (05c) → target F1 > 0.40
