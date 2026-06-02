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

### Keputusan Arsitektur (sinkron share 55e525f4 — teks lengkap dari user)
- Filosofi: sederhana, interpretable, anti-noise; otomatisasi swing/exhaustion/acceleration/structure
- H4 primary, H1 gate only; ~66 fitur (56+10)
- Label exhaustion arah: long/short 3.5 ATR + accel sign (`compute_exhaustion_directional`)
- Swing: backward-only di kode (bukan center=True share — anti look-ahead)

### Implementasi teknis
- LGBM tabular: **56 fitur H4** (`config.LGBM_FEATURE_COLS`)
- LSTM trajectory: **10 fitur H4**, sequence **24 bar**
- LGBM label: **5-class** ATR-return (bukan 3-class TP scan)
- AttentionLSTM: 3 heads (momentum, exhaustion, residual)
- Dynamic Fusion + Smart Entry Gate + H1 confirmation
- MODAL_PER_TRADE = 5.0 USD (mutlak)
- Guardian v3.5 + 9 dynamic features (exhaustion, momentum)

### Referensi rencana
- [Grok Share — Arsitektur lengkap + GitHub quant (utama)](https://grok.com/share/bGVnYWN5LWNvcHk_55e525f4-0219-45ca-b213-2cb8ec512b5a)
- [Grok Share — Penjelasan detail per model](https://grok.com/share/bGVnYWN5LWNvcHk_8697395c-f55a-47d1-81ef-e9708a33a48a)
- Dokumen repo: `CLAUDE.md`, checklist: `python tools/benchmark_plan.py`

### Feature Selection (audit v4 → v5)
- Total input v4: 104 fitur
- LGBM v5: 56 fitur tabular H4 (snapshot + structure OK untuk LGBM)
- LSTM v5: 10 fitur trajectory H4 (derived, bukan 20 fitur H1 audit)
- DROP: 5 fitur (FVG_up, FVG_down, dynamic_position_pressure, hidden_divergence, sell_volume)

### Status Implementasi
- [x] Repo & `config.py` (MODAL=5, H4 primary)
- [x] `CLAUDE.md` disinkronkan dengan Grok share + kode
- [x] `core/features.py`, `core/models.py`, `core/fusion.py`, `core/evaluator.py`
- [x] Pipeline `01`–`07` + `pipeline/shared.py`
- [x] Integrasi path H4/H1 (06, 07, 05d)
- [x] `tools/benchmark_plan.py`
- [ ] Data fetch + clean (belum dijalankan)
- [ ] Train LGBM / LSTM / Guardian
- [ ] Holdout backtest hasil metrik

### Next Steps
1. `python pipeline/01_fetch.py --all`
2. `02_clean` → `03_engineer` → validasi distribusi label 5-class
3. `04_train_lgbm` → cek feature importance
4. `05a`–`05d`–`05c` → validasi loss LSTM per head
5. `06` Guardian → `07` holdout (Mei 2025 – Apr 2026)