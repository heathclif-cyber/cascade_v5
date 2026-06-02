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

1. `python pipeline/01_fetch.py --all` (atau pilot 3 koin via `notebooks/cascade_v5_jupyter.py`)
2. `02_clean` → `03_engineer` → validasi distribusi label 5-class
3. `04_train_lgbm` → `python tools/overfitting_report.py`
4. `05a`–`05d`–`05c` → validasi loss LSTM per head
5. `06` Guardian → `07` holdout (Mei 2025 – Apr 2026)

---

## 2026-06-02 — Fine-Tuning Roadmap: Hal yang Belum Pasti

Catatan ini mendokumentasikan semua parameter yang saat ini di-hardcode tanpa validasi data.
Urutan prioritas: jalankan pipeline penuh dulu, baru sweep per kelompok.

---

### KELOMPOK A — Dynamic Fusion Weights
*Paling berpengaruh ke PnL. Ubah satu-satu, bukan sekaligus.*

| Parameter | Nilai Sekarang | Belum Pasti Karena |
|---|---|---|
| `FUSION_LGBM_WEIGHT` | 0.55 | Belum tahu seberapa besar LGBM berkontribusi vs residual di holdout |
| `FUSION_RESIDUAL_WEIGHT` | 0.35 | Bergantung pada kualitas residual LSTM — kalau MSE residual tinggi, weight ini harus diturunkan |
| `FUSION_EXHAUSTION_PENALTY` | 0.25 | Belum tahu apakah penalty 0.25 terlalu agresif (terlalu banyak sinyal diblok) atau terlalu lunak |
| `FUSION_MOMENTUM_BOOST_THR` | 0.75 | Ambang batas momentum yang "sangat kuat" — belum ada distribusi aktual momentum_strength di data real |
| `FUSION_MOMENTUM_BOOST_VAL` | 0.05 | Magnitude boost 0.05 — bisa jadi terlalu kecil untuk berdampak |

**Cara tune:**
1. Jalankan backtest baseline dengan nilai sekarang → catat PnL, WR, Sharpe
2. Grid sweep `FUSION_LGBM_WEIGHT` + `FUSION_RESIDUAL_WEIGHT` (pastikan sum ≤ 0.95):
   ```
   LGBM:     [0.45, 0.55, 0.65]
   Residual: [0.25, 0.35, 0.45]
   ```
3. Setelah bobot optimal ditemukan, sweep `FUSION_EXHAUSTION_PENALTY` di [0.15, 0.20, 0.25, 0.30]
4. Cek: apakah exhaustion penalty memang membuang trade yang loss? (Bandingkan WR trade yang diblok vs yang lolos)

**Metrik keberhasilan:** PnL holdout naik, jumlah trade tidak turun > 30% dari baseline.

---

### KELOMPOK B — Smart Entry Gate Thresholds
*Menentukan berapa banyak trade yang masuk. Terlalu tinggi = sedikit trade. Terlalu rendah = banyak noise.*

| Parameter | Nilai Sekarang | Belum Pasti Karena |
|---|---|---|
| `ENTRY_FINAL_PROB_THR` | 0.68 | Belum tahu distribusi `final_prob` di data holdout — mungkin 0.68 memotong terlalu banyak |
| `ENTRY_MOMENTUM_STR_THR` | 0.55 | Belum tahu distribusi `momentum_strength` output LSTM di data real |
| `ENTRY_EXHAUSTION_MAX` | 0.60 | Belum tahu berapa persen bar yang `exhaustion_score > 0.60` |
| `ENTRY_HIGH_CONV_MOMENTUM_THR` | 0.85 | High-conviction path — apakah 0.85 realistis atau hampir tidak pernah tercapai? |
| `ENTRY_HIGH_CONV_PROB_THR` | 0.60 | Apakah 0.60 terlalu mudah di-trigger saat momentum 0.85? |

**Cara tune:**
1. Setelah LSTM dilatih, plot distribusi `momentum_strength` dan `exhaustion_score` di seluruh data holdout
2. Cari percentile 70, 80, 90 dari masing-masing distribusi — jadikan acuan threshold
3. Hitung berapa trade per bulan yang lolos gate dengan nilai sekarang — target minimal 15-20 trade/bulan per coin
4. Kalau < 10 trade/bulan, turunkan threshold; kalau > 50 trade/bulan, naikkan

**Metrik keberhasilan:** Trade count cukup untuk statistik valid (≥ 150 total holdout), WR > 52%.

---

### KELOMPOK C — Exhaustion Detection Rules
*Rule-based, tapi ambang batasnya belum dikalibrasi ke distribusi data aktual.*

| Parameter | Nilai Sekarang | Belum Pasti Karena |
|---|---|---|
| `EXHAUSTION_SWING_ATR_THR` | 3.5 | Berapa persen bar yang `dist > 3.5 ATR`? Kalau terlalu jarang, fitur ini tidak efektif |
| `EXHAUSTION_VOL_DROP` | 0.7 | Volume drop < 70% dari rata-rata — apakah threshold ini cocok untuk semua coin? |
| `EXHAUSTION_WICK_RATIO` | 0.5 | Wick > 50% — cukup lunak, bisa jadi terlalu banyak false positives |
| `EXHAUSTION_ACCEL_SIGN_CHG` | True | Apakah acceleration sign change harus wajib (AND) atau opsional (OR)? |

**Cara tune:**
1. Hitung base rate: berapa persen bar H4 yang trigger masing-masing kondisi exhaustion?
   - Kalau < 5% → threshold terlalu ketat, tidak informatif
   - Kalau > 30% → threshold terlalu longgar, noise
2. Target base rate per kondisi: 10-20%
3. Cek korelasi: apakah bar dengan `exhaustion_score = 1` memang diikuti reversal dalam 6 H4 bar berikutnya?

**Metrik keberhasilan:** Precision exhaustion label (exhaustion bar → reversal dalam 6 bar) > 55%.

---

### KELOMPOK D — LGBM Label Thresholds
*Menentukan distribusi kelas. Kalau imbalanced, model bias ke FLAT.*

| Parameter | Nilai Sekarang | Belum Pasti Karena |
|---|---|---|
| `LGBM_STRONG_THR` | 2.5 | ATR multiplier untuk "strong" move — belum tahu berapa persen data masuk class strong vs weak |
| `LGBM_WEAK_THR` | 1.0 | Batas bawah untuk weak long/short — kalau terlalu rendah, terlalu banyak noise masuk sebagai signal |
| `LGBM_LABEL_HORIZON` | 18 bar H4 | ~3 hari ke depan — apakah ini horizon yang tepat untuk setup v5? |

**Cara tune:**
1. Plot distribusi 5 kelas setelah labeling: target distribusi kelas tidak lebih dari 5:1 antara FLAT dan STRONG
2. Kalau FLAT > 60% data → naikkan `LGBM_WEAK_THR` atau turunkan `LGBM_STRONG_THR`
3. Kalau STRONG < 5% data → turunkan `LGBM_STRONG_THR` ke 2.0

**Metrik keberhasilan:** Class balance 5-kelas tidak lebih ekstrem dari [15%, 20%, 30%, 20%, 15%].

---

### KELOMPOK E — H1 Confirmation Gate
*Hanya aktif kalau `H1_CONFIRMATION_ENABLED = True`. Belum tahu apakah filter ini membantu atau membuang trade bagus.*

| Parameter | Nilai Sekarang | Belum Pasti Karena |
|---|---|---|
| `H1_CONF_RSI_LONG_MIN` | 45.0 | RSI H1 ≥ 45 untuk LONG — apakah ini terlalu ketat kalau entry justru bagus di RSI 40-45? |
| `H1_CONF_RSI_SHORT_MAX` | 55.0 | RSI H1 ≤ 55 untuk SHORT — simetris dengan LONG |
| `H1_CONF_RET_DIRECTION` | True | Wajib H1 return searah — apakah ini membuang counter-trend yang valid? |

**Cara tune:**
1. Jalankan backtest dua kali: dengan `H1_CONFIRMATION_ENABLED = True` dan `False`
2. Bandingkan: WR, PnL, drawdown
3. Kalau True tidak lebih baik → disable atau longgarkan threshold RSI ke [40, 60]

**Metrik keberhasilan:** H1 confirmation meningkatkan WR minimal 2% vs tanpa konfirmasi.

---

### KELOMPOK F — LSTM Architecture & Training
*Convergence belum tervalidasi. Baru diketahui setelah training selesai.*

| Parameter | Nilai Sekarang | Belum Pasti Karena |
|---|---|---|
| `LSTM_HIDDEN` | 128 | Kapasitas model — mungkin underfitting (perlu 256) atau overfitting (cukup 64) |
| `LSTM_LAYERS` | 2 | Standard 2-layer — belum ada ablation |
| `LSTM_DROPOUT` | 0.3 | Standar, tapi belum diveri apakah val loss gap train/val menunjukkan overfit |
| `LSTM_SEQ_LEN` | 24 bar H4 | 4 hari context — apakah 16 (2.5 hari) atau 32 (5 hari) lebih baik? |
| `LSTM_LR` | 0.001 | Standard Adam LR — perlu dicek loss curve |

**Cara tune:**
1. Setelah training pertama, plot train loss vs val loss per epoch
   - Kalau val loss tidak turun setelah epoch 20 → turunkan LR ke 0.0005
   - Kalau gap train/val besar → naikkan LSTM_DROPOUT ke 0.4
2. Cek residual head MSE: kalau > 0.8 (near-random), residual tidak informatif → reduce `FUSION_RESIDUAL_WEIGHT`
3. Ablation LSTM_SEQ_LEN: test [16, 24, 32] — pilih yang validation loss terendah

**Metrik keberhasilan:** Residual head MSE < 0.5 (artinya LSTM mengkoreksi LGBM secara meaningful).

---

### KELOMPOK G — Guardian v3.5
*Logika guardian sudah benar, tapi threshold exit belum dikalibrasi.*

| Parameter | Nilai Sekarang | Belum Pasti Karena |
|---|---|---|
| `GUARDIAN_EXIT_THRESHOLD` | 0.60 | Apakah 0.60 terlalu sering trigger FULL exit (memotong profit) atau terlalu jarang? |
| `GUARDIAN_MIN_HOLD_BARS` | 3 bar H1 | 3 jam minimum — apakah cukup untuk menghindari noise keluar terlalu cepat? |
| `GUARDIAN_ACTIVATION_ATR` | 1.0 ATR | Guardian aktif setelah trade bergerak 1 ATR — apakah terlalu cepat? |
| `GUARDIAN_PARTIAL_EXIT_RATIO` | 0.50 | Keluar 50% posisi — apakah 30% atau 70% lebih optimal? |

**Cara tune:**
1. Jalankan backtest dengan `GUARDIAN_ENABLED = False` → catat PnL baseline
2. Jalankan dengan Guardian aktif → bandingkan
3. Kalau Guardian menurunkan PnL → naikkan `GUARDIAN_EXIT_THRESHOLD` ke 0.70
4. Analisis: berapa persen trade yang terkena partial exit? Berapa persen yang kemudian hit TP setelah partial?

**Metrik keberhasilan:** Guardian meningkatkan Sharpe ratio atau menurunkan max drawdown tanpa mengurangi total PnL > 10%.

---

### URUTAN EKSPERIMEN YANG DISARANKAN

```
Step 1: Jalankan pipeline penuh (01 → 07) dengan config sekarang
        → Catat: PnL holdout, WR, trade count, Sharpe

Step 2: Cek distribusi output model
        → Plot distribusi final_prob, momentum_strength, exhaustion_score
        → Cek label balance 5-kelas LGBM
        → Cek residual MSE LSTM

Step 3: Sweep KELOMPOK A (Fusion Weights)
        → 9 kombinasi LGBM + Residual weight

Step 4: Sweep KELOMPOK B (Entry Gate)
        → Grid 3×3 untuk ENTRY_FINAL_PROB_THR × ENTRY_MOMENTUM_STR_THR

Step 5: Ablasi KELOMPOK C (Exhaustion)
        → Test dengan FUSION_EXHAUSTION_PENALTY di [0.15, 0.20, 0.25, 0.30]

Step 6: Ablasi Guardian
        → On vs Off, lalu sweep GUARDIAN_EXIT_THRESHOLD [0.55, 0.60, 0.65, 0.70]

Step 7: Ablasi H1 Confirmation
        → On vs Off

Step 8: Finalisasi config terbaik → catat di EXPERIMENTS.md
```

---

### METRIK EVALUASI UTAMA

| Metrik | Target Minimum | Keterangan |
|---|---|---|
| PnL holdout (Mei 2025–Apr 2026) | > 0 USD | Dasar — harus profitable |
| Win Rate | > 52% | Di atas random untuk kripto dengan fee |
| Sharpe Ratio | > 1.0 | Risk-adjusted return layak |
| Max Drawdown | < 30% dari modal | Batas psikologis |
| Trade Count | ≥ 150 total | Cukup untuk statistik valid |
| Residual LSTM MSE | < 0.50 | LSTM memberikan koreksi bermakna |
| Exhaustion Precision | > 55% | Rule-based exhaustion aktually prediktif |
