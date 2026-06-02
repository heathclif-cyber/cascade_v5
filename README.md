# Cascade v5 — Focused Residual + Exhaustion Hybrid (H4 Primary)

Trading ML untuk Binance Futures. **MODAL_PER_TRADE = 5.0 USD** (mutlak).

## Mulai di sini (utama)

| Platform | Langkah |
|----------|---------|
| **Google Colab** | Clone repo → Runtime **GPU** → buka **[`notebooks/Cascade_v5_Jupyter.ipynb`](notebooks/Cascade_v5_Jupyter.ipynb)** → jalankan sel berurutan |
| **Jupyter lokal** | File yang sama di folder `notebooks/` |
| **Terminal** | Hanya jika perlu: `python pipeline/01_fetch.py` … (urutan di `CLAUDE.md`) |

```python
# Colab — setelah clone:
!git clone https://github.com/heathclif-cyber/cascade_v5.git /content/cascade_v5
%cd /content/cascade_v5/notebooks
# Buka Cascade_v5_Jupyter.ipynb di UI Colab
```

Detail notebook: [`notebooks/README.md`](notebooks/README.md)

## Dokumen

- [`CLAUDE.md`](CLAUDE.md) — arsitektur lengkap
- [`EXPERIMENTS.md`](EXPERIMENTS.md) — log hasil run
- [`config.py`](config.py) — semua parameter

## Setelah training

```powershell
python tools/overfitting_report.py
python tools/benchmark_plan.py
```

## Clone

```bash
git clone https://github.com/heathclif-cyber/cascade_v5.git
cd cascade_v5/notebooks
```