# Jupyter / Colab

## File utama

| File | Cara pakai |
|------|------------|
| **`cascade_v5_jupyter.py`** | Skrip `# %%` — buka di VS Code/Cursor, klik *Run Cell* per bagian |
| **`Cascade_v5_Jupyter.ipynb`** | Notebook hasil konversi (25 sel) — Colab / JupyterLab |
| `Cascade_v5_Colab.ipynb` | Versi ringkas Colab (legacy) |
| `py_percent_to_ipynb.py` | Regenerate `.ipynb` setelah edit `.py` |

```powershell
cd notebooks
python py_percent_to_ipynb.py
```

## Konfigurasi (sel 0 di skrip)

Edit di `cascade_v5_jupyter.py`:

```python
USE_PILOT = True          # 3 koin vs 20 koin
RUN_FETCH_CLEAN_ENGINEER = True
RUN_TRAIN_LGBM = True
RUN_HOLDOUT = False
```

## Colab

1. Runtime → **GPU**
2. Upload repo ke `/content/cascade-v5-architecture`
3. Buka `Cascade_v5_Jupyter.ipynb` atau `cascade_v5_jupyter.py`
4. Jalankan sel 0 → 9

## Lokal (Jupyter / VS Code)

1. Buka folder repo sebagai workspace
2. Buka `notebooks/cascade_v5_jupyter.py`
3. Pilih kernel Python 3.12 + GPU/CUDA jika ada
4. `USE_PILOT = True` untuk uji cepat

## Laporan

Sel 9 menjalankan `tools/benchmark_plan.py` dan `tools/overfitting_report.py`.