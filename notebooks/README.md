# Notebook — pintu masuk utama Cascade v5

## Pakai file ini saja

### [`Cascade_v5_Jupyter.ipynb`](Cascade_v5_Jupyter.ipynb)

Satu notebook untuk **Colab** dan **Jupyter lokal**. Urutan lengkap: setup → fetch → train → laporan.

**Colab**

1. Runtime → **Change runtime type → GPU**
2. Clone: `!git clone https://github.com/heathclif-cyber/cascade_v5.git /content/cascade_v5`
3. `%cd /content/cascade_v5/notebooks`
4. Buka **`Cascade_v5_Jupyter.ipynb`**
5. Jalankan semua sel dari atas

**Konfigurasi:** edit variabel di **sel awal** (`USE_PILOT`, `RUN_*`).

---

## File lain di folder ini (bukan untuk user)

| File | Siapa |
|------|--------|
| `cascade_v5_jupyter.py` | Maintainer — sumber sel `# %%`; setelah edit: `python py_percent_to_ipynb.py` |
| `py_percent_to_ipynb.py` | Regenerate `.ipynb` dari `.py` |

Tidak perlu membuka file di atas kalau Anda hanya menjalankan pipeline di Colab.