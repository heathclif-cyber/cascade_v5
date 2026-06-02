# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Cascade v5 — Skrip Jupyter (Colab / lokal)
#
# Buka file ini sebagai notebook:
# - **VS Code / Cursor:** buka `.py` → klik "Run Cell" di setiap `# %%`
# - **Jupyter:** `jupytext --to notebook cascade_v5_jupyter.py` lalu buka `.ipynb`
# - **Colab:** upload repo + file ini, runtime **GPU**
#
# **MODAL_PER_TRADE = 5.0 USD** (jangan diubah)

# %% [markdown]
# ## 0. Konfigurasi — edit di sini

# %%
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

# --- path repo (Colab vs lokal) ---
IN_COLAB = "google.colab" in sys.modules

if IN_COLAB:
    ROOT = Path("/content/cascade-v5-architecture")
else:
    # notebook di notebooks/ → parent = repo root
    _here = Path.cwd()
    ROOT = _here if (_here / "config.py").exists() else _here.parent

# --- koin ---
# Pilot cepat (Colab): 3 koin | Full: None → pakai TRAINING_COINS di 04
PILOT_COINS = ["SOLUSDT", "ETHUSDT", "BNBUSDT"]
USE_PILOT = True  # False = --all di fetch/clean/engineer

# --- langkah pipeline (True/False) ---
RUN_SETUP = True
RUN_FETCH_CLEAN_ENGINEER = True
RUN_TRAIN_LGBM = True
RUN_LSTM_PIPELINE = True  # 05a → 05b → 05d → 05c
RUN_GUARDIAN = True
RUN_HOLDOUT = False       # butuh data holdout 01-03 dulu
RUN_REPORTS = True

# --- holdout ---
RUN_HOLDOUT_FETCH = False  # 01-03 --holdout
RUN_ID = "jupyter_run"

COINS_ARG = " ".join(PILOT_COINS) if USE_PILOT else ""
ALL_FLAG = "" if USE_PILOT else "--all"
HOLDOUT_FLAG = "--holdout" if RUN_HOLDOUT_FETCH else ""

print("IN_COLAB:", IN_COLAB)
print("ROOT:", ROOT.resolve())

# %% [markdown]
# ## 1. Clone / mount (Colab saja)

# %%
if IN_COLAB:
    REPO_URL = ""  # isi URL git jika clone; kosong = sudah upload manual
    if REPO_URL:
        subprocess.call(["git", "clone", REPO_URL, str(ROOT)])
    else:
        print(f"Pastikan repo ada di: {ROOT}")
        print("Atau: Files upload zip -> !unzip -q repo.zip -d /content/")

# %% [markdown]
# ## 2. Bootstrap environment

# %%
os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if RUN_SETUP:
    if IN_COLAB:
        from tools.colab_bootstrap import setup_colab

        setup_colab(repo_root=ROOT, mount_drive=False, install_deps=True)
    else:
        req = ROOT / "requirements-colab.txt"
        if req.exists():
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-q", "-r", str(req)]
            )
        import torch
        from core.utils import get_lstm_device

        print("Mode: lokal / Jupyter")
        print("LSTM device:", get_lstm_device())
        print("CUDA:", torch.cuda.is_available())

from config import COLAB_QUICK_COINS, TRAINING_COINS, MODAL_PER_TRADE

print("MODAL_PER_TRADE:", MODAL_PER_TRADE)
print("Coins:", PILOT_COINS if USE_PILOT else f"ALL ({len(TRAINING_COINS)})")

# %% [markdown]
# ## 3. Helper — jalankan perintah pipeline

# %%
def run(cmd: str, check: bool = True) -> int:
    """Jalankan shell command; print stdout realtime."""
    print("\n" + "=" * 60)
    print("$", cmd)
    print("=" * 60)
    rc = subprocess.call(cmd, shell=True, cwd=str(ROOT))
    if check and rc != 0:
        raise RuntimeError(f"Command failed (exit {rc}): {cmd}")
    return rc


def coin_flags() -> str:
    return f"--coins {COINS_ARG}" if USE_PILOT else "--all"

# %% [markdown]
# ## 4. Data — fetch, clean, engineer

# %%
if RUN_FETCH_CLEAN_ENGINEER:
    run(f"python pipeline/01_fetch.py {coin_flags()} {HOLDOUT_FLAG}".strip())
    run(f"python pipeline/02_clean.py {coin_flags()} {HOLDOUT_FLAG}".strip())
    run(f"python pipeline/03_engineer.py {coin_flags()} {HOLDOUT_FLAG}".strip())

# %% [markdown]
# ## 5. Train LGBM (5-class, purged CV)

# %%
if RUN_TRAIN_LGBM:
    # 04 hanya --all: pakai semua *_h4_lgbm.parquet yang ada
    run("python pipeline/04_train_lgbm.py --all")

# %% [markdown]
# ## 6. LSTM — labels, sequences, OOF residual, train

# %%
if RUN_LSTM_PIPELINE:
    run(f"python pipeline/05a_momentum_labels.py {coin_flags()}")
    run(f"python pipeline/05b_build_sequences.py {coin_flags()}")
    run("python pipeline/05d_oof_residuals.py --all")
    run("python pipeline/05c_train_momentum_expert.py --all --run-id " + RUN_ID)

# %% [markdown]
# ## 7. Guardian v3.5

# %%
if RUN_GUARDIAN:
    run(f"python pipeline/06_train_guardian.py {coin_flags()}")

# %% [markdown]
# ## 8. Holdout backtest (opsional)

# %%
if RUN_HOLDOUT:
    if RUN_HOLDOUT_FETCH:
        run("python pipeline/01_fetch.py --all --holdout")
        run("python pipeline/02_clean.py --all --holdout")
        run("python pipeline/03_engineer.py --all --holdout")
    run(f"python pipeline/07_holdout_backtest.py {coin_flags()} --run-id {RUN_ID}")

# %% [markdown]
# ## 9. Laporan — benchmark + overfitting

# %%
if RUN_REPORTS:
    run("python tools/benchmark_plan.py", check=False)
    run("python tools/overfitting_report.py", check=False)
    if (ROOT / "models" / "runs").exists():
        runs = sorted((ROOT / "models" / "runs").glob("holdout_*"))
        if runs:
            run(f"python tools/overfitting_report.py --holdout-run {runs[-1]}", check=False)

# %% [markdown]
# ## 10. Simpan ke Google Drive (Colab, opsional)

# %%
if IN_COLAB and False:  # ubah ke True untuk backup
    from google.colab import drive

    drive.mount("/content/drive")
    dest = "/content/drive/MyDrive/cascade_v5_backup"
    run(f"mkdir -p {dest} && cp -r data models reports {dest}", check=False)
    print("Backup ke", dest)

# %% [markdown]
# ## 11. Cek artefak

# %%
from pathlib import Path

artifacts = [
    "models/lgbm_tabular.pkl",
    "models/lgbm_cv_results.json",
    "models/lstm_momentum_expert.pt",
    "models/lstm_cv_results.json",
    "models/guardian_best.pkl",
]
for rel in artifacts:
    p = ROOT / rel
    tag = "OK" if p.exists() else "MISSING"
    print(f"[{tag}] {rel}")

reports = sorted((ROOT / "reports").glob("overfitting_*.md")) if (ROOT / "reports").exists() else []
if reports:
    print("\nLaporan overfitting terbaru:", reports[-1])