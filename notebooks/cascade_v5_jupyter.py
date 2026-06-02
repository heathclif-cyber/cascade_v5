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
# # Cascade v5 — Colab / Jupyter
#
# **Colab:** Runtime → **GPU** → jalankan **semua sel dari atas** (jangan loncat).
#
# Log pipeline tampil langsung di cell (bukan subprocess sunyi).
# Tabel **STATUS DATA** muncul setelah fetch/clean/engineer.
#
# **MODAL_PER_TRADE = 5.0 USD** (jangan diubah)

# %% [markdown]
# ## 0a. Colab — clone / update repo (WAJIB sel pertama)

# %%
import sys
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules
ROOT = Path("/content/cascade_v5") if IN_COLAB else Path.cwd()
if not (ROOT / "config.py").exists() and (ROOT / "notebooks").exists():
    ROOT = ROOT.parent if (ROOT.parent / "config.py").exists() else ROOT

if IN_COLAB:
    import subprocess
    if not (ROOT / "config.py").exists():
        subprocess.call([
            "git", "clone", "--depth", "1",
            "https://github.com/heathclif-cyber/cascade_v5.git",
            str(ROOT),
        ])
    else:
        subprocess.call(["git", "-C", str(ROOT), "pull", "origin", "master"])
    # Cek patch bug CV (wajib ada di repo terbaru)
    shared = (ROOT / "pipeline" / "shared.py").read_text(encoding="utf-8")
    if "_build_purged_folds_ordinal" not in shared:
        raise RuntimeError(
            "Repo masih versi lama. Runtime -> Restart -> jalankan sel ini lagi, "
            "atau hapus folder /content/cascade_v5 lalu clone ulang."
        )
    print("Colab repo OK:", ROOT)
else:
    print("Mode lokal, ROOT =", ROOT.resolve())

# %% [markdown]
# ## 0b. Konfigurasi — edit di sini

# %%
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

# ROOT & IN_COLAB sudah dari sel 0a
if not IN_COLAB:
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
# ## 1. Bootstrap environment

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
# ## 2. Helper + status (log informatif di Jupyter)

# %%
from tools.jupyter_ui import (
    enable_verbose_logging,
    status_table,
    step_start,
    step_ok,
    step_fail,
)

enable_verbose_logging()
os.environ["CASCADE_JUPYTER"] = "1"
ACTIVE_COINS = PILOT_COINS if USE_PILOT else TRAINING_COINS
status_table(ACTIVE_COINS, ROOT)


def run_inprocess(rel_path: str, *argv: str) -> None:
    """Jalankan skrip di kernel yang sama — log + banner jelas."""
    import runpy

    path = ROOT / rel_path
    if not path.exists():
        raise FileNotFoundError(f"Tidak ada: {path}")

    os.chdir(ROOT)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    if IN_COLAB:
        os.environ["CASCADE_COLAB"] = "1"
    os.environ["PYTHONUNBUFFERED"] = "1"

    label = path.name
    t0 = step_start(label, " ".join(argv))
    old_argv = sys.argv[:]
    sys.argv = [str(path)] + list(argv)
    try:
        runpy.run_path(str(path), run_name="__main__")
        step_ok(label, t0)
    except SystemExit as e:
        code = e.code if e.code is not None else 0
        if code != 0:
            step_fail(label, t0, f"exit code {code}")
            raise RuntimeError(f"{rel_path} gagal (exit {code})") from e
        step_ok(label, t0)
    except Exception as e:
        step_fail(label, t0, str(e))
        raise
    finally:
        sys.argv = old_argv
        sys.stdout.flush()
        sys.stderr.flush()


def pipeline_argv() -> list[str]:
    """Argumen --coins / --all (+ opsional --holdout)."""
    args = (["--coins"] + PILOT_COINS) if USE_PILOT else ["--all"]
    if HOLDOUT_FLAG.strip():
        args.append("--holdout")
    return args


def run_audit() -> None:
    coins = PILOT_COINS if USE_PILOT else TRAINING_COINS
    run_inprocess("tools/audit_pipeline.py", "--coins", *coins)

# %% [markdown]
# ## 3. Data — fetch, clean, engineer

# %%
if RUN_FETCH_CLEAN_ENGINEER:
    pargs = pipeline_argv()
    print("Tahap 1/3: FETCH (Binance, bisa 10-30 menit untuk 3 koin)...", flush=True)
    run_inprocess("pipeline/01_fetch.py", *pargs)
    status_table(ACTIVE_COINS, ROOT)

    print("Tahap 2/3: CLEAN...", flush=True)
    run_inprocess("pipeline/02_clean.py", *pargs)
    status_table(ACTIVE_COINS, ROOT)

    print("Tahap 3/3: ENGINEER...", flush=True)
    run_inprocess("pipeline/03_engineer.py", *pargs)
    status_table(ACTIVE_COINS, ROOT)
    run_audit()

# %% [markdown]
# ## 4. Train LGBM (5-class, purged CV)

# %%
def _preflight_lgbm_data() -> None:
    from config import LABEL_DIR
    files = list(LABEL_DIR.glob("*_h4_lgbm.parquet"))
    if not files:
        raise RuntimeError(
            "Tidak ada data/labeled/*_h4_lgbm.parquet — "
            "jalankan sel 3 (fetch/clean/engineer) dulu."
        )
    print(f"Data OK: {len(files)} file parquet")


if RUN_TRAIN_LGBM:
    _preflight_lgbm_data()
    run_inprocess("pipeline/04_train_lgbm.py", "--all")

# %% [markdown]
# ## 5. LSTM — labels, sequences, OOF residual, train

# %%
if RUN_LSTM_PIPELINE:
    ca = pipeline_argv()
    run_inprocess("pipeline/05a_momentum_labels.py", *ca)
    run_inprocess("pipeline/05b_build_sequences.py", *ca)
    run_inprocess("pipeline/05d_oof_residuals.py", "--all")
    run_inprocess("pipeline/05c_train_momentum_expert.py", "--all", "--run-id", RUN_ID)

# %% [markdown]
# ## 6. Guardian v3.5

# %%
if RUN_GUARDIAN:
    run_inprocess("pipeline/06_train_guardian.py", *pipeline_argv())

# %% [markdown]
# ## 7. Holdout backtest (opsional)

# %%
if RUN_HOLDOUT:
    if RUN_HOLDOUT_FETCH:
        run_inprocess("pipeline/01_fetch.py", "--all", "--holdout")
        run_inprocess("pipeline/02_clean.py", "--all", "--holdout")
        run_inprocess("pipeline/03_engineer.py", "--all", "--holdout")
    args = pipeline_argv() + ["--run-id", RUN_ID]
    run_inprocess("pipeline/07_holdout_backtest.py", *args)

# %% [markdown]
# ## 8. Laporan — benchmark + overfitting

# %%
if RUN_REPORTS:
    for tool, targv in [
        ("tools/benchmark_plan.py", []),
        ("tools/overfitting_report.py", []),
    ]:
        try:
            run_inprocess(tool, *targv)
        except RuntimeError:
            print(f"(skip {tool})")
    if (ROOT / "models" / "runs").exists():
        runs = sorted((ROOT / "models" / "runs").glob("holdout_*"))
        if runs:
            try:
                run_inprocess(
                    "tools/overfitting_report.py",
                    "--holdout-run",
                    str(runs[-1].relative_to(ROOT)),
                )
            except RuntimeError:
                pass

# %% [markdown]
# ## 9. Simpan ke Google Drive (Colab, opsional)

# %%
if IN_COLAB and False:  # ubah ke True untuk backup
    from google.colab import drive

    drive.mount("/content/drive")
    dest = "/content/drive/MyDrive/cascade_v5_backup"
    subprocess.call(f"mkdir -p {dest} && cp -r data models reports {dest}", shell=True)
    print("Backup ke", dest)

# %% [markdown]
# ## 10. Cek artefak

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