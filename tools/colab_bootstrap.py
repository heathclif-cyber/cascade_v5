"""
tools/colab_bootstrap.py — Bootstrap Cascade v5 di Google Colab

Dipanggil dari notebook sel "Bootstrap environment":
    from tools.colab_bootstrap import setup_colab
    setup_colab(repo_root=ROOT, mount_drive=False, install_deps=True)
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def setup_colab(
    repo_root: "str | Path",
    mount_drive: bool = False,
    install_deps: bool = True,
    drive_dest: str = "/content/drive/MyDrive/cascade_v5_backup",
) -> None:
    """
    Install deps, set env vars, dan print ringkasan device untuk Colab run.

    Args:
        repo_root    : root folder repo (Path atau str)
        mount_drive  : mount Google Drive untuk backup model
        install_deps : pip install dari requirements-colab.txt
        drive_dest   : path Drive tujuan backup (hanya jika mount_drive=True)
    """
    repo_root = Path(repo_root)

    SEP = "=" * 58
    print(SEP)
    print("  Cascade v5 — Colab Bootstrap")
    print(SEP)

    # Wajib set SEBELUM import apapun dari config (auto-applies CPU fallback)
    os.environ["CASCADE_COLAB"] = "1"
    print("[ENV] CASCADE_COLAB=1 — LightGBM akan pakai CPU (OpenCL tidak tersedia di Colab)")

    # --- Install dependencies ---
    if install_deps:
        req = repo_root / "requirements-colab.txt"
        if req.exists():
            print(f"\n[PIP] Installing dari {req.name} ...")
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "-q", "-r", str(req)],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                print("[PIP][WARN] Ada error saat install:")
                print(result.stderr[-2000:])
            else:
                print("[PIP] Selesai.")
        else:
            print(f"[PIP][WARN] {req} tidak ditemukan — skip install")

    # --- GPU / device report ---
    print()
    try:
        import torch  # noqa: PLC0415

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            mem_gb = props.total_memory / 1_073_741_824
            print(f"[GPU] {props.name}  ({mem_gb:.1f} GB VRAM)")
            print("[GPU] LSTM training : CUDA")
        else:
            print("[GPU] Tidak terdeteksi — LSTM training di CPU (lebih lambat ~3-5x)")
    except ImportError:
        print("[GPU][WARN] torch belum tersedia — jalankan ulang sel ini setelah install")

    # --- Pastikan direktori output ada ---
    for d in ("data/raw", "data/processed", "data/labeled", "data/sequences",
              "data/holdout", "models/runs", "reports/experiments"):
        (repo_root / d).mkdir(parents=True, exist_ok=True)
    print("[DIR] Output directories OK")

    # --- Mount Google Drive (opsional, untuk backup model) ---
    if mount_drive:
        try:
            from google.colab import drive  # noqa: PLC0415
            drive.mount("/content/drive")
            print(f"[DRV] Drive mounted — hasil training bisa dibackup ke: {drive_dest}")
        except Exception as e:
            print(f"[DRV][WARN] Gagal mount Drive: {e}")

    print()
    print("[OK]  Bootstrap selesai — lanjutkan ke sel berikutnya.")
    print(SEP)
