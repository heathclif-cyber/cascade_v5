"""
colab_bootstrap.py — Siapkan environment Google Colab untuk Cascade v5.

Di notebook:
  from tools.colab_bootstrap import setup_colab
  ROOT = setup_colab(mount_drive=True)  # opsional persist ke Drive
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def setup_jupyter(
    repo_root: Path | None = None,
    mount_drive: bool = False,
    install_deps: bool = True,
    quick_coins: bool = True,
    force_colab_lgbm_cpu: bool | None = None,
) -> Path:
    """Alias: Colab pakai LGBM CPU; lokal biarkan config default (GPU OpenCL jika ada)."""
    in_colab = "google.colab" in sys.modules
    if force_colab_lgbm_cpu is None:
        force_colab_lgbm_cpu = in_colab
    root = setup_colab(
        repo_root=repo_root,
        mount_drive=mount_drive,
        install_deps=install_deps,
        quick_coins=quick_coins,
    ) if force_colab_lgbm_cpu else _setup_local(
        repo_root, mount_drive, install_deps, quick_coins
    )
    return root


def _setup_local(
    repo_root: Path | None,
    mount_drive: bool,
    install_deps: bool,
    quick_coins: bool,
) -> Path:
    if repo_root is None:
        repo_root = Path.cwd()
    repo_root = repo_root.resolve()
    os.chdir(repo_root)
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    if install_deps:
        req = repo_root / "requirements-colab.txt"
        if req.exists():
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-q", "-r", str(req)]
            )
    import torch
    from core.utils import get_lstm_device
    from config import COLAB_QUICK_COINS

    print(f"Repo root : {repo_root}")
    print(f"LSTM device: {get_lstm_device()}")
    print(f"CUDA avail: {torch.cuda.is_available()}")
    if quick_coins:
        print(f"Pilot coins: {COLAB_QUICK_COINS}")
    return repo_root


def setup_colab(
    repo_root: Path | None = None,
    mount_drive: bool = False,
    install_deps: bool = True,
    quick_coins: bool = True,
) -> Path:
    """
    - cd ke repo root
    - CASCADE_COLAB=1 + patch LGBM CPU
    - opsional: mount Google Drive
    - opsional: pip install requirements-colab.txt
    """
    if repo_root is None:
        repo_root = Path.cwd()
    repo_root = repo_root.resolve()
    os.chdir(repo_root)
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    os.environ["CASCADE_COLAB"] = "1"

    if mount_drive:
        try:
            from google.colab import drive  # type: ignore

            drive.mount("/content/drive", force_remount=False)
            print("Google Drive mounted at /content/drive")
        except ImportError:
            print("google.colab tidak tersedia — skip mount Drive")

    if install_deps:
        req = repo_root / "requirements-colab.txt"
        if req.exists():
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-q", "-r", str(req)]
            )

    from config import apply_colab_settings, COLAB_QUICK_COINS

    apply_colab_settings()

    # Patch Guardian (file 06_*.py tidak bisa di-import sebagai nama modul biasa)
    import importlib.util

    g06_path = repo_root / "pipeline" / "06_train_guardian.py"
    if g06_path.exists():
        spec = importlib.util.spec_from_file_location("cascade_guardian_train", g06_path)
        if spec and spec.loader:
            g06 = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(g06)
            g06.GUARDIAN_LGBM_PARAMS["device_type"] = "cpu"
            g06.GUARDIAN_LGBM_PARAMS.pop("gpu_platform_id", None)
            g06.GUARDIAN_LGBM_PARAMS.pop("gpu_device_id", None)

    import torch
    from core.utils import get_lstm_device

    print(f"Repo root : {repo_root}")
    print(f"LGBM      : CPU (Colab)")
    print(f"LSTM device: {get_lstm_device()}")
    print(f"CUDA avail: {torch.cuda.is_available()}")
    if quick_coins:
        print(f"Pilot coins (disarankan): {COLAB_QUICK_COINS}")
        print("  Gunakan: --coins SOLUSDT ETHUSDT BNBUSDT")

    return repo_root


def run(cmd: str, cwd: Path | None = None) -> int:
    """Jalankan perintah pipeline dari notebook."""
    print(f"$ {cmd}")
    return subprocess.call(cmd, shell=True, cwd=str(cwd or Path.cwd()))