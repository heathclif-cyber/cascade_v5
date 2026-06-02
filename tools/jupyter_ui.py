"""
jupyter_ui.py — Output jelas di Jupyter / Colab (tabel status, banner langkah).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parent.parent


def in_notebook() -> bool:
    try:
        from IPython import get_ipython  # type: ignore
        ip = get_ipython()
        if ip is None:
            return False
        return "IPKernelApp" in str(type(ip))
    except Exception:
        return False


def enable_verbose_logging() -> None:
    """Pastikan log pipeline tampil di cell Jupyter."""
    import logging

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(fmt)
        root.addHandler(h)
    for name in ("01_fetch", "02_clean", "03_engineer", "04_train_lgbm",
                 "fetchers", "binance_client"):
        lg = logging.getLogger(name)
        lg.setLevel(logging.INFO)
        lg.propagate = True


def _kb(path: Path) -> str:
    if not path.exists():
        return "-"
    return f"{path.stat().st_size // 1024:,} KB"


def step_start(name: str, detail: str = "") -> float:
    print("\n" + "=" * 72)
    print(f">>> MULAI: {name}")
    if detail:
        print(f"    {detail}")
    print("=" * 72, flush=True)
    return time.time()


def step_ok(name: str, t0: float, extra: str = "") -> None:
    elapsed = time.time() - t0
    print("-" * 72)
    print(f"<<< SELESAI: {name} ({elapsed:.0f} detik)")
    if extra:
        print(f"    {extra}")
    print("-" * 72, flush=True)


def step_fail(name: str, t0: float, err: str) -> None:
    elapsed = time.time() - t0
    print("!" * 72)
    print(f"!!! GAGAL: {name} ({elapsed:.0f} detik)")
    print(f"    {err}")
    print("!" * 72, flush=True)


def status_table(coins: Iterable[str], root: Path | None = None) -> None:
    """Ringkasan file data — jalankan kapan saja untuk cek progress."""
    root = root or ROOT
    sys.path.insert(0, str(root))
    from config import RAW_DIR, PROC_DIR, LABEL_DIR

    coins = list(coins)
    print("\n" + "#" * 72)
    print("STATUS DATA")
    print(f"ROOT: {root.resolve()}")
    print("#" * 72)
    header = f"{'COIN':<14} {'1h raw':>10} {'4h raw':>10} {'h1 clean':>10} {'h4 clean':>10} {'lgbm':>10}"
    print(header)
    print("-" * len(header))

    ok = 0
    for sym in coins:
        r1 = RAW_DIR / "klines" / sym / "1h_all.parquet"
        r4 = RAW_DIR / "klines" / sym / "4h_all.parquet"
        p1 = PROC_DIR / f"{sym}_h1_clean.parquet"
        p4 = PROC_DIR / f"{sym}_h4_clean.parquet"
        lb = LABEL_DIR / f"{sym}_h4_lgbm.parquet"

        def mark(p: Path, min_kb: int = 5) -> str:
            if p.exists() and p.stat().st_size > min_kb * 1024:
                return "OK"
            return "MISS"

        row = [sym, mark(r1), mark(r4), mark(p1, 50), mark(p4, 50), mark(lb, 50)]
        if all(x == "OK" for x in row[1:]):
            ok += 1
        print(f"{row[0]:<14} {row[1]:>10} {row[2]:>10} {row[3]:>10} {row[4]:>10} {row[5]:>10}")

    print("-" * len(header))
    print(f"Siap train LGBM: {ok}/{len(coins)} koin (semua kolom OK)")
    print("#" * 72 + "\n", flush=True)

    if in_notebook():
        try:
            from IPython.display import HTML, display  # type: ignore
            rows_html = "".join(
                f"<tr><td>{sym}</td>"
                f"<td>{_kb(RAW_DIR / 'klines' / sym / '1h_all.parquet')}</td>"
                f"<td>{_kb(RAW_DIR / 'klines' / sym / '4h_all.parquet')}</td>"
                f"<td>{_kb(PROC_DIR / f'{sym}_h1_clean.parquet')}</td>"
                f"<td>{_kb(PROC_DIR / f'{sym}_h4_clean.parquet')}</td>"
                f"<td>{_kb(LABEL_DIR / f'{sym}_h4_lgbm.parquet')}</td></tr>"
                for sym in coins
            )
            display(HTML(
                "<h4>Cascade v5 — Status Data</h4>"
                "<table border='1' cellpadding='4' style='border-collapse:collapse'>"
                "<tr><th>Coin</th><th>1h raw</th><th>4h raw</th>"
                "<th>h1 clean</th><th>h4 clean</th><th>lgbm</th></tr>"
                + rows_html + "</table>"
            ))
        except Exception:
            pass