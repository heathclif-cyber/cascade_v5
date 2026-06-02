"""
pipeline/02_clean.py — Data Cleaning & Split H1/H4 (Cascade v5)

Berbeda dari v4: output adalah DUA file terpisah per koin:
  data/processed/{symbol}_h1_clean.parquet  ← untuk LSTM features (H1 resolution)
  data/processed/{symbol}_h4_clean.parquet  ← untuk LGBM features (H4 resolution)

Anti look-ahead:
  - H4 di-shift(1) sebelum di-ffill ke H1 grid — bar H4 yang belum tutup tidak bocor ke H1
  - Macro (fear_greed, btc_dominance) di-ffill tapi tidak di-shift (harian, bukan intrabar)

Jalankan:
  python pipeline/02_clean.py --all
  python pipeline/02_clean.py --coins SOLUSDT ETHUSDT
  python pipeline/02_clean.py --all --holdout
"""

import argparse, json, sys, traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config import (
    TRAINING_COINS, ALL_COINS,
    RAW_DIR, PROC_DIR,
)
from core.utils import setup_logger, ensure_utc_index

logger = setup_logger("02_clean")

INTERVALS     = ["1h", "4h"]
INTERVAL_FREQ = {"1h": "1h", "4h": "4h"}
HOLDOUT_RAW   = ROOT / "data" / "holdout" / "raw"
HOLDOUT_PROC  = ROOT / "data" / "holdout" / "processed"


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _load(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
        return ensure_utc_index(df) if not df.empty else None
    except Exception as e:
        logger.warning(f"Gagal load {path.name}: {e}")
        return None


def _save(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df, preserve_index=True)
    pq.write_table(table, str(path), compression="snappy")


def fix_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    """Pastikan high = max(O,H,L,C) dan low = min(O,H,L,C)."""
    col_map = {c.lower(): c for c in df.columns}
    if not {"open", "high", "low", "close"}.issubset(col_map):
        return df
    ocols = [col_map[k] for k in ("open", "high", "low", "close")]
    mat   = df[ocols].values.astype(float)
    df    = df.copy()
    df[col_map["high"]] = np.nanmax(mat, axis=1)
    df[col_map["low"]]  = np.nanmin(mat, axis=1)
    return df


def detect_gaps(df: pd.DataFrame, freq: str) -> int:
    """Hitung jumlah gap dalam series (bar yang hilang)."""
    if len(df) < 2:
        return 0
    expected_ns = pd.tseries.frequencies.to_offset(freq).nanos
    diffs = df.index.to_series().diff().dropna()
    return int((diffs.dt.total_seconds() * 1e9 > expected_ns * 1.5).sum())


def ffill_to_index(df: pd.DataFrame, target_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Reindex df ke target_index dan ffill."""
    combined = df.reindex(df.index.union(target_index)).sort_index().ffill()
    return combined.reindex(target_index)


# ─── Main Clean Function ───────────────────────────────────────────────────────

def clean_symbol(symbol: str, raw_dir: Path, proc_dir: Path) -> bool:
    logger.info(f"[{symbol}] Cleaning...")

    # ── Load klines ───────────────────────────────────────────────────────────
    klines = {}
    for tf in INTERVALS:
        path = raw_dir / "klines" / symbol / f"{tf}_all.parquet"
        df   = _load(path)
        if df is None:
            logger.warning(f"  [{symbol}] {tf} klines tidak ditemukan: {path}")
        else:
            df = fix_ohlc(df)
            klines[tf] = df

    h1 = klines.get("1h")
    h4 = klines.get("4h")

    if h1 is None:
        logger.error(f"  [{symbol}] H1 klines tidak ada — skip")
        return False

    # ── Load auxiliary (funding rate, open interest) ──────────────────────────
    aux_paths = {
        "funding_rate":  raw_dir / "funding_rate"  / f"{symbol}_8h.parquet",
        "open_interest": raw_dir / "open_interest"  / f"{symbol}_1h.parquet",
    }
    aux = {}
    for name, path in aux_paths.items():
        df = _load(path)
        aux[name] = df
        status = f"{len(df):,} rows" if df is not None else "MISSING"
        logger.info(f"  [{symbol}] {name}: {status}")

    # ── Load macro ────────────────────────────────────────────────────────────
    macro_paths = {
        "btc_dominance":    raw_dir / "macro" / "btc_dominance.parquet",
        "fear_greed_index": raw_dir / "macro" / "fear_greed_index.parquet",
    }
    macro = {}
    for name, path in macro_paths.items():
        df = _load(path)
        if df is None:
            # Fallback ke training macro jika holdout
            fallback = RAW_DIR / "macro" / path.name
            df = _load(fallback)
        macro[name] = df

    # ── BTC close untuk Relative Strength ────────────────────────────────────
    btc_close = None
    if symbol != "BTCUSDT":
        btc_clean = proc_dir / "BTCUSDT_h1_clean.parquet"
        if btc_clean.exists():
            try:
                btc_df    = pd.read_parquet(btc_clean)
                btc_close = btc_df.get("close", btc_df.get("1h_close"))
                if btc_close is not None:
                    btc_close = btc_close.rename("btc_close")
            except Exception as e:
                logger.warning(f"  [{symbol}] Gagal load BTCUSDT: {e}")

    # ══════════════════════════════════════════════════════════════════════════
    # BUILD H1 MASTER (untuk LSTM features)
    # ══════════════════════════════════════════════════════════════════════════
    h1_master = h1.copy()

    # Join H4 dengan shift(1) ke H1 — cegah look-ahead dari H4 bar yang belum tutup
    if h4 is not None:
        h4_shifted = h4.shift(1)   # <── kunci anti look-ahead
        h4_renamed = h4_shifted.add_prefix("h4_")
        h4_on_h1   = ffill_to_index(h4_renamed, h1_master.index)
        h1_master  = h1_master.join(h4_on_h1, how="left")

    # Join auxiliary ke H1
    for name, df_aux in aux.items():
        if df_aux is None:
            continue
        df_renamed = df_aux.add_prefix(f"{name}_")
        df_on_h1   = ffill_to_index(df_renamed, h1_master.index)
        h1_master  = h1_master.join(df_on_h1, how="left")

    # Join macro ke H1 (daily → ffill, tidak shift)
    for name, df_macro in macro.items():
        if df_macro is None:
            continue
        col_name = "btc_dominance" if "dominance" in name else "fear_greed"
        # Ambil kolom value jika ada
        value_col = next((c for c in df_macro.columns if "value" in c.lower()
                         or "dominance" in c.lower() or "greed" in c.lower()), None)
        if value_col:
            series = df_macro[value_col].rename(col_name)
            series_h1 = ffill_to_index(series.to_frame(), h1_master.index)[col_name]
            h1_master[col_name] = series_h1

    # BTC close
    if btc_close is not None:
        h1_master["btc_close"] = ffill_to_index(btc_close.to_frame(), h1_master.index)["btc_close"]

    # NaN fill untuk kolom non-kritikal
    for col in ["btc_dominance", "fear_greed", "btc_close"]:
        if col in h1_master.columns:
            h1_master[col] = h1_master[col].ffill().fillna(0)

    # Drop baris di mana close NaN
    h1_master = h1_master.dropna(subset=["close"])

    # ══════════════════════════════════════════════════════════════════════════
    # BUILD H4 MASTER (untuk LGBM Tabular Expert)
    # ══════════════════════════════════════════════════════════════════════════
    if h4 is not None:
        h4_master = h4.copy()

        # Join auxiliary ke H4 (ffill dari sumber yg lebih jarang)
        for name, df_aux in aux.items():
            if df_aux is None:
                continue
            df_renamed = df_aux.add_prefix(f"{name}_")
            df_on_h4   = ffill_to_index(df_renamed, h4_master.index)
            h4_master  = h4_master.join(df_on_h4, how="left")

        # Join macro ke H4
        for name, df_macro in macro.items():
            if df_macro is None:
                continue
            col_name  = "btc_dominance" if "dominance" in name else "fear_greed"
            value_col = next((c for c in df_macro.columns if "value" in c.lower()
                             or "dominance" in c.lower() or "greed" in c.lower()), None)
            if value_col:
                series    = df_macro[value_col].rename(col_name)
                series_h4 = ffill_to_index(series.to_frame(), h4_master.index)[col_name]
                h4_master[col_name] = series_h4

        h4_master = h4_master.dropna(subset=["close"])
    else:
        h4_master = None

    # ── Save ──────────────────────────────────────────────────────────────────
    out_h1 = proc_dir / f"{symbol}_h1_clean.parquet"
    _save(h1_master, out_h1)

    h4_info = ""
    if h4_master is not None:
        out_h4 = proc_dir / f"{symbol}_h4_clean.parquet"
        _save(h4_master, out_h4)
        h4_info = f" | H4={len(h4_master):,} bars"

    n_gaps_h1 = detect_gaps(h1_master, "1h")
    logger.info(
        f"[{symbol}] OK — H1={len(h1_master):,} bars{h4_info} | "
        f"H1 gaps={n_gaps_h1} | "
        f"cols_h1={len(h1_master.columns)}"
    )
    return True


def main():
    parser = argparse.ArgumentParser(description="Clean & split H1/H4 data (v5)")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--all",   action="store_true")
    group.add_argument("--coins", nargs="+")
    parser.add_argument("--holdout", action="store_true",
                        help="Clean holdout data dari data/holdout/raw/")
    args = parser.parse_args()

    if args.all:
        coins = ALL_COINS
    elif args.coins:
        coins = [c.upper() for c in args.coins]
    else:
        coins = TRAINING_COINS

    if args.holdout:
        raw_dir  = HOLDOUT_RAW
        proc_dir = HOLDOUT_PROC
        logger.info("Mode: HOLDOUT")
    else:
        raw_dir  = RAW_DIR
        proc_dir = PROC_DIR
        logger.info("Mode: TRAINING")

    proc_dir.mkdir(parents=True, exist_ok=True)

    # BTCUSDT harus diclean pertama (dipakai sebagai referensi BTC close)
    if "BTCUSDT" in coins:
        coins = ["BTCUSDT"] + [c for c in coins if c != "BTCUSDT"]

    success = 0
    for symbol in coins:
        try:
            if clean_symbol(symbol, raw_dir, proc_dir):
                success += 1
        except Exception as e:
            logger.error(f"[{symbol}] Error: {e}")
            logger.error(traceback.format_exc())

    logger.info(f"\nSelesai: {success}/{len(coins)} koin OK")
    if success < len(coins):
        logger.warning(f"{len(coins) - success} koin gagal — cek log di atas")


if __name__ == "__main__":
    main()
