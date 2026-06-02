"""
pipeline/01_fetch.py — Fetch Data dari Binance (Cascade v5)

Mengambil klines H1 + H4, funding rate, open interest, dan macro data.
Tidak ada D1 (tidak dipakai di v5).

Jalankan:
  python pipeline/01_fetch.py --all              # fetch semua koin (training)
  python pipeline/01_fetch.py --coins SOLUSDT    # koin spesifik
  python pipeline/01_fetch.py --all --holdout    # fetch holdout (Mei 2025 – Apr 2026)
  python pipeline/01_fetch.py --all --reset      # reset progress, fetch ulang

Output:
  data/raw/klines/{symbol}/{1h,4h}_all.parquet
  data/raw/funding_rate/{symbol}_8h.parquet
  data/raw/open_interest/{symbol}_1h.parquet
  data/raw/macro/btc_dominance.parquet
  data/raw/macro/fear_greed_index.parquet
"""

import argparse, json, sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config import (
    TRAINING_COINS, ALL_COINS,
    TRAIN_START, TRAIN_END, TRAIN_CUTOFF_DATE,
    BINANCE_BASE_URL, SLEEP_BETWEEN_REQUESTS,
    SLEEP_ON_RATE_LIMIT, MAX_RETRIES, RETRY_BACKOFF_BASE,
    KLINE_INTERVALS, KLINE_LIMIT, FUNDING_LIMIT,
    RAW_DIR,
)
from core.binance_client import BinanceClient
from core.fetchers import fetch_coin, fetch_all_macro
from core.utils import setup_logger

logger = setup_logger("01_fetch")

HOLDOUT_START = datetime(2025, 5, 1, tzinfo=timezone.utc)
HOLDOUT_END   = datetime(2026, 4, 1, tzinfo=timezone.utc)
HOLDOUT_RAW   = ROOT / "data" / "holdout" / "raw"
PROGRESS_FILE = RAW_DIR / ".fetch_progress.json"


def load_progress(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            pass
    return {}


def save_progress(progress: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(progress, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description="Fetch Binance data untuk Cascade v5")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--all",   action="store_true", help="Semua koin training")
    group.add_argument("--coins", nargs="+",           help="Koin spesifik")
    parser.add_argument("--holdout", action="store_true",
                        help="Fetch data holdout (Mei 2025 – Apr 2026) ke data/holdout/raw/")
    parser.add_argument("--reset", action="store_true", help="Reset progress, fetch ulang")
    return parser.parse_args()


def main():
    args = parse_args()

    # Tentukan coins, periode, dan output dir
    if args.all:
        coins = ALL_COINS
    elif args.coins:
        coins = [c.upper() for c in args.coins]
    else:
        coins = TRAINING_COINS

    if args.holdout:
        fetch_start = HOLDOUT_START
        fetch_end   = HOLDOUT_END
        output_dir  = HOLDOUT_RAW
        prog_file   = HOLDOUT_RAW / ".fetch_progress.json"
        mode_str    = "HOLDOUT"
    else:
        fetch_start = TRAIN_START
        fetch_end   = TRAIN_END
        output_dir  = RAW_DIR
        prog_file   = PROGRESS_FILE
        mode_str    = "TRAINING"

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info(f"  MODE    : {mode_str}")
    logger.info(f"  PERIODE : {fetch_start.date()} -> {fetch_end.date()}")
    logger.info(f"  KOIN    : {len(coins)} koin")
    logger.info(f"  OUTPUT  : {output_dir}")
    logger.info("=" * 60)

    # Progress
    if args.reset:
        progress = {}
        logger.info("Progress di-reset.")
    else:
        progress = load_progress(prog_file)
        done = sum(1 for v in progress.values() if v == "done")
        logger.info(f"Progress loaded: {done} koin sudah selesai.")

    # Init Binance client
    client = BinanceClient(
        base_url         = BINANCE_BASE_URL,
        sleep_between    = SLEEP_BETWEEN_REQUESTS,
        sleep_rate_limit = SLEEP_ON_RATE_LIMIT,
        max_retries      = MAX_RETRIES,
        backoff_base     = RETRY_BACKOFF_BASE,
    )
    if not client.test_connection():
        logger.error("Koneksi Binance gagal. Cek internet/VPN.")
        sys.exit(1)
    logger.info("Koneksi Binance OK.")

    # Fetch macro
    logger.info("\n--- Fetching macro data ---")
    try:
        fetch_all_macro(fetch_start, fetch_end, progress=progress)
        save_progress(progress, prog_file)
    except Exception as e:
        logger.warning(f"Macro fetch gagal: {e} — lanjut fetch koin")

    # Fetch per koin
    success, failed = [], []
    for i, symbol in enumerate(coins, 1):
        prog_key = f"{symbol}_{mode_str}"
        if progress.get(prog_key) == "done" and not args.reset:
            h1p = output_dir / "klines" / symbol / "1h_all.parquet"
            h4p = output_dir / "klines" / symbol / "4h_all.parquet"
            if h1p.exists() and h4p.exists() and h1p.stat().st_size > 500:
                logger.info(f"[{i}/{len(coins)}] {symbol} — sudah selesai, skip")
                success.append(symbol)
                continue
            logger.warning(
                f"[{i}/{len(coins)}] {symbol} — progress=done tapi file hilang, fetch ulang"
            )
            progress.pop(prog_key, None)

        logger.info(f"[{i}/{len(coins)}] Fetching {symbol}...")
        try:
            result = fetch_coin(
                client        = client,
                symbol        = symbol,
                start         = fetch_start,
                end           = fetch_end,
                intervals     = KLINE_INTERVALS,   # ["1h", "4h"]
                progress      = progress,
                kline_limit   = KLINE_LIMIT,
                funding_limit = FUNDING_LIMIT,
            )
            if result:
                progress[prog_key] = "done"
                save_progress(progress, prog_file)
                success.append(symbol)
            else:
                logger.warning(f"  [{symbol}] Fetch partial/gagal")
                failed.append(symbol)
        except Exception as e:
            logger.error(f"  [{symbol}] Error: {e}")
            failed.append(symbol)

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info(f"SELESAI: {len(success)}/{len(coins)} koin berhasil")
    if failed:
        logger.warning(f"GAGAL: {failed}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
