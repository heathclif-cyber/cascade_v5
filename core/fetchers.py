"""
core/fetchers.py — Fetch semua data dari Binance + Macro sources
Gabungan dari futures_fetcher.py dan macro_fetcher.py

Fungsi utama:
  fetch_klines()        — OHLCV multi-timeframe
  fetch_funding_rate()  — Funding rate 8h
  fetch_btc_dominance() — BTC dominance (CoinGecko / proxy)
  fetch_fear_greed()    — Fear & Greed Index (Alternative.me)
  fetch_coin()          — Fetch semua data untuk satu koin
"""

import time
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from core.binance_client import BinanceClient
from core.utils import (
    setup_logger, save_df, load_df, get_filepath,
    mark_done, is_done, make_key,
    to_ms, from_ms, chunk_time_range, interval_to_ms,
    validate_ohlcv, print_summary,
)

logger = setup_logger("fetchers")


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _safe_float(val, default=None):
    try:
        if val is None or val == "":
            return default
        return float(val)
    except (ValueError, TypeError):
        return default


# ─── KLINE / OHLCV ───────────────────────────────────────────────────────────

def _parse_klines(raw: list) -> pd.DataFrame:
    cols = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades",
        "taker_buy_volume", "taker_buy_quote_volume", "_ignore"
    ]
    df = pd.DataFrame(raw, columns=cols)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.set_index("open_time")
    df.index.name = "timestamp"
    float_cols = ["open", "high", "low", "close", "volume",
                  "quote_volume", "taker_buy_volume", "taker_buy_quote_volume"]
    df[float_cols] = df[float_cols].astype(float)
    df["trades"]   = df["trades"].astype(int)
    df = df.drop(columns=["close_time", "_ignore"])
    return df


def fetch_klines(
    client: BinanceClient,
    symbol: str,
    interval: str,
    start: datetime,
    end: datetime,
    progress: dict = None,
    kline_limit: int = 1500,
    raw_dir: "Path | None" = None,
) -> Optional[pd.DataFrame]:
    """Fetch OHLCV data untuk satu koin satu interval. Resume-capable.
    raw_dir: override direktori simpan (default: data/raw/).
    """
    key = make_key("klines", symbol, interval)
    if progress and is_done(progress, key):
        logger.info(f"[{symbol}] {interval} klines sudah ada, skip.")
        filepath = get_filepath("klines", symbol, interval, base_dir=raw_dir)
        return load_df(filepath, logger)

    start_ms = to_ms(start)
    end_ms   = to_ms(end)
    logger.info(f"[{symbol}] Fetching {interval} klines {start.date()} -> {end.date()}")

    interval_ms  = interval_to_ms(interval)
    total_bars   = (end_ms - start_ms) // interval_ms
    total_chunks = (total_bars // kline_limit) + 1

    all_frames = []
    chunk_num  = 0
    for chunk_start, chunk_end in chunk_time_range(start_ms, end_ms, interval, kline_limit):
        chunk_num += 1
        raw = client.get_klines(
            symbol=symbol, interval=interval,
            start_time_ms=chunk_start,
            end_time_ms=chunk_end - 1,
            limit=kline_limit,
        )
        if not raw:
            logger.warning(
                f"[{symbol}] {interval} chunk {chunk_num}/{total_chunks}: "
                f"tidak ada data ({from_ms(chunk_start).date()} -> {from_ms(chunk_end).date()})"
            )
            continue
        df_chunk = _parse_klines(raw)
        all_frames.append(df_chunk)
        if chunk_num % 20 == 0 or chunk_num == total_chunks:
            pct = chunk_num / total_chunks * 100
            logger.info(
                f"[{symbol}] {interval}: {chunk_num}/{total_chunks} chunks "
                f"({pct:.0f}%)  last={df_chunk.index[-1].date()}"
            )

    if not all_frames:
        logger.error(f"[{symbol}] {interval}: tidak ada data sama sekali!")
        return None

    df = pd.concat(all_frames)
    df = df[~df.index.duplicated(keep="first")].sort_index()
    validate_ohlcv(df, symbol, interval, logger)

    filepath = get_filepath("klines", symbol, interval, base_dir=raw_dir)
    if save_df(df, filepath, logger):
        logger.info(f"[{symbol}] {interval}: {len(df):,} candle disimpan -> {filepath}")
        if progress is not None:
            mark_done(progress, key)

    return df


# ─── FUNDING RATE ─────────────────────────────────────────────────────────────

def _parse_funding_rate(raw: list) -> pd.DataFrame:
    records = []
    for item in raw:
        ts = int(item["fundingTime"])
        records.append({
            "timestamp":    datetime.fromtimestamp(ts / 1000, tz=timezone.utc),
            "funding_rate": _safe_float(item.get("fundingRate"), 0.0),
            "mark_price":   _safe_float(item.get("markPrice")),
        })
    df = pd.DataFrame(records).set_index("timestamp")
    df.index = pd.DatetimeIndex(df.index, tz=timezone.utc)
    return df.sort_index()


def fetch_funding_rate(
    client: BinanceClient,
    symbol: str,
    start: datetime,
    end: datetime,
    progress: dict = None,
    funding_limit: int = 1000,
    raw_dir: "Path | None" = None,
) -> Optional[pd.DataFrame]:
    """Fetch Funding Rate history (setiap 8 jam)."""
    key = make_key("funding_rate", symbol)
    if progress and is_done(progress, key):
        logger.info(f"[{symbol}] Funding rate sudah ada, skip.")
        return load_df(get_filepath("funding_rate", symbol, base_dir=raw_dir), logger)

    logger.info(f"[{symbol}] Fetching Funding Rate...")

    start_ms = to_ms(start)
    end_ms   = to_ms(end)
    step_per_chunk = 8 * 3_600_000 * funding_limit

    all_frames = []
    current = start_ms

    while current < end_ms:
        chunk_end = min(current + step_per_chunk, end_ms)
        raw = client.get_funding_rate(
            symbol=symbol,
            start_time_ms=current,
            end_time_ms=chunk_end,
            limit=funding_limit,
        )
        if raw:
            df_chunk = _parse_funding_rate(raw)
            all_frames.append(df_chunk)
            if not df_chunk.empty:
                last_ts = int(df_chunk.index[-1].timestamp() * 1000)
                current = last_ts + 8 * 3_600_000
            else:
                current = chunk_end
        else:
            current = chunk_end

    if not all_frames:
        logger.warning(f"[{symbol}] Funding Rate: tidak ada data")
        return None

    df = pd.concat(all_frames)
    df = df[~df.index.duplicated(keep="first")].sort_index()

    filepath = get_filepath("funding_rate", symbol, base_dir=raw_dir)
    if save_df(df, filepath, logger):
        logger.info(f"[{symbol}] Funding Rate: {len(df):,} records disimpan -> {filepath}")
        if progress is not None:
            mark_done(progress, key)

    return df


# ─── MACRO: BTC DOMINANCE ─────────────────────────────────────────────────────

def _fetch_btc_dom_proxy(start: datetime, end: datetime) -> Optional[pd.DataFrame]:
    """Proxy BTC dominance dari Binance BTCUSDT daily kline + approximate values."""
    import requests as req
    logger.info("Menggunakan proxy BTC dominance dari Binance BTCUSDT daily kline...")

    session  = req.Session()
    start_ms = to_ms(start)
    end_ms   = to_ms(end)
    records  = []
    current  = start_ms

    while current < end_ms:
        try:
            resp = session.get(
                "https://data-api.binance.vision/api/v3/klines",
                params={"symbol": "BTCUSDT", "interval": "1d",
                        "startTime": current, "limit": 1000},
                timeout=30,
            )
            resp.raise_for_status()
            raw = resp.json()
            if not raw:
                break
            for k in raw:
                ts = int(k[0])
                dt = datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
                if dt > end:
                    break
                year = dt.year
                dom  = {2020: 63.0, 2021: 45.0, 2022: 43.0, 2023: 47.0, 2024: 52.0, 2025: 58.0, 2026: 58.0}.get(year, 55.0)
                records.append({
                    "timestamp":         dt,
                    "btc_close":         float(k[4]),
                    "btc_dominance_pct": dom,
                })
            last_ts = int(raw[-1][0])
            current = last_ts + 86_400_000
            time.sleep(0.1)
        except Exception as e:
            logger.error(f"Binance BTC proxy error: {e}")
            break

    if not records:
        return None

    df = pd.DataFrame(records).set_index("timestamp")
    df.index = pd.DatetimeIndex(df.index, tz=timezone.utc)
    return df.sort_index()


def fetch_btc_dominance(
    start: datetime,
    end: datetime,
    progress: dict = None,
    coingecko_url: str = "https://api.coingecko.com/api/v3",
    sleep_coingecko: float = 2.0,
) -> Optional[pd.DataFrame]:
    """Fetch BTC dominance dari CoinGecko. Fallback ke proxy Binance jika gagal."""
    import requests

    key = "macro_btc_dominance"
    if progress and is_done(progress, key):
        logger.info("BTC Dominance sudah di-fetch sebelumnya, skip.")
        return load_df(get_filepath("macro_btc_dom", ""), logger)

    logger.info("Fetching BTC Dominance dari CoinGecko...")
    session = requests.Session()
    session.headers.update({"Accept": "application/json"})

    df = None
    try:
        time.sleep(sleep_coingecko)
        resp = session.get(
            f"{coingecko_url}/coins/bitcoin/market_chart",
            params={"vs_currency": "usd", "days": "max", "interval": "daily"},
            timeout=60,
        )
        if resp.status_code in (401, 403):
            logger.warning(f"CoinGecko return status {resp.status_code} — menggunakan proxy Binance")
            df = _fetch_btc_dom_proxy(start, end)
        elif resp.status_code == 429:
            time.sleep(60)
            resp = session.get(
                f"{coingecko_url}/coins/bitcoin/market_chart",
                params={"vs_currency": "usd", "days": "max", "interval": "daily"},
                timeout=60,
            )
        
        if df is None:
            resp.raise_for_status()
            data     = resp.json()
            mcap_data = data.get("market_caps", [])

            records = []
            for ts_ms, btc_mc in mcap_data:
                dt = datetime.fromtimestamp(int(ts_ms) / 1000, tz=timezone.utc)
                if start <= dt <= end:
                    year = dt.year
                    dom  = {2020: 63.0, 2021: 45.0, 2022: 43.0, 2023: 47.0, 2024: 52.0, 2025: 58.0, 2026: 58.0}.get(year, 55.0)
                    records.append({"timestamp": dt, "btc_market_cap_usd": btc_mc,
                                     "btc_dominance_pct": dom})

            if not records:
                df = _fetch_btc_dom_proxy(start, end)
            else:
                df = pd.DataFrame(records).drop_duplicates(subset=["timestamp"])
                df = df.sort_values("timestamp").set_index("timestamp")
                df.index = pd.DatetimeIndex(df.index, tz=timezone.utc)
                df = df.resample("1D").last().ffill()

    except Exception as e:
        logger.warning(f"CoinGecko error: {e} — menggunakan proxy Binance")
        df = _fetch_btc_dom_proxy(start, end)

    if df is not None:
        filepath = get_filepath("macro_btc_dom", "")
        if save_df(df, filepath, logger):
            logger.info(f"BTC Dominance disimpan: {len(df)} hari → {filepath}")
            if progress is not None:
                mark_done(progress, key)
        return df
    return None



# ─── MACRO: FEAR & GREED ──────────────────────────────────────────────────────

def fetch_fear_greed(
    start: datetime,
    end: datetime,
    progress: dict = None,
    fg_url: str = "https://api.alternative.me/fng/",
) -> Optional[pd.DataFrame]:
    """Fetch Fear & Greed Index dari Alternative.me. Gratis tanpa auth."""
    import requests

    key = "macro_fear_greed"
    if progress and is_done(progress, key):
        logger.info("Fear & Greed sudah di-fetch sebelumnya, skip.")
        return load_df(get_filepath("macro_fear_greed", ""), logger)

    logger.info("Fetching Fear & Greed Index dari alternative.me...")
    days_needed = (end - start).days + 10
    session     = requests.Session()

    try:
        time.sleep(1)
        resp = session.get(
            fg_url,
            params={"limit": 0, "format": "json"},  # limit=0 = semua data historis (since 2018)
            timeout=30,
        )
        resp.raise_for_status()
        data    = resp.json()
        entries = data.get("data", [])
        if not entries:
            return None

        records = []
        for entry in entries:
            ts  = int(entry["timestamp"])
            dt  = datetime.fromtimestamp(ts, tz=timezone.utc)
            records.append({
                "timestamp":  dt,
                "fear_greed": int(entry["value"]),
                "fg_class":   entry.get("value_classification", ""),
            })

        df = pd.DataFrame(records).sort_values("timestamp")
        df = df.set_index("timestamp")
        df.index = pd.DatetimeIndex(df.index, tz=timezone.utc)
        df = df[(df.index >= start) & (df.index <= end)]
        df = df.resample("1D").last()

        filepath = get_filepath("macro_fear_greed", "")
        if save_df(df, filepath, logger):
            logger.info(f"Fear & Greed disimpan: {len(df)} hari → {filepath}")
            if progress is not None:
                mark_done(progress, key)
        return df

    except Exception as e:
        logger.error(f"Error fetching Fear & Greed: {e}")
        return None


# ─── FETCH SEMUA MACRO ────────────────────────────────────────────────────────

def fetch_all_macro(
    start: datetime,
    end: datetime,
    progress: dict = None,
) -> dict:
    """Fetch BTC dominance dan Fear & Greed."""
    logger.info("=" * 55)
    logger.info("FETCH MACRO DATA")
    logger.info("=" * 55)

    results = {}

    fg_df = fetch_fear_greed(start, end, progress=progress)
    if fg_df is not None:
        results["fear_greed"] = fg_df
        logger.info(f"Fear & Greed: {len(fg_df)} hari OK")
    else:
        logger.warning("Fear & Greed: GAGAL")

    btc_df = fetch_btc_dominance(start, end, progress=progress)
    if btc_df is not None:
        results["btc_dominance"] = btc_df
        logger.info(f"BTC Dominance: {len(btc_df)} hari OK")
    else:
        logger.warning("BTC Dominance: GAGAL")

    return results


# ─── OPEN INTEREST ────────────────────────────────────────────────────────────

def _parse_open_interest(raw: list) -> pd.DataFrame:
    records = []
    for item in raw:
        ts = int(item["timestamp"])
        records.append({
            "timestamp":         datetime.fromtimestamp(ts / 1000, tz=timezone.utc),
            "open_interest":     _safe_float(item.get("sumOpenInterest"), 0.0),
            "open_interest_usd": _safe_float(item.get("sumOpenInterestValue"), 0.0),
        })
    df = pd.DataFrame(records).set_index("timestamp")
    df.index = pd.DatetimeIndex(df.index, tz=timezone.utc)
    return df.sort_index()


def fetch_open_interest_hist(
    client: BinanceClient,
    symbol: str,
    start: datetime,
    end: datetime,
    progress: dict = None,
    limit: int = 500,
    raw_dir: "Path | None" = None,
) -> Optional[pd.DataFrame]:
    """Fetch Open Interest history (hourly, up to last 500 hours/limit)."""
    key = make_key("open_interest", symbol)
    if progress and is_done(progress, key):
        logger.info(f"[{symbol}] Open Interest sudah ada, skip.")
        return load_df(get_filepath("open_interest", symbol, base_dir=raw_dir), logger)

    logger.info(f"[{symbol}] Fetching Open Interest...")

    start_ms = to_ms(start)
    end_ms   = to_ms(end)
    step_per_chunk = 3600000 * limit

    all_frames = []
    current = start_ms

    while current < end_ms:
        chunk_end = min(current + step_per_chunk, end_ms)
        try:
            raw = client.get_open_interest_hist(
                symbol=symbol,
                period="1h",
                start_time_ms=current,
                end_time_ms=chunk_end - 1,
                limit=limit,
            )
            if raw:
                df_chunk = _parse_open_interest(raw)
                all_frames.append(df_chunk)
                if not df_chunk.empty:
                    last_ts = int(df_chunk.index[-1].timestamp() * 1000)
                    current = last_ts + 3600000
                else:
                    current = chunk_end
            else:
                current = chunk_end
        except Exception as e:
            logger.warning(f"[{symbol}] Gagal fetch Open Interest chunk: {e}")
            current = chunk_end

    if not all_frames:
        logger.warning(f"[{symbol}] Open Interest: tidak ada data (kemungkinan di-block atau limit waktu terlampaui)")
        return None

    df = pd.concat(all_frames)
    df = df[~df.index.duplicated(keep="first")].sort_index()

    filepath = get_filepath("open_interest", symbol, base_dir=raw_dir)
    if save_df(df, filepath, logger):
        logger.info(f"[{symbol}] Open Interest: {len(df):,} records disimpan -> {filepath}")
        if progress is not None:
            mark_done(progress, key)

    return df


# ─── GLOBAL LONG/SHORT RATIO ──────────────────────────────────────────────────

def _parse_long_short_ratio(raw: list) -> pd.DataFrame:
    records = []
    for item in raw:
        ts = int(item["timestamp"])
        records.append({
            "timestamp":        datetime.fromtimestamp(ts / 1000, tz=timezone.utc),
            "long_short_ratio": _safe_float(item.get("longShortRatio"), 1.0),
            "long_account":     _safe_float(item.get("longAccount"), 0.5),
            "short_account":    _safe_float(item.get("shortAccount"), 0.5),
        })
    df = pd.DataFrame(records).set_index("timestamp")
    df.index = pd.DatetimeIndex(df.index, tz=timezone.utc)
    return df.sort_index()


def fetch_long_short_ratio_hist(
    client: BinanceClient,
    symbol: str,
    start: datetime,
    end: datetime,
    progress: dict = None,
    limit: int = 500,
    raw_dir: "Path | None" = None,
) -> Optional[pd.DataFrame]:
    """Fetch global Long/Short Account Ratio history (hourly, up to last 500 hours/limit)."""
    key = make_key("long_short_ratio", symbol)
    if progress and is_done(progress, key):
        logger.info(f"[{symbol}] Long/Short Ratio sudah ada, skip.")
        return load_df(get_filepath("long_short_ratio", symbol, base_dir=raw_dir), logger)

    logger.info(f"[{symbol}] Fetching Long/Short Ratio...")

    start_ms = to_ms(start)
    end_ms   = to_ms(end)
    step_per_chunk = 3600000 * limit

    all_frames = []
    current = start_ms

    while current < end_ms:
        chunk_end = min(current + step_per_chunk, end_ms)
        try:
            raw = client.get_global_long_short_ratio(
                symbol=symbol,
                period="1h",
                start_time_ms=current,
                end_time_ms=chunk_end - 1,
                limit=limit,
            )
            if raw:
                df_chunk = _parse_long_short_ratio(raw)
                all_frames.append(df_chunk)
                if not df_chunk.empty:
                    last_ts = int(df_chunk.index[-1].timestamp() * 1000)
                    current = last_ts + 3600000
                else:
                    current = chunk_end
            else:
                current = chunk_end
        except Exception as e:
            logger.warning(f"[{symbol}] Gagal fetch Long/Short Ratio chunk: {e}")
            current = chunk_end

    if not all_frames:
        logger.warning(f"[{symbol}] Long/Short Ratio: tidak ada data (kemungkinan di-block atau limit waktu terlampaui)")
        return None

    df = pd.concat(all_frames)
    df = df[~df.index.duplicated(keep="first")].sort_index()

    filepath = get_filepath("long_short_ratio", symbol, base_dir=raw_dir)
    if save_df(df, filepath, logger):
        logger.info(f"[{symbol}] Long/Short Ratio: {len(df):,} records disimpan -> {filepath}")
        if progress is not None:
            mark_done(progress, key)

    return df


# ─── FETCH SATU KOIN LENGKAP ──────────────────────────────────────────────────

def fetch_coin(
    client: BinanceClient,
    symbol: str,
    start: datetime,
    end: datetime,
    intervals: list = None,
    progress: dict = None,
    kline_limit: int = 1500,
    funding_limit: int = 1000,
    oi_limit: int = 500,
    long_short_limit: int = 500,
    raw_dir: "Path | None" = None,
) -> dict:
    """
    Fetch OHLCV semua interval + funding rate + open interest + long/short ratio untuk satu koin.
    raw_dir: override direktori simpan (default: data/raw/ untuk training, data/holdout/raw/ untuk OOS).
    OI dan Long/Short ratio ditarik dari REST API (terbatas pada histori jangka pendek, misal 500 jam / 30 hari).
    """
    if intervals is None:
        intervals = ["1h", "4h", "1d"]

    logger.info(f"\n{'='*55}")
    logger.info(f"  FETCHING: {symbol}")
    logger.info(f"  Periode: {start.date()} -> {end.date()}")
    if raw_dir:
        logger.info(f"  Output : {raw_dir}")
    logger.info(f"{'='*55}")

    results     = {}
    val_results = []

    for interval in intervals:
        df = fetch_klines(client, symbol, interval, start, end,
                          progress=progress, kline_limit=kline_limit, raw_dir=raw_dir)
        if df is not None:
            results[f"klines_{interval}"] = df
            val_results.append(validate_ohlcv(df, symbol, interval))
        else:
            logger.error(f"[{symbol}] {interval} klines GAGAL")

    df_fr = fetch_funding_rate(client, symbol, start, end,
                               progress=progress, funding_limit=funding_limit)
    if df_fr is not None:
        results["funding_rate"] = df_fr

    df_oi = fetch_open_interest_hist(client, symbol, start, end,
                                     progress=progress, limit=oi_limit)
    if df_oi is not None:
        results["open_interest"] = df_oi

    df_ls = fetch_long_short_ratio_hist(client, symbol, start, end,
                                        progress=progress, limit=long_short_limit)
    if df_ls is not None:
        results["long_short_ratio"] = df_ls

    if val_results:
        print_summary(symbol, val_results, logger)

    return results
