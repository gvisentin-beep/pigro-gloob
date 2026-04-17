from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import requests
import yfinance as yf

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

TWELVE_DATA_API_KEY = os.getenv("TWELVE_DATA_API_KEY", "").strip()
TWELVE_DATA_BASE_URL = "https://api.twelvedata.com/time_series"

MIN_VALID_ROWS = 100

ASSETS = {
    "ls80": {
        "symbol": "VNGA80.MI",
        "path": DATA_DIR / "ls80.csv",
        "source": "yahoo",
        "update_enabled": True,
        "force_refresh": True,
    },
    "gold": {
        "symbol": os.getenv("GOLD_TICKER", "GLD").strip(),
        "path": DATA_DIR / "gold.csv",
        "source": "twelve",
        "update_enabled": True,
    },
    "btc": {
        "symbol": os.getenv("BTC_TICKER", "BTC/EUR").strip(),
        "path": DATA_DIR / "btc.csv",
        "source": "twelve",
        "update_enabled": True,
    },
    "world": {
        "symbol": os.getenv("WORLD_TICKER", "URTH").strip(),
        "path": DATA_DIR / "world.csv",
        "source": "twelve",
        "update_enabled": True,
    },
    "mib": {
        "symbol": os.getenv("MIB_TICKER", "EWI").strip(),
        "path": DATA_DIR / "mib.csv",
        "source": "twelve",
        "update_enabled": True,
    },
    "sp500": {
        "symbol": os.getenv("SP500_TICKER", "SPY").strip(),
        "path": DATA_DIR / "sp500.csv",
        "source": "twelve",
        "update_enabled": True,
    },
}


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def log(msg: str) -> None:
    print(msg, flush=True)


def read_existing_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None

    df = pd.read_csv(path, sep=";", dtype=str, encoding="utf-8-sig")

    if "date" not in df.columns:
        df.columns = ["date", "close"]

    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    df["close"] = pd.to_numeric(df["close"].astype(str).str.replace(",", "."), errors="coerce")

    df = df.dropna().sort_values("date").drop_duplicates("date")

    return df if not df.empty else None


def write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    df = df.sort_values("date").drop_duplicates("date")

    df["date"] = df["date"].dt.strftime("%d/%m/%Y")

    df.to_csv(path, sep=";", index=False)


def fetch_series_twelve(symbol: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    if not TWELVE_DATA_API_KEY:
        return None, "API key mancante"

    try:
        r = requests.get(
            TWELVE_DATA_BASE_URL,
            params={
                "symbol": symbol,
                "interval": "1day",
                "outputsize": 5000,
                "apikey": TWELVE_DATA_API_KEY,
            },
            timeout=30,
        )

        data = r.json()
        values = data.get("values")

        if not values:
            return None, "no data"

        df = pd.DataFrame(values)
        df["date"] = pd.to_datetime(df["datetime"])
        df["close"] = pd.to_numeric(df["close"])

        return df[["date", "close"]], None

    except Exception as e:
        return None, str(e)


def fetch_series_yahoo(symbol: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period="max")

        if df.empty:
            return None, "no data"

        df = df.reset_index()

        df = df.rename(columns={"Date": "date", "Close": "close"})

        df["date"] = pd.to_datetime(df["date"])
        df["close"] = pd.to_numeric(df["close"])

        return df[["date", "close"]], None

    except Exception as e:
        return None, str(e)


def update_asset(name: str, cfg: dict) -> bool:
    symbol = cfg["symbol"]
    path = cfg["path"]
    source = cfg["source"]

    log(f"\n[{name.upper()}]")

    force = cfg.get("force_refresh", False)

    existing = None if force else read_existing_csv(path)

    if source == "yahoo":
        fresh, err = fetch_series_yahoo(symbol)
    else:
        fresh, err = fetch_series_twelve(symbol)

    if fresh is None:
        log(f"Errore: {err}")
        return False

    df = fresh if existing is None else pd.concat([existing, fresh])

    write_csv(path, df)

    log(f"OK aggiornato {len(df)} righe")
    return True


def main():
    log("START UPDATE")

    for name, cfg in ASSETS.items():
        update_asset(name, cfg)

    log("END UPDATE")


if __name__ == "__main__":
    main()
