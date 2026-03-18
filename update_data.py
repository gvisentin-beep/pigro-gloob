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

ASSETS = {
    "ls80": {"symbol": os.getenv("LS80_TICKER", "VNGA80.MI").strip(), "path": DATA_DIR / "ls80.csv"},
    "gold": {"symbol": os.getenv("GOLD_TICKER", "GLD").strip(), "path": DATA_DIR / "gold.csv"},
    "btc": {"symbol": os.getenv("BTC_TICKER", "BTC/EUR").strip(), "path": DATA_DIR / "btc.csv"},
    "world": {"symbol": os.getenv("WORLD_TICKER", "URTH").strip(), "path": DATA_DIR / "world.csv"},
}


def log(msg: str):
    print(msg, flush=True)


# ========================
# CSV
# ========================

def read_existing_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None

    df = pd.read_csv(path, sep=";", dtype=str)
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    df["close"] = df["close"].str.replace(",", ".").astype(float)
    df = df.dropna().sort_values("date").drop_duplicates("date")

    return df


def write_csv(path: Path, df: pd.DataFrame):
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy().sort_values("date").drop_duplicates("date")
    out["date"] = out["date"].dt.strftime("%d/%m/%Y")
    out.to_csv(path, sep=";", index=False)


# ========================
# YAHOO (LS80)
# ========================

def fetch_yahoo(symbol: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    try:
        df = yf.download(symbol, period="max", interval="1d", progress=False)

        if df is None or df.empty:
            return None, "Yahoo vuoto"

        df = df.reset_index()
        df = df[["Date", "Close"]]
        df.columns = ["date", "close"]

        df["date"] = pd.to_datetime(df["date"])
        df["close"] = df["close"].astype(float)

        return df, None

    except Exception as e:
        return None, str(e)


# ========================
# TWELVE DATA
# ========================

def fetch_twelve(symbol: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    if not TWELVE_DATA_API_KEY:
        return None, "API key mancante"

    params = {
        "symbol": symbol,
        "interval": "1day",
        "outputsize": 5000,
        "apikey": TWELVE_DATA_API_KEY,
    }

    try:
        r = requests.get(TWELVE_DATA_BASE_URL, params=params)
        data = r.json()

        if "values" not in data:
            return None, str(data)

        df = pd.DataFrame(data["values"])
        df["date"] = pd.to_datetime(df["datetime"])
        df["close"] = df["close"].astype(float)

        return df[["date", "close"]], None

    except Exception as e:
        return None, str(e)


# ========================
# UPDATE
# ========================

def update_asset(name, symbol, path):
    log(f"\n{name.upper()}")

    old = read_existing_csv(path)

    # 🔥 QUI LA DIFFERENZA
    if name == "ls80":
        new, err = fetch_yahoo(symbol)
    else:
        new, err = fetch_twelve(symbol)

    if new is None:
        log(f"Errore: {err}")
        return

    merged = pd.concat([old, new]) if old is not None else new
    merged = merged.drop_duplicates("date").sort_values("date")

    write_csv(path, merged)

    log(f"OK - ultima data: {merged['date'].iloc[-1]}")


def main():
    for name, cfg in ASSETS.items():
        update_asset(name, cfg["symbol"], cfg["path"])


if __name__ == "__main__":
    main()
