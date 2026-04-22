from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import yfinance as yf

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

MIN_VALID_ROWS = 50


ASSETS = {
    "ls80": {
        "symbol": "",
        "path": DATA_DIR / "ls80.csv",
        "source": "manual",
        "update_enabled": False,
    },
    "gold": {
        "symbol": "GLD",
        "path": DATA_DIR / "gold.csv",
        "source": "yahoo",
        "update_enabled": True,
    },
    "btc": {
        "symbol": "BTC-EUR",
        "path": DATA_DIR / "btc.csv",
        "source": "yahoo",
        "update_enabled": True,
    },
    "world": {
        "symbol": "URTH",
        "path": DATA_DIR / "world.csv",
        "source": "yahoo",
        "update_enabled": True,
    },
    "mib": {
        "symbol": "MSE.PA",  # EURO STOXX 50
        "path": DATA_DIR / "mib.csv",
        "source": "yahoo",
        "update_enabled": True,
    },
    "sp500": {
        "symbol": "^GSPC",
        "path": DATA_DIR / "sp500.csv",
        "source": "yahoo",
        "update_enabled": True,
    },
}


def log(msg: str):
    print(msg, flush=True)


def read_existing_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None

    try:
        df = pd.read_csv(path, sep=";")
        df.columns = ["date", "close"]

        df["date"] = pd.to_datetime(df["date"], dayfirst=True)
        df["close"] = pd.to_numeric(df["close"], errors="coerce")

        df = df.dropna().sort_values("date").drop_duplicates("date")

        return df
    except:
        return None


def write_csv(path: Path, df: pd.DataFrame):
    path.parent.mkdir(parents=True, exist_ok=True)

    df = df.sort_values("date").drop_duplicates("date")

    df["date"] = df["date"].dt.strftime("%d/%m/%Y")

    df.to_csv(path, sep=";", index=False)


def fetch_yahoo(symbol: str):
    try:
        df = yf.download(
            symbol,
            period="max",
            interval="1d",
            progress=False,
            threads=False,
        )

        if df is None or df.empty:
            return None, "Yahoo vuoto"

        # gestione MultiIndex
        if isinstance(df.columns, pd.MultiIndex):
            close = df[[col for col in df.columns if col[0] == "Close"]][df.columns[0]]
        else:
            if "Close" not in df.columns:
                return None, "Colonna Close non trovata"
            close = df["Close"]

        out = pd.DataFrame({
            "date": pd.to_datetime(df.index),
            "close": pd.to_numeric(close, errors="coerce"),
        })

        out = out.dropna().sort_values("date")

        return out, None

    except Exception as e:
        return None, str(e)


def update_asset(name, cfg):
    symbol = cfg["symbol"]
    path = cfg["path"]

    log(f"\n[{name.upper()}] {symbol}")

    existing = read_existing_csv(path)

    if not cfg["update_enabled"]:
        log("Manuale → salto")
        return True

    fresh, err = fetch_yahoo(symbol)

    if fresh is None:
        log(f"Errore Yahoo: {err}")
        return existing is not None

    merged = (
        pd.concat([existing, fresh])
        if existing is not None
        else fresh
    )

    merged = merged.drop_duplicates("date").sort_values("date")

    if len(merged) < MIN_VALID_ROWS:
        log("Serie troppo corta → skip")
        return False

    write_csv(path, merged)

    log(f"Aggiornato: {len(merged)} righe")
    return True


def main():
    log("=== AGGIORNAMENTO DATI ===")

    results = []

    for name, cfg in ASSETS.items():
        ok = update_asset(name, cfg)
        results.append(ok)

    log("=== FINE ===")

    if not all(results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
