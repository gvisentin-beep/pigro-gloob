from __future__ import annotations

import os
import time
from pathlib import Path
from datetime import datetime

import pandas as pd
import yfinance as yf


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

LS80_TICKER = os.getenv("LS80_TICKER", "VANG80.MI").strip()
GOLD_TICKER = os.getenv("GOLD_TICKER", "GLD").strip()

LS80_FILE = DATA_DIR / "ls80.csv"
GOLD_FILE = DATA_DIR / "gold.csv"

MAX_RETRIES = 6
BASE_SLEEP = 15


def download_ticker(ticker: str) -> pd.DataFrame:
    for attempt in range(MAX_RETRIES):
        try:
            df = yf.download(ticker, period="max", progress=False)
            if not df.empty:
                df = df.reset_index()[["Date", "Adj Close"]]
                df.columns = ["date", "close"]
                return df
        except Exception:
            pass

        sleep_time = BASE_SLEEP * (attempt + 1)
        time.sleep(sleep_time)

    raise RuntimeError(f"Impossibile scaricare dati per {ticker}")


def save_csv(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main():
    print("Aggiornamento dati in corso...")

    ls80_df = download_ticker(LS80_TICKER)
    gold_df = download_ticker(GOLD_TICKER)

    save_csv(ls80_df, LS80_FILE)
    save_csv(gold_df, GOLD_FILE)

    print("Aggiornamento completato.")
    print("Ultima data LS80:", ls80_df["date"].max())
    print("Ultima data GOLD:", gold_df["date"].max())


if __name__ == "__main__":
    main()
