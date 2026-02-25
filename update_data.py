from __future__ import annotations

import os
import time
from pathlib import Path
from datetime import datetime

import pandas as pd
import yfinance as yf

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

LS80_TICKER = os.getenv("LS80_TICKER", "VINGA80.MI").strip()
GOLD_TICKER = os.getenv("GOLD_TICKER", "GLD").strip()

LS80_FILE = DATA_DIR / "ls80.csv"
GOLD_FILE = DATA_DIR / "gold.csv"

MAX_RETRIES = 5
SLEEP_SECONDS = 20  # anti rate limit


def safe_download(ticker: str):
    """Download con retry anti rate limit Yahoo"""
    for attempt in range(MAX_RETRIES):
        try:
            print(f"Download {ticker} - tentativo {attempt+1}")
            df = yf.download(
                ticker,
                period="6mo",
                interval="1d",
                auto_adjust=False,
                progress=False,
                threads=False,
            )

            if df is not None and not df.empty:
                df = df.reset_index()[["Date", "Close"]]
                df.columns = ["date", "price"]
                df["date"] = pd.to_datetime(df["date"]).dt.date
                return df

            print(f"Nessun dato per {ticker}")
        except Exception as e:
            print(f"Errore download {ticker}: {e}")

        time.sleep(SLEEP_SECONDS)

    print(f"⚠️ Skip {ticker} (rate limit Yahoo)")
    return None


def merge_and_save(file_path: Path, new_df: pd.DataFrame):
    if not file_path.exists():
        print(f"Creazione nuovo file: {file_path}")
        file_path.parent.mkdir(parents=True, exist_ok=True)
        new_df.to_csv(file_path, index=False)
        return

    old_df = pd.read_csv(file_path)
    old_df["date"] = pd.to_datetime(old_df["date"]).dt.date

    merged = pd.concat([old_df, new_df])
    merged = merged.drop_duplicates(subset=["date"]).sort_values("date")

    merged.to_csv(file_path, index=False)
    print(f"Aggiornato: {file_path} → {merged['date'].iloc[-1]}")


def update_ticker(ticker: str, file_path: Path):
    print(f"\n=== Aggiornamento {ticker} ===")

    df_new = safe_download(ticker)

    if df_new is None:
        print(f"Skip aggiornamento {ticker} (Yahoo bloccato)")
        return

    merge_and_save(file_path, df_new)


def main():
    print("Aggiornamento dati in corso...")
    print(f"Timestamp: {datetime.utcnow()} UTC")

    update_ticker(LS80_TICKER, LS80_FILE)

    # pausa tra i due download (fondamentale)
    time.sleep(15)

    update_ticker(GOLD_TICKER, GOLD_FILE)

    print("Aggiornamento completato con successo.")


if __name__ == "__main__":
    main()
