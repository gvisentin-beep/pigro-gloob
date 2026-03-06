from __future__ import annotations
import pandas as pd
import yfinance as yf
from pathlib import Path
from datetime import datetime

DATA_DIR = Path("data")

FILES = {
    "ls80": ("VNGA80.MI", DATA_DIR / "ls80.csv"),
    "gold": ("SGLD.MI", DATA_DIR / "gold.csv"),
    "world": ("SMSWLD.MI", DATA_DIR / "world.csv")
}


def download_asset(ticker):
    df = yf.download(ticker, period="20y", interval="1d", progress=False)

    df = df.reset_index()

    df = df.rename(columns={
        "Date": "date",
        "Adj Close": "close"
    })

    df = df[["date", "close"]]

    df["date"] = df["date"].dt.strftime("%d/%m/%Y")

    return df


def save_csv(df, path):
    path.parent.mkdir(exist_ok=True)
    df.to_csv(path, sep=";", index=False)


def update_asset(name, ticker, path):
    print(f"Aggiorno {name} ({ticker})")

    df = download_asset(ticker)

    save_csv(df, path)

    print(
        f"{name}: righe {len(df)} "
        f"ultimo {df.iloc[-1]['date']} "
        f"valore {df.iloc[-1]['close']}"
    )


def main():

    print("Aggiornamento dati")

    for name, (ticker, path) in FILES.items():
        update_asset(name, ticker, path)

    print("Completato")


if __name__ == "__main__":
    main()
