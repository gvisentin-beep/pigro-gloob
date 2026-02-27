import os
import time
from datetime import datetime, timezone

import pandas as pd
import yfinance as yf

DATA_DIR = "data"

LS80_TICKER = "VNGA80.MI"
GOLD_TICKER = "GLD"

LS80_FILE = os.path.join(DATA_DIR, "ls80.csv")
GOLD_FILE = os.path.join(DATA_DIR, "gold.csv")


def log(msg):
    print(msg, flush=True)


def ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)


def safe_read_csv(file_path):
    """Legge CSV anche se formato diverso o mancante"""
    if not os.path.exists(file_path):
        log(f"File non trovato, creo nuovo: {file_path}")
        return pd.DataFrame(columns=["date", "price"])

    try:
        df = pd.read_csv(file_path, sep=";")
        df.columns = [c.lower() for c in df.columns]

        if "date" in df.columns and "close" in df.columns:
            df = df.rename(columns={"close": "price"})
            df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
            df = df.dropna(subset=["date"])
            return df[["date", "price"]]

        # fallback se colonne diverse
        log(f"Formato CSV non standard, ricostruisco: {file_path}")
        return pd.DataFrame(columns=["date", "price"])

    except Exception as e:
        log(f"Errore lettura CSV {file_path}: {e}")
        return pd.DataFrame(columns=["date", "price"])


def save_csv(df, file_path):
    df = df.sort_values("date")
    df_out = pd.DataFrame({
        "Date": df["date"].dt.strftime("%d/%m/%Y"),
        "Close": df["price"].round(4)
    })
    df_out.to_csv(file_path, sep=";", index=False)
    log(f"Salvato: {file_path} ({len(df_out)} righe)")


def download_yahoo_safe(ticker):
    """Download robusto che NON fa fallire il workflow"""
    try:
        log(f"Download dati per {ticker}...")
        data = yf.download(
            ticker,
            period="1y",
            interval="1d",
            progress=False,
            auto_adjust=True,
            threads=False
        )

        if data is None or data.empty:
            log(f"Nessun dato da Yahoo per {ticker} (skip)")
            return None

        data = data.reset_index()
        data = data.rename(columns={"Date": "date", "Close": "price"})
        data = data[["date", "price"]]
        data["date"] = pd.to_datetime(data["date"])
        return data

    except Exception as e:
        log(f"Errore download {ticker}: {e}")
        return None  # NON blocca più il workflow!


def update_one(ticker, file_path):
    log(f"\n=== Aggiornamento {ticker} ===")

    old_df = safe_read_csv(file_path)
    new_df = download_yahoo_safe(ticker)

    if new_df is None:
        log(f"Uso dati esistenti per {ticker}")
        final_df = old_df
    else:
        final_df = pd.concat([old_df, new_df])
        final_df = final_df.drop_duplicates(subset="date", keep="last")
        final_df = final_df.sort_values("date")

    if len(final_df) == 0:
        log(f"ATTENZIONE: nessun dato per {ticker}, ma continuo senza errore")
        return  # NON crasha più

    save_csv(final_df, file_path)


def main():
    log("=== Aggiornamento automatico dati Gloob ===")
    log(f"Timestamp UTC: {datetime.now(timezone.utc)}")

    ensure_data_dir()

    update_one(LS80_TICKER, LS80_FILE)
    update_one(GOLD_TICKER, GOLD_FILE)

    log("Aggiornamento completato (senza crash).")


if __name__ == "__main__":
    main()
