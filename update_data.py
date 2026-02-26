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
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)


def read_local_csv(file_path):
    if not os.path.exists(file_path):
        log(f"Nessun file locale: {file_path}")
        return pd.DataFrame(columns=["date", "price"])

    df = pd.read_csv(file_path, sep=";")
    df.columns = [c.lower() for c in df.columns]

    if "date" not in df.columns or "close" not in df.columns:
        raise ValueError(f"CSV formato errato: {file_path}")

    df = df.rename(columns={"close": "price"})
    df["date"] = pd.to_datetime(df["date"], dayfirst=True)
    return df[["date", "price"]]


def save_csv(df, file_path):
    df = df.sort_values("date")
    df_out = pd.DataFrame({
        "Date": df["date"].dt.strftime("%d/%m/%Y"),
        "Close": df["price"].round(4)
    })
    df_out.to_csv(file_path, sep=";", index=False)
    log(f"Salvato: {file_path} ({len(df_out)} righe)")


def download_yahoo(ticker, retries=5):
    for i in range(retries):
        try:
            log(f"[{ticker}] Tentativo {i+1}/{retries}")
            data = yf.download(
                ticker,
                period="6mo",
                interval="1d",
                progress=False,
                auto_adjust=True,
                threads=False  # IMPORTANTISSIMO anti-rate limit
            )

            if data.empty:
                raise RuntimeError("No data returned")

            data = data.reset_index()
            data = data.rename(columns={"Date": "date", "Close": "price"})
            data = data[["date", "price"]]
            return data

        except Exception as e:
            wait = 10 * (i + 1)
            log(f"[{ticker}] Errore: {e}")
            log(f"[{ticker}] Attendo {wait}s e riprovo...")
            time.sleep(wait)

    log(f"[{ticker}] Download fallito definitivamente (uso dati esistenti)")
    return None


def merge_data(old_df, new_df):
    if new_df is None:
        return old_df

    merged = pd.concat([old_df, new_df])
    merged = merged.drop_duplicates(subset="date", keep="last")
    merged = merged.sort_values("date")
    return merged


def update_one(ticker, file_path):
    log(f"\n=== Aggiornamento {ticker} ===")
    old_df = read_local_csv(file_path)
    new_df = download_yahoo(ticker)

    final_df = merge_data(old_df, new_df)

    if len(final_df) == 0:
        raise RuntimeError(f"Nessun dato disponibile per {ticker}")

    save_csv(final_df, file_path)


def main():
    log("Aggiornamento dati Gloob in corso...")
    log(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")

    ensure_data_dir()

    update_one(LS80_TICKER, LS80_FILE)
    update_one(GOLD_TICKER, GOLD_FILE)

    log("Aggiornamento completato con successo.")


if __name__ == "__main__":
    main()
