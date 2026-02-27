import os
from datetime import datetime, timezone
import pandas as pd
import yfinance as yf

DATA_DIR = "data"

# Puoi cambiare ticker se vuoi (es. VWCE + oro ETC europeo)
LS80_TICKER = "VNGA80.MI"
GOLD_TICKER = "GLD"

LS80_FILE = os.path.join(DATA_DIR, "ls80.csv")
GOLD_FILE = os.path.join(DATA_DIR, "gold.csv")


def log(msg):
    print(msg, flush=True)


def ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)


def safe_read_csv(file_path):
    if not os.path.exists(file_path):
        log(f"Creo nuovo file: {file_path}")
        return pd.DataFrame(columns=["date", "price"])

    try:
        df = pd.read_csv(file_path, sep=";")
        df.columns = [c.lower() for c in df.columns]

        if "date" in df.columns and "close" in df.columns:
            df = df.rename(columns={"close": "price"})
            df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
            df = df.dropna(subset=["date"])
            return df[["date", "price"]]

        return pd.DataFrame(columns=["date", "price"])

    except Exception as e:
        log(f"Errore lettura CSV: {e}")
        return pd.DataFrame(columns=["date", "price"])


def download_data(ticker):
    try:
        log(f"Download {ticker}...")
        df = yf.download(
            ticker,
            period="1y",
            interval="1d",
            progress=False,
            auto_adjust=True,
            threads=False
        )

        if df is None or df.empty:
            log(f"Nessun dato nuovo per {ticker} (skip)")
            return None

        df = df.reset_index()
        df = df.rename(columns={"Date": "date", "Close": "price"})
        df = df[["date", "price"]]
        return df

    except Exception as e:
        log(f"Errore download {ticker}: {e}")
        return None  # NON crasha più!


def save_csv(df, file_path):
    df = df.sort_values("date")
    out = pd.DataFrame({
        "Date": df["date"].dt.strftime("%d/%m/%Y"),
        "Close": df["price"].round(4)
    })
    out.to_csv(file_path, sep=";", index=False)
    log(f"Salvato {file_path} ({len(out)} righe)")


def update_one(ticker, file_path):
    log(f"\n=== Aggiornamento {ticker} ===")
    old_df = safe_read_csv(file_path)
    new_df = download_data(ticker)

    if new_df is not None:
        df = pd.concat([old_df, new_df])
        df = df.drop_duplicates(subset="date", keep="last")
    else:
        df = old_df

    if len(df) == 0:
        log(f"Nessun dato disponibile per {ticker}, continuo senza errore.")
        return

    save_csv(df, file_path)


def main():
    log("Aggiornamento automatico dati Gloob")
    log(f"UTC: {datetime.now(timezone.utc)}")

    ensure_data_dir()
    update_one(LS80_TICKER, LS80_FILE)
    update_one(GOLD_TICKER, GOLD_FILE)

    log("Aggiornamento completato con successo.")


if __name__ == "__main__":
    main()
