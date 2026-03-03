import os
from datetime import datetime, timezone
import pandas as pd
import yfinance as yf

DATA_DIR = "data"

def log(msg: str):
    print(msg, flush=True)

def ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)

def detect_sep(file_path: str) -> str:
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            head = f.readline()
        if ";" in head and "," not in head:
            return ";"
        if "," in head and ";" not in head:
            return ","
        return ";" if head.count(";") >= head.count(",") else ","
    except Exception:
        return ";"

def safe_read_csv(file_path: str) -> pd.DataFrame:
    if not os.path.exists(file_path):
        log(f"File non trovato, creo vuoto: {file_path}")
        return pd.DataFrame(columns=["date", "price"])

    try:
        sep = detect_sep(file_path)
        df = pd.read_csv(file_path, sep=sep)
        df.columns = [c.strip().lower() for c in df.columns]

        # accetta Date/Close (formato del tuo progetto)
        if "date" in df.columns and "close" in df.columns:
            df = df.rename(columns={"close": "price"})
        elif "date" in df.columns and "price" in df.columns:
            pass
        else:
            log(f"CSV {file_path}: colonne non riconosciute {list(df.columns)}")
            return pd.DataFrame(columns=["date", "price"])

        df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
        df = df.dropna(subset=["date", "price"])
        df = df.sort_values("date").drop_duplicates(subset="date", keep="last")
        return df[["date", "price"]]
    except Exception as e:
        log(f"Errore lettura CSV {file_path}: {type(e).__name__}: {e}")
        return pd.DataFrame(columns=["date", "price"])

def download_data(ticker: str) -> pd.DataFrame | None:
    try:
        log(f"Download {ticker}...")
        df = yf.download(
            ticker,
            period="1y",
            interval="1d",
            progress=False,
            auto_adjust=True,
            threads=False,
        )
        if df is None or df.empty:
            return None

        df = df.reset_index()

        # yfinance può restituire Date o Datetime a seconda degli asset
        if "Date" in df.columns:
            df = df.rename(columns={"Date": "date"})
        elif "Datetime" in df.columns:
            df = df.rename(columns={"Datetime": "date"})
        else:
            return None

        if "Close" not in df.columns:
            return None

        df = df.rename(columns={"Close": "price"})
        df = df[["date", "price"]]
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
        df = df.dropna(subset=["date", "price"])
        df = df.sort_values("date").drop_duplicates(subset="date", keep="last")
        return df
    except Exception as e:
        log(f"Errore download {ticker}: {type(e).__name__}: {e}")
        return None

def save_csv(df: pd.DataFrame, file_path: str):
    df = df.sort_values("date").drop_duplicates(subset="date", keep="last")
    out = pd.DataFrame(
        {
            "Date": df["date"].dt.strftime("%d/%m/%Y"),
            "Close": df["price"].round(4),
        }
    )
    out.to_csv(file_path, sep=";", index=False)
    log(f"Salvato {file_path} ({len(out)} righe)")

def update_one(ticker: str, file_path: str):
    log(f"\n=== Aggiornamento {ticker} ===")
    old_df = safe_read_csv(file_path)
    new_df = download_data(ticker)

    if new_df is None or new_df.empty:
        log(f"Nessun dato nuovo per {ticker} (skip)")
        if not old_df.empty:
            log(f"Ultima data locale {ticker}: {old_df['date'].iloc[-1].date()}")
        return

    df = pd.concat([old_df, new_df], ignore_index=True)
    df = df.drop_duplicates(subset="date", keep="last").sort_values("date")

    save_csv(df, file_path)
    log(f"Ultima data aggiornata {ticker}: {df['date'].iloc[-1].date()}")

def main():
    # ✅ usa gli stessi ticker del tuo Render, se presenti
    ls80_ticker = os.getenv("LS80_TICKER", "VNGA80.MI").strip()
    gold_ticker = os.getenv("GOLD_TICKER", "SGLD.MI").strip()

    ls80_file = os.path.join(DATA_DIR, "ls80.csv")
    gold_file = os.path.join(DATA_DIR, "gold.csv")

    log("Aggiornamento automatico dati Gloob")
    log(f"UTC: {datetime.now(timezone.utc)}")
    log(f"Tickers: LS80={ls80_ticker} | GOLD={gold_ticker}")

    ensure_data_dir()
    update_one(ls80_ticker, ls80_file)
    update_one(gold_ticker, gold_file)

    log("Aggiornamento completato con successo.")

if __name__ == "__main__":
    main()
