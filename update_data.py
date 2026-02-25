from __future__ import annotations

import os
import time
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
import yfinance as yf

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

LS80_TICKER = os.getenv("LS80_TICKER", "VINGA80.MI").strip()
GOLD_TICKER = os.getenv("GOLD_TICKER", "GLD").strip()

LS80_FILE = DATA_DIR / "ls80.csv"
GOLD_FILE = DATA_DIR / "gold.csv"

MAX_RETRIES = int(os.getenv("YF_MAX_RETRIES", "5"))
SLEEP_SECONDS = int(os.getenv("YF_SLEEP_SECONDS", "25"))  # anti rate-limit


# ---------- CSV helpers (robusti) ----------
def _normalize_local_prices_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rende compatibili i CSV locali (accetta: date/Date, price/Close/Adj Close).
    Output: colonne ['date','price'] con date tipo datetime.date
    """
    cols = {c: c.strip() for c in df.columns}
    df = df.rename(columns=cols)

    # data col
    if "date" not in df.columns:
        if "Date" in df.columns:
            df = df.rename(columns={"Date": "date"})
        elif df.columns.size >= 1:
            # spesso la prima colonna è la data
            df = df.rename(columns={df.columns[0]: "date"})

    # price col
    if "price" not in df.columns:
        if "Close" in df.columns:
            df = df.rename(columns={"Close": "price"})
        elif "Adj Close" in df.columns:
            df = df.rename(columns={"Adj Close": "price"})
        elif df.columns.size >= 2:
            df = df.rename(columns={df.columns[1]: "price"})

    # pulizia
    df = df[["date", "price"]].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df["price"] = pd.to_numeric(df["price"], errors="coerce")

    df = df.dropna(subset=["date", "price"]).sort_values("date")
    df = df.drop_duplicates(subset=["date"], keep="last")
    return df


def read_local_csv(file_path: Path) -> Optional[pd.DataFrame]:
    if not file_path.exists():
        return None
    df = pd.read_csv(file_path)
    if df.empty:
        return None
    return _normalize_local_prices_df(df)


def write_csv(file_path: Path, df: pd.DataFrame) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(file_path, index=False)


# ---------- Yahoo download (robusto) ----------
def safe_download(ticker: str, start_date: Optional[pd.Timestamp] = None) -> Optional[pd.DataFrame]:
    """
    Scarica dati da Yahoo con retry. Ritorna DF normalizzato ['date','price'] oppure None se fallisce.
    Nota: su GitHub Actions Yahoo può restituire vuoto ("possibly delisted"): lo gestiamo.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"Download {ticker} - tentativo {attempt}/{MAX_RETRIES}")

            if start_date is None:
                raw = yf.download(
                    ticker,
                    period="2y",
                    interval="1d",
                    auto_adjust=False,
                    progress=False,
                    threads=False,
                )
            else:
                raw = yf.download(
                    ticker,
                    start=start_date.strftime("%Y-%m-%d"),
                    interval="1d",
                    auto_adjust=False,
                    progress=False,
                    threads=False,
                )

            if raw is not None and not raw.empty:
                raw = raw.reset_index()
                # Nome colonna data può essere "Date"
                if "Date" in raw.columns:
                    raw = raw.rename(columns={"Date": "date"})
                elif "date" not in raw.columns and raw.columns.size > 0:
                    raw = raw.rename(columns={raw.columns[0]: "date"})

                # prezzo: preferiamo Close
                if "Close" in raw.columns:
                    raw = raw.rename(columns={"Close": "price"})
                elif "Adj Close" in raw.columns:
                    raw = raw.rename(columns={"Adj Close": "price"})
                elif "price" not in raw.columns and raw.columns.size > 1:
                    raw = raw.rename(columns={raw.columns[1]: "price"})

                out = raw[["date", "price"]].copy()
                out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.date
                out["price"] = pd.to_numeric(out["price"], errors="coerce")
                out = out.dropna(subset=["date", "price"]).drop_duplicates(subset=["date"]).sort_values("date")

                if not out.empty:
                    return out

            print(f"Nessun dato valido ricevuto per {ticker} (Yahoo vuoto/limitato).")

        except Exception as e:
            print(f"Errore download {ticker}: {e}")

        time.sleep(SLEEP_SECONDS)

    print(f"⚠️  Skip {ticker}: Yahoo non disponibile (rate-limit / vuoto).")
    return None


def merge_and_save(file_path: Path, df_new: pd.DataFrame) -> None:
    df_old = read_local_csv(file_path)
    if df_old is None:
        write_csv(file_path, df_new)
        print(f"Creato {file_path} → last={df_new['date'].iloc[-1]}")
        return

    merged = pd.concat([df_old, df_new], ignore_index=True)
    merged = merged.drop_duplicates(subset=["date"], keep="last").sort_values("date")

    write_csv(file_path, merged)
    print(f"Aggiornato {file_path} → last={merged['date'].iloc[-1]} (rows={len(merged)})")


def update_one(ticker: str, file_path: Path) -> None:
    print(f"\n=== Aggiornamento {ticker} ===")

    df_old = read_local_csv(file_path)
    if df_old is not None and not df_old.empty:
        last_date = df_old["date"].iloc[-1]
        # scarico dal giorno successivo
        start = pd.Timestamp(last_date) + pd.Timedelta(days=1)
    else:
        start = None

    df_new = safe_download(ticker, start_date=start)

    if df_new is None or df_new.empty:
        print(f"SKIP: nessun aggiornamento per {ticker} (mantengo file invariato).")
        return

    merge_and_save(file_path, df_new)


def main() -> int:
    print("Aggiornamento dati Gloob in corso...")
    print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")

    # 1) LS80
    update_one(LS80_TICKER, LS80_FILE)

    # pausa tra richieste (aiuta molto)
    time.sleep(15)

    # 2) GOLD
    update_one(GOLD_TICKER, GOLD_FILE)

    print("\n✅ Fine aggiornamento (senza errori bloccanti).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
