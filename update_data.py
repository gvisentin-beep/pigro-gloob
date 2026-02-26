from __future__ import annotations

import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import yfinance as yf


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

LS80_TICKER = os.getenv("LS80_TICKER", "VNGA80.MI").strip()
GOLD_TICKER = os.getenv("GOLD_TICKER", "GLD").strip()

LS80_FILE = DATA_DIR / "ls80.csv"
GOLD_FILE = DATA_DIR / "gold.csv"

MAX_RETRIES = int(os.getenv("YF_MAX_RETRIES", "6"))
BASE_SLEEP = float(os.getenv("YF_BASE_SLEEP", "10"))  # seconds


# -----------------------------
# CSV helpers (robusti)
# -----------------------------
def _detect_sep(sample: str) -> str:
    # Se troviamo ';' quasi certamente è il tuo formato
    if sample.count(";") >= sample.count(","):
        return ";"
    return ","


def _parse_date_series(s: pd.Series) -> pd.Series:
    """
    Supporta:
    - dd/mm/yyyy (tuo formato attuale)
    - yyyy-mm-dd (formato ISO)
    """
    s = s.astype(str).str.strip()
    # prova prima gg/mm/aaaa
    d1 = pd.to_datetime(s, format="%d/%m/%Y", errors="coerce")
    if d1.notna().any():
        return d1
    # fallback generico
    return pd.to_datetime(s, errors="coerce")


def read_local_prices_csv(path: Path) -> pd.DataFrame:
    """
    Ritorna un DF normalizzato con colonne:
    - date (datetime64[ns])
    - close (float)
    """
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=["date", "close"])

    raw = path.read_text(encoding="utf-8", errors="ignore")
    lines = [ln for ln in raw.splitlines() if ln.strip()]
    if not lines:
        return pd.DataFrame(columns=["date", "close"])

    sep = _detect_sep(lines[0])

    df = pd.read_csv(
        path,
        sep=sep,
        engine="python",
        dtype=str,
    )

    # Normalizza nomi colonne possibili
    cols = {c.strip(): c for c in df.columns}

    # data
    date_col = None
    for cand in ["Date", "date", "DATA", "Data"]:
        if cand in cols:
            date_col = cols[cand]
            break

    # close
    close_col = None
    for cand in ["Close", "close", "CLOSE", "Adj Close", "AdjClose", "adjclose", "Price", "price"]:
        if cand in cols:
            close_col = cols[cand]
            break

    if date_col is None or close_col is None:
        # Non distruggiamo nulla: segnaliamo e torniamo vuoto
        raise RuntimeError(f"CSV locale {path} non ha colonne riconoscibili (Date/Close). Colonne: {list(df.columns)}")

    df = df[[date_col, close_col]].copy()
    df.columns = ["date_raw", "close_raw"]

    df["date"] = _parse_date_series(df["date_raw"])
    df["close"] = pd.to_numeric(df["close_raw"].astype(str).str.replace(",", ".", regex=False), errors="coerce")

    df = df.drop(columns=["date_raw", "close_raw"]).dropna(subset=["date", "close"])
    df = df.sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)

    return df


def write_local_prices_csv(path: Path, df: pd.DataFrame) -> None:
    """
    Salva nel tuo formato:
    Date;Close
    21/01/2026;39.37
    (date DESC, come nei tuoi file)
    """
    out = df.copy()
    out = out.dropna(subset=["date", "close"]).copy()
    out["Date"] = pd.to_datetime(out["date"]).dt.strftime("%d/%m/%Y")
    out["Close"] = out["close"].astype(float).map(lambda x: f"{x:.2f}")
    out = out[["Date", "Close"]].sort_values("Date", ascending=False)

    path.parent.mkdir(parents=True, exist_ok=True)
    # sep=';' e niente indice
    out.to_csv(path, sep=";", index=False, encoding="utf-8")


# -----------------------------
# Download + merge
# -----------------------------
def _yf_download_close(ticker: str) -> pd.DataFrame:
    """
    Scarica dati da yfinance e ritorna DF normalizzato (date/close).
    Usa auto_adjust=True così Close è già "aggiustato" (più stabile).
    """
    # Tentiamo più finestre, in caso Yahoo dia "no data" su period=6mo ecc.
    periods = ["max", "10y", "5y", "2y", "1y"]
    last_err: Optional[Exception] = None

    for period in periods:
        try:
            df = yf.download(
                ticker,
                period=period,
                auto_adjust=True,
                progress=False,
                group_by="column",
                threads=False,
            )
            if df is None or df.empty:
                continue

            # yfinance mette DateTimeIndex
            df = df.reset_index()

            # La colonna data può chiamarsi "Date" o "Datetime"
            date_col = "Date" if "Date" in df.columns else ("Datetime" if "Datetime" in df.columns else None)
            if date_col is None:
                continue

            # Close dovrebbe esserci; se no proviamo "Adj Close"
            close_col = "Close" if "Close" in df.columns else ("Adj Close" if "Adj Close" in df.columns else None)
            if close_col is None:
                continue

            out = df[[date_col, close_col]].copy()
            out.columns = ["date", "close"]
            out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.tz_localize(None)
            out["close"] = pd.to_numeric(out["close"], errors="coerce")
            out = out.dropna(subset=["date", "close"]).sort_values("date").drop_duplicates(subset=["date"], keep="last")

            if not out.empty:
                return out.reset_index(drop=True)
        except Exception as e:
            last_err = e

    if last_err is None:
        raise RuntimeError(f"Nessun dato restituito da yfinance per {ticker}")
    raise RuntimeError(f"Nessun dato valido da yfinance per {ticker}. Ultimo errore: {last_err}")


def download_with_retry(ticker: str, max_retries: int = MAX_RETRIES, base_sleep: float = BASE_SLEEP) -> pd.DataFrame:
    last_err: Optional[Exception] = None
    for i in range(1, max_retries + 1):
        try:
            return _yf_download_close(ticker)
        except Exception as e:
            last_err = e
            # backoff con jitter
            sleep_s = base_sleep * i + random.uniform(0, base_sleep)
            print(f"[{ticker}] tentativo {i}/{max_retries} fallito: {e}. Attendo {sleep_s:.1f}s e riprovo...")
            time.sleep(sleep_s)

    raise RuntimeError(f"[{ticker}] falliti tutti i tentativi ({max_retries}). Ultimo errore: {last_err}")


def merge_old_new(old_df: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
    """
    Unisce e mantiene:
    - date uniche
    - preferisce i valori nuovi per date coincidenti
    - ordina crescente internamente
    """
    if old_df is None or old_df.empty:
        merged = new_df.copy()
    else:
        merged = pd.concat([old_df, new_df], ignore_index=True)
        merged = merged.sort_values("date").drop_duplicates(subset=["date"], keep="last")

    merged = merged.sort_values("date").reset_index(drop=True)
    return merged


def update_one(ticker: str, file_path: Path) -> Tuple[bool, Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    """
    Ritorna:
      (updated?, old_last_date, new_last_date)
    """
    old_df = read_local_prices_csv(file_path)
    old_last = old_df["date"].max() if not old_df.empty else None

    new_df = download_with_retry(ticker)
    new_last = new_df["date"].max() if not new_df.empty else None

    merged = merge_old_new(old_df, new_df)

    # scriviamo solo se è cambiato qualcosa (numero righe o last date)
    changed = False
    if old_df.empty and not merged.empty:
        changed = True
    elif (old_last is None and new_last is not None) or (old_last is not None and new_last is not None and new_last > old_last):
        changed = True
    elif len(merged) != len(old_df):
        changed = True

    if changed:
        write_local_prices_csv(file_path, merged)

    return changed, old_last, merged["date"].max() if not merged.empty else None


def main() -> int:
    print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Aggiornamento {LS80_TICKER} -> {LS80_FILE} ===")
    ch1, old1, new1 = update_one(LS80_TICKER, LS80_FILE)
    print(f"{LS80_TICKER}: changed={ch1} old_last={old1} new_last={new1}")

    print(f"\n=== Aggiornamento {GOLD_TICKER} -> {GOLD_FILE} ===")
    ch2, old2, new2 = update_one(GOLD_TICKER, GOLD_FILE)
    print(f"{GOLD_TICKER}: changed={ch2} old_last={old2} new_last={new2}")

    print("\nFINE.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
