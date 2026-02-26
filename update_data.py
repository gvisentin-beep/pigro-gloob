from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import yfinance as yf


# =========================
# Config
# =========================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

LS80_TICKER_DEFAULT = "VNGA80.MI"  # <-- CORRETTO
GOLD_TICKER_DEFAULT = "GLD"

LS80_FILE = DATA_DIR / "ls80.csv"
GOLD_FILE = DATA_DIR / "gold.csv"

MAX_RETRIES = int(os.getenv("YF_MAX_RETRIES", "8"))
BASE_SLEEP = float(os.getenv("YF_BASE_SLEEP", "10"))  # secondi
JITTER = float(os.getenv("YF_JITTER", "1.5"))         # randomizzazione semplice

# Se i file sono nel formato italiano Date;Close con date dd/mm/yyyy e ordine decrescente
OUT_SEP = ";"
OUT_DATE_FMT = "%d/%m/%Y"


# =========================
# Utils
# =========================
def _now_utc_str() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _detect_sep(text: str) -> str:
    # preferisci ; se presente nella prima riga (il tuo caso)
    first_line = text.splitlines()[0] if text else ""
    if ";" in first_line:
        return ";"
    if "," in first_line:
        return ","
    # fallback
    return ";"


def _read_local_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists() or path.stat().st_size == 0:
        return None

    raw = path.read_text(encoding="utf-8", errors="replace")
    sep = _detect_sep(raw)

    df = pd.read_csv(path, sep=sep, engine="python")
    if df.empty:
        return None
    return _normalize_prices_df(df)


def _normalize_prices_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizza vari formati possibili in due colonne: date, price
    Accetta:
    - Date/Close (il tuo)
    - date/price
    - Date/Adj Close, ecc.
    """
    cols = {c.strip(): c for c in df.columns}
    lower = {c.strip().lower(): c for c in df.columns}

    # date column
    date_col = None
    for key in ("date", "data"):
        if key in lower:
            date_col = lower[key]
            break
    if date_col is None:
        # prova la prima colonna
        date_col = df.columns[0]

    # price column (preferenze)
    price_col = None
    for key in ("adj close", "adj_close", "close", "prezzo", "price", "valore"):
        if key in lower:
            price_col = lower[key]
            break
    if price_col is None:
        # prova la seconda colonna
        if len(df.columns) >= 2:
            price_col = df.columns[1]
        else:
            raise ValueError("CSV locale: non trovo una colonna prezzo.")

    out = df[[date_col, price_col]].copy()
    out.columns = ["date", "price"]

    # parse date: dayfirst True (dd/mm/yyyy)
    out["date"] = pd.to_datetime(out["date"], errors="coerce", dayfirst=True).dt.date
    out["price"] = pd.to_numeric(out["price"], errors="coerce")

    out = out.dropna(subset=["date", "price"])
    out = out.drop_duplicates(subset=["date"]).sort_values("date")  # crescente internamente
    return out.reset_index(drop=True)


def _yf_download_daily(ticker: str, start: Optional[str] = None) -> pd.DataFrame:
    """
    Scarica dati giornalieri da Yahoo via yfinance.
    Ritorna DF con index datetime e colonne OHLC + Adj Close/Close.
    """
    # Nota: yfinance è instabile; usiamo retry fuori
    df = yf.download(
        tickers=ticker,
        start=start,
        progress=False,
        interval="1d",
        auto_adjust=False,
        group_by="column",
        threads=False,
    )
    # df può essere vuoto
    return df


def _download_with_retry(ticker: str, start: Optional[str]) -> pd.DataFrame:
    last_err = None
    for i in range(1, MAX_RETRIES + 1):
        try:
            df = _yf_download_daily(ticker, start=start)
            if df is not None and not df.empty:
                return df
            last_err = RuntimeError(f"Nessun dato da Yahoo per {ticker} (vuoto).")
        except Exception as e:
            last_err = e

        sleep_s = BASE_SLEEP * i
        # jitter semplice senza random (per evitare import extra)
        sleep_s += (i % 3) * JITTER
        print(f"  Tentativo {i}/{MAX_RETRIES} fallito per {ticker}: {last_err}")
        print(f"  Attendo {sleep_s:.1f}s e riprovo...")
        time.sleep(sleep_s)

    raise RuntimeError(f"Impossibile scaricare dati per {ticker}. Ultimo errore: {last_err}")


def _yf_to_prices(df_yf: pd.DataFrame) -> pd.DataFrame:
    """
    Converte df Yahoo in df con colonne date, price.
    Preferisce Adj Close se disponibile altrimenti Close.
    """
    col = "Adj Close" if "Adj Close" in df_yf.columns else "Close"
    if col not in df_yf.columns:
        raise RuntimeError("Yahoo DF: non trovo né 'Adj Close' né 'Close'.")

    out = df_yf[[col]].copy()
    out = out.reset_index()  # Date column
    out.columns = ["date", "price"]
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.date
    out["price"] = pd.to_numeric(out["price"], errors="coerce")
    out = out.dropna(subset=["date", "price"])
    out = out.drop_duplicates(subset=["date"]).sort_values("date")
    return out.reset_index(drop=True)


def _merge_and_save(path: Path, df_old: Optional[pd.DataFrame], df_new: pd.DataFrame) -> Tuple[int, int]:
    """
    Merge per data, salva in formato Date;Close con date dd/mm/yyyy, ordine decrescente.
    Ritorna: (n_righe_prima, n_righe_dopo)
    """
    n_before = int(df_old.shape[0]) if df_old is not None else 0

    if df_old is None or df_old.empty:
        merged = df_new.copy()
    else:
        merged = pd.concat([df_old, df_new], ignore_index=True)
        merged = merged.drop_duplicates(subset=["date"], keep="last").sort_values("date")

    merged = merged.reset_index(drop=True)

    # scrivi in formato "Date;Close" con date italiane
    out = merged.copy()
    out["Date"] = pd.to_datetime(out["date"]).dt.strftime(OUT_DATE_FMT)
    out["Close"] = out["price"].astype(float).round(6)
    out = out[["Date", "Close"]]

    # per coerenza con i tuoi file: ordine decrescente (ultima data in alto)
    out = out.iloc[::-1].reset_index(drop=True)

    path.write_text("", encoding="utf-8")  # reset
    out.to_csv(path, sep=OUT_SEP, index=False, encoding="utf-8")

    n_after = int(merged.shape[0])
    return n_before, n_after


def _ticker_fixups(t: str) -> str:
    t = (t or "").strip()
    # Fix automatico errore comune (VINGA80.MI -> VNGA80.MI)
    if t.upper() == "VINGA80.MI":
        return "VNGA80.MI"
    return t


def update_one(ticker: str, file_path: Path) -> bool:
    ticker = _ticker_fixups(ticker)
    print(f"\n=== Aggiornamento {ticker} -> {file_path} ===")

    df_old = _read_local_csv(file_path)

    # start: giorno dopo l'ultima data locale, così scarichiamo solo incrementale
    start = None
    if df_old is not None and not df_old.empty:
        last_date = max(df_old["date"])
        start = (pd.Timestamp(last_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        print(f"  Ultima data locale: {last_date} | Start download: {start}")
    else:
        # scarica parecchio storico (da 2000 in poi)
        start = "2000-01-01"
        print("  Nessun file locale valido. Scarico storico da 2000-01-01.")

    df_yf = _download_with_retry(ticker, start=start)
    df_new = _yf_to_prices(df_yf)

    if df_new.empty:
        print("  Nessun dato nuovo da unire.")
        return False

    before, after = _merge_and_save(file_path, df_old, df_new)
    print(f"  Righe: {before} -> {after}")
    print(f"  Ultima data (dopo merge): {pd.to_datetime(df_new['date'].max()).date()}")
    return after > before


def main() -> int:
    print(f"Timestamp UTC: {_now_utc_str()}")
    ls80_ticker = _ticker_fixups(os.getenv("LS80_TICKER", LS80_TICKER_DEFAULT))
    gold_ticker = _ticker_fixups(os.getenv("GOLD_TICKER", GOLD_TICKER_DEFAULT))

    changed_any = False
    ok_any = False

    # LS80
    try:
        ok_any = True
        changed_any |= update_one(ls80_ticker, LS80_FILE)
    except Exception as e:
        print(f"ERRORE LS80 ({ls80_ticker}): {e}")

    # GOLD
    try:
        ok_any = True
        changed_any |= update_one(gold_ticker, GOLD_FILE)
    except Exception as e:
        print(f"ERRORE GOLD ({gold_ticker}): {e}")

    if not ok_any:
        return 1

    # se non cambia nulla non è un errore: magari mercato chiuso
    print("\nDONE.")
    print("Changed:", changed_any)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
