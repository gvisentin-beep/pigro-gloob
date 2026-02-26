from __future__ import annotations

import os
import random
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

LS80_TICKER = os.getenv("LS80_TICKER", "VNGA80.MI").strip()
GOLD_TICKER = os.getenv("GOLD_TICKER", "GLD").strip()

LS80_FILE = DATA_DIR / "ls80.csv"
GOLD_FILE = DATA_DIR / "gold.csv"

MAX_RETRIES = int(os.getenv("YF_MAX_RETRIES", "6"))
BASE_SLEEP = float(os.getenv("YF_BASE_SLEEP", "8"))  # seconds
JITTER = float(os.getenv("YF_JITTER", "2.5"))  # seconds


# =========================
# Helpers
# =========================
def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _detect_sep(path: Path) -> str:
    # Prefer ';' for your files, but auto-detect if needed
    try:
        head = path.read_text(encoding="utf-8", errors="ignore")[:2048]
    except Exception:
        return ";"
    if head.count(";") >= head.count(","):
        return ";"
    return ","


def _normalize_local_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Accepts many shapes, returns columns: date (datetime64), close (float)
    """
    cols = {c.strip(): c for c in df.columns}

    # Pick date column
    date_col = None
    for cand in ["Date", "date", "DATA", "Data", "timestamp", "Datetime"]:
        if cand in cols:
            date_col = cols[cand]
            break
    if date_col is None:
        # Sometimes index is date-like
        if df.index.name and "date" in df.index.name.lower():
            df = df.reset_index()
            date_col = df.columns[0]
        else:
            raise KeyError("Nessuna colonna data trovata (attese: Date/date/...).")

    # Pick close column
    close_col = None
    for cand in ["Close", "close", "Adj Close", "adjclose", "price", "Price", "PREZZO", "Prezzo"]:
        if cand in cols:
            close_col = cols[cand]
            break
    if close_col is None:
        raise KeyError("Nessuna colonna prezzo trovata (attese: Close/Adj Close/price/...).")

    out = df[[date_col, close_col]].copy()
    out.columns = ["date", "close"]

    # Parse date: your format is dd/mm/yyyy
    out["date"] = pd.to_datetime(out["date"], errors="coerce", dayfirst=True).dt.tz_localize(None)
    out = out.dropna(subset=["date"])

    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    out = out.dropna(subset=["close"])

    out = out.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    return out


def read_local_prices(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["date", "close"])

    sep = _detect_sep(path)
    df = pd.read_csv(path, sep=sep, engine="python")
    return _normalize_local_df(df)


def yf_download(ticker: str, start: Optional[pd.Timestamp]) -> pd.DataFrame:
    """
    Download daily prices. Returns df with columns: date, close
    """
    last_err: Optional[Exception] = None

    # Strategy:
    # - Try with start (if available) to reduce load
    # - If empty, fall back to period='max' and filter
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            kwargs = dict(interval="1d", auto_adjust=True, progress=False)
            if start is not None:
                df = yf.download(ticker, start=start.date().isoformat(), **kwargs)
            else:
                df = yf.download(ticker, period="max", **kwargs)

            if df is None or df.empty:
                # fallback once
                df = yf.download(ticker, period="max", **kwargs)

            if df is None or df.empty:
                raise RuntimeError("download vuoto (nessun dato)")

            # yfinance index is DatetimeIndex
            df = df.reset_index()

            # Normalize columns
            # With auto_adjust=True, Close exists
            date_col = None
            for c in df.columns:
                if str(c).lower() in ["date", "datetime"]:
                    date_col = c
                    break
            if date_col is None:
                date_col = df.columns[0]

            close_col = None
            for cand in ["Close", "Adj Close"]:
                if cand in df.columns:
                    close_col = cand
                    break
            if close_col is None:
                # sometimes lowercase
                for c in df.columns:
                    if str(c).lower() == "close":
                        close_col = c
                        break
            if close_col is None:
                raise RuntimeError("colonna Close non trovata nei dati scaricati")

            out = df[[date_col, close_col]].copy()
            out.columns = ["date", "close"]
            out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.tz_localize(None)
            out["close"] = pd.to_numeric(out["close"], errors="coerce")
            out = out.dropna(subset=["date", "close"]).sort_values("date").drop_duplicates("date", keep="last")

            if start is not None:
                out = out[out["date"] >= start]

            return out

        except Exception as e:
            last_err = e
            # backoff with jitter
            sleep_s = BASE_SLEEP * (attempt ** 1.2) + random.uniform(0, JITTER)
            print(f"[{ticker}] Tentativo {attempt}/{MAX_RETRIES} fallito: {e}. Sleep {sleep_s:.1f}s")
            time.sleep(sleep_s)

    raise RuntimeError(f"Impossibile scaricare dati per {ticker}. Ultimo errore: {last_err}")


def write_local_prices(path: Path, df: pd.DataFrame) -> None:
    """
    Writes Date;Close with dd/mm/yyyy and ';' separator
    """
    df2 = df.copy()
    df2 = df2.sort_values("date").drop_duplicates("date", keep="last")
    df2["Date"] = df2["date"].dt.strftime("%d/%m/%Y")
    df2["Close"] = df2["close"].map(lambda x: f"{x:.2f}")
    out = df2[["Date", "Close"]]
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, sep=";", index=False)


@dataclass
class UpdateResult:
    ticker: str
    file: Path
    updated: bool
    last_before: Optional[str]
    last_after: Optional[str]
    note: str


def update_one(ticker: str, file_path: Path) -> UpdateResult:
    local = read_local_prices(file_path)
    last_before = None
    if not local.empty:
        last_before = local["date"].iloc[-1].strftime("%Y-%m-%d")

    start = None
    if not local.empty:
        # start from next day (safe)
        start = local["date"].iloc[-1] + pd.Timedelta(days=1)

    try:
        new = yf_download(ticker, start=start)
    except Exception as e:
        # IMPORTANT: do not fail the whole workflow
        note = f"Download fallito: {e}. File lasciato invariato."
        print(f"[{ticker}] {note}")
        last_after = last_before
        return UpdateResult(ticker, file_path, False, last_before, last_after, note)

    if new.empty:
        note = "Nessun nuovo dato (download vuoto dopo filtro)."
        print(f"[{ticker}] {note}")
        last_after = last_before
        return UpdateResult(ticker, file_path, False, last_before, last_after, note)

    merged = pd.concat([local, new], ignore_index=True) if not local.empty else new
    merged = merged.sort_values("date").drop_duplicates("date", keep="last")

    last_after = merged["date"].iloc[-1].strftime("%Y-%m-%d")

    if last_before == last_after:
        note = "Nessun avanzamento (ultima data invariata)."
        print(f"[{ticker}] {note}")
        return UpdateResult(ticker, file_path, False, last_before, last_after, note)

    write_local_prices(file_path, merged)
    note = "Aggiornato."
    print(f"[{ticker}] {note} {last_before} -> {last_after}")
    return UpdateResult(ticker, file_path, True, last_before, last_after, note)


def main() -> int:
    print(f"Timestamp: {_now_utc()}")
    print(f"LS80_TICKER={LS80_TICKER} | GOLD_TICKER={GOLD_TICKER}")
    print(f"DATA_DIR={DATA_DIR}")

    res1 = update_one(LS80_TICKER, LS80_FILE)
    res2 = update_one(GOLD_TICKER, GOLD_FILE)

    print("--- RISULTATO ---")
    for r in [res1, res2]:
        print(f"{r.ticker} -> {r.file.name} | updated={r.updated} | {r.last_before} -> {r.last_after} | {r.note}")

    # Always exit 0 so scheduled job doesn't become a graveyard of failures.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
