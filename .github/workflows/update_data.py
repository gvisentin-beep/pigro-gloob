from __future__ import annotations

import os
import time
from pathlib import Path
from datetime import datetime

import pandas as pd
import yfinance as yf


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

LS80_TICKER = os.getenv("LS80_TICKER", "VNGA80.MI").strip()
GOLD_TICKER = os.getenv("GOLD_TICKER", "GLD").strip()

LS80_FILE = DATA_DIR / "ls80.csv"
GOLD_FILE = DATA_DIR / "gold.csv"

# Retry anti rate-limit / instabilità Yahoo
MAX_RETRIES = int(os.getenv("YF_MAX_RETRIES", "6"))
BASE_SLEEP = int(os.getenv("YF_BASE_SLEEP", "10"))  # secondi


def _detect_sep(path: Path) -> str:
    """Rileva separatore CSV (preferenza ;)."""
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            head = f.readline()
        if ";" in head and "," not in head:
            return ";"
        if "," in head and ";" not in head:
            return ","
        return ";" if head.count(";") >= head.count(",") else ","
    except Exception:
        return ";"


def _read_local_csv(path: Path) -> pd.DataFrame:
    """Legge CSV locale e restituisce colonne: date(datetime), close(float)."""
    if not path.exists():
        raise FileNotFoundError(f"File mancante: {path}")

    sep = _detect_sep(path)
    df = pd.read_csv(path, sep=sep)
    df.columns = [c.strip() for c in df.columns]

    cols = {c.lower(): c for c in df.columns}
    if "date" not in cols or "close" not in cols:
        raise ValueError(f"{path.name}: attese colonne Date e Close. Trovate: {list(df.columns)}")

    dcol = cols["date"]
    ccol = cols["close"]

    out = pd.DataFrame(
        {
            "date": pd.to_datetime(df[dcol], errors="coerce", dayfirst=True),
            "close": pd.to_numeric(df[ccol], errors="coerce"),
        }
    ).dropna()

    out = out.sort_values("date").drop_duplicates(subset=["date"], keep="last")

    # elimina timezone se presente
    try:
        out["date"] = out["date"].dt.tz_localize(None)
    except Exception:
        pass

    if len(out) < 10:
        raise ValueError(f"{path.name}: troppo poche righe valide ({len(out)}).")

    return out[["date", "close"]]


def _write_csv(path: Path, df: pd.DataFrame) -> None:
    """Scrive CSV in formato compatibile col tuo progetto: Date dd/mm/YYYY ; Close con 2 decimali."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    save = pd.DataFrame(
        {
            "Date": pd.to_datetime(df["date"]).dt.strftime("%d/%m/%Y"),
            "Close": pd.to_numeric(df["close"]).map(lambda x: f"{float(x):.2f}"),
        }
    )
    save.to_csv(path, sep=";", index=False)


def _fetch_history_with_retry(ticker: str) -> pd.DataFrame:
    """Scarica storico giornaliero da Yahoo con retry/backoff."""
    last_err: Exception | None = None

    for i in range(1, MAX_RETRIES + 1):
        try:
            t = yf.Ticker(ticker)
            hist = t.history(period="2y", interval="1d", auto_adjust=False)

            if hist is None or hist.empty or "Close" not in hist.columns:
                raise RuntimeError("Empty history from Yahoo")

            df = hist[["Close"]].rename(columns={"Close": "close"}).copy()
            df["date"] = pd.to_datetime(df.index, errors="coerce")
            try:
                df["date"] = df["date"].dt.tz_localize(None)
            except Exception:
                pass

            df["close"] = pd.to_numeric(df["close"], errors="coerce")
            df = df.dropna().sort_values("date")

            if len(df) < 10:
                raise RuntimeError("History too short")

            return df[["date", "close"]]

        except Exception as e:
            last_err = e
            sleep_s = BASE_SLEEP * i
            print(f"[{ticker}] tentativo {i}/{MAX_RETRIES} fallito: {type(e).__name__}: {e}")
            if i < MAX_RETRIES:
                print(f"[{ticker}] attendo {sleep_s}s e riprovo...")
                time.sleep(sleep_s)

    raise RuntimeError(f"[{ticker}] fallito dopo {MAX_RETRIES} tentativi: {last_err}")


def _update_one(ticker: str, path: Path) -> bool:
    """Aggiorna un CSV (se necessario). Ritorna True se ha scritto modifiche."""
    local = _read_local_csv(path)
    last_local = local["date"].iloc[-1].date()

    remote = _fetch_history_with_retry(ticker)
    last_remote = remote["date"].iloc[-1].date()

    new_rows = remote[remote["date"].dt.date > last_local]
    if new_rows.empty:
        print(f"[{ticker}] già aggiornato. local={last_local} remote={last_remote}")
        return False

    out = pd.concat([local, new_rows], ignore_index=True)
    out = out.sort_values("date").drop_duplicates(subset=["date"], keep="last")

    _write_csv(path, out)

    print(f"[{ticker}] aggiornato: +{len(new_rows)} righe, nuova ultima data={out['date'].iloc[-1].date()}")
    return True


def main() -> None:
    print("---- GLOOB update_data.py ----")
    print("UTC:", datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"))
    print("DATA_DIR:", DATA_DIR)
    print("LS80_TICKER:", LS80_TICKER)
    print("GOLD_TICKER:", GOLD_TICKER)
    print("-----------------------------")

    changed = False
    changed |= _update_one(LS80_TICKER, LS80_FILE)
    changed |= _update_one(GOLD_TICKER, GOLD_FILE)

    if changed:
        print("OK: file CSV aggiornati.")
    else:
        print("OK: nessuna modifica (già aggiornati).")


if __name__ == "__main__":
    main()
