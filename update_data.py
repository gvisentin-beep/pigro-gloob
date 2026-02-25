from __future__ import annotations

import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
import yfinance as yf


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

LS80_TICKER = os.getenv("LS80_TICKER", "VNGA80.MI").strip()
GOLD_TICKER = os.getenv("GOLD_TICKER", "GLD").strip()

LS80_FILE = DATA_DIR / "ls80.csv"
GOLD_FILE = DATA_DIR / "gold.csv"

MAX_RETRIES = int(os.getenv("YF_MAX_RETRIES", "7"))
BASE_SLEEP = float(os.getenv("YF_BASE_SLEEP", "8"))  # seconds


def _log(msg: str) -> None:
    print(msg, flush=True)


def _detect_sep_from_header_line(path: Path) -> str:
    try:
        head = path.read_text(encoding="utf-8", errors="ignore").splitlines()[0]
    except Exception:
        return ";"
    # i tuoi file sono tipicamente "Date;Close"
    if ";" in head:
        return ";"
    if "," in head:
        return ","
    # fallback
    return ";"


def _read_local_prices(path: Path) -> pd.DataFrame:
    """
    Legge CSV locali in molti formati e restituisce SEMPRE un DF con colonne:
      - date (datetime64[ns])
      - price (float)
    Supporta:
      - Date;Close (anche quando finisce in un'unica colonna "Date;Close")
      - date,price
      - Date,Close
      - ecc.
    """
    if not path.exists():
        raise FileNotFoundError(f"File non trovato: {path}")

    sep = _detect_sep_from_header_line(path)

    # Primo tentativo: sep rilevato
    df = pd.read_csv(path, sep=sep, engine="python")

    # Caso “tutto in una colonna” (tipico quando il separatore non viene interpretato)
    # es: colonna unica chiamata "Date;Close" con righe tipo "21/01/2026;39.37"
    if df.shape[1] == 1:
        col0 = df.columns[0]
        if ";" in col0 or "," in col0:
            # ri-leggi forzando il separatore giusto
            forced_sep = ";" if ";" in col0 else ","
            df = pd.read_csv(path, sep=forced_sep, engine="python")

    # Normalizza nomi colonne
    cols = {c: c.strip() for c in df.columns}
    df = df.rename(columns=cols)

    # Trova colonna data
    date_col = None
    for c in df.columns:
        lc = c.lower()
        if lc in ("date", "data", "day", "datetime"):
            date_col = c
            break
    if date_col is None:
        date_col = df.columns[0]

    # Trova colonna prezzo
    price_col = None
    for c in df.columns:
        lc = c.lower().replace(" ", "")
        if lc in ("price", "close", "adjclose", "adj_close", "value", "last"):
            price_col = c
            break
    if price_col is None:
        # spesso è la seconda colonna
        if df.shape[1] >= 2:
            price_col = df.columns[1]
        else:
            raise ValueError(f"Formato CSV non riconosciuto (manca colonna prezzo): {path}")

    out = pd.DataFrame()
    out["date"] = pd.to_datetime(df[date_col], dayfirst=True, errors="coerce")
    # forza numerico (gestisce anche eventuali virgole decimali)
    out["price"] = (
        df[price_col]
        .astype(str)
        .str.replace(",", ".", regex=False)
        .str.strip()
    )
    out["price"] = pd.to_numeric(out["price"], errors="coerce")

    out = out.dropna(subset=["date", "price"]).copy()
    out["date"] = out["date"].dt.normalize()
    out = out.drop_duplicates(subset=["date"], keep="last")
    out = out.sort_values("date").reset_index(drop=True)

    if out.empty:
        raise ValueError(f"CSV vuoto o non parsabile: {path}")

    return out


def _yf_download_daily(ticker: str, start: datetime) -> pd.DataFrame:
    """
    Scarica daily OHLCV da Yahoo e restituisce DF con colonne date, price (Close).
    """
    # start leggermente “prima” per sicurezza su festivi/timezone
    start2 = start - timedelta(days=7)

    last_err: Optional[Exception] = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            hist = yf.download(
                ticker,
                start=start2.strftime("%Y-%m-%d"),
                interval="1d",
                auto_adjust=False,
                progress=False,
                threads=False,
            )
            if hist is None or hist.empty:
                raise RuntimeError("No data returned")

            # Se MultiIndex, appiattisci
            if isinstance(hist.columns, pd.MultiIndex):
                hist.columns = [c[0] for c in hist.columns]

            # Preferisci Close
            col = "Close" if "Close" in hist.columns else None
            if col is None:
                # fallback: Adj Close
                col = "Adj Close" if "Adj Close" in hist.columns else None
            if col is None:
                raise RuntimeError(f"Missing Close/Adj Close in Yahoo data. Columns={list(hist.columns)}")

            df = pd.DataFrame({"date": hist.index, "price": hist[col].values})
            df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
            df["price"] = pd.to_numeric(df["price"], errors="coerce")
            df = df.dropna(subset=["date", "price"]).copy()
            df = df.sort_values("date").drop_duplicates(subset=["date"], keep="last")

            # tieni solo da start in poi (giorno successivo all'ultimo locale)
            df = df[df["date"] >= pd.to_datetime(start).normalize()]

            return df.reset_index(drop=True)

        except Exception as e:
            last_err = e
            sleep_s = BASE_SLEEP * (1.3 ** (attempt - 1))
            _log(f"[{ticker}] Tentativo {attempt}/{MAX_RETRIES} fallito: {type(e).__name__}: {e}")
            if attempt < MAX_RETRIES:
                _log(f"[{ticker}] Attendo {sleep_s:.1f}s e riprovo...")
                time.sleep(sleep_s)

    raise RuntimeError(f"[{ticker}] Impossibile scaricare dati dopo {MAX_RETRIES} tentativi. Ultimo errore: {last_err}")


def _merge_and_save_legacy(path: Path, df_old: pd.DataFrame, df_new: pd.DataFrame) -> Tuple[int, datetime, datetime]:
    """
    Unisce e salva nel formato “legacy” identico ai tuoi file:
      - separatore ';'
      - header: Date;Close
      - date: dd/mm/YYYY
    """
    before_last = df_old["date"].max()

    df_all = pd.concat([df_old, df_new], ignore_index=True)
    df_all = df_all.drop_duplicates(subset=["date"], keep="last").sort_values("date").reset_index(drop=True)

    after_last = df_all["date"].max()

    # salva legacy
    out = pd.DataFrame()
    out["Date"] = df_all["date"].dt.strftime("%d/%m/%Y")
    # mantieni tanti decimali quanti arrivano, senza forzare troppo
    out["Close"] = df_all["price"].map(lambda x: f"{float(x):.6f}".rstrip("0").rstrip("."))
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, sep=";", index=False)

    added = int((after_last - before_last).days)  # grezzo ma utile come indicatore
    return len(df_new), before_last.to_pydatetime(), after_last.to_pydatetime()


def update_one(name: str, ticker: str, file_path: Path) -> Dict[str, object]:
    info: Dict[str, object] = {"asset": name, "ticker": ticker, "file": str(file_path), "updated": False}

    df_old = _read_local_prices(file_path)
    last_local = df_old["date"].max().to_pydatetime()

    # chiedi da “giorno dopo”
    start = (last_local + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)

    info["last_local_date"] = last_local.date().isoformat()

    df_new = _yf_download_daily(ticker, start=start)

    if df_new.empty:
        info["reason"] = "already_up_to_date_or_no_new_data"
        info["updated"] = False
        return info

    n_new, before_last, after_last = _merge_and_save_legacy(file_path, df_old, df_new)

    info["updated"] = True
    info["new_rows"] = n_new
    info["before_last_date"] = before_last.date().isoformat()
    info["after_last_date"] = after_last.date().isoformat()
    return info


def main() -> int:
    _log(f"Timestamp: {datetime.utcnow().isoformat()}Z")
    _log(f"Tickers: LS80={LS80_TICKER} | GOLD={GOLD_TICKER}")
    _log(f"Data dir: {DATA_DIR}")

    results = []
    ok_any = False

    for name, ticker, fpath in [
        ("ls80", LS80_TICKER, LS80_FILE),
        ("gold", GOLD_TICKER, GOLD_FILE),
    ]:
        _log(f"\n=== Aggiornamento {ticker} ({name}) ===")
        try:
            r = update_one(name, ticker, fpath)
            results.append(r)
            if r.get("updated"):
                ok_any = True
                _log(f"[OK] {ticker} aggiornato: +{r.get('new_rows')} righe, last={r.get('after_last_date')}")
            else:
                _log(f"[OK] {ticker} non aggiornato: {r.get('reason')}, last_local={r.get('last_local_date')}")
        except Exception as e:
            # IMPORTANTISSIMO: non facciamo fallire il workflow.
            # Logghiamo l’errore e continuiamo.
            results.append({"asset": name, "ticker": ticker, "file": str(fpath), "updated": False, "error": str(e)})
            _log(f"[WARN] {ticker} errore: {type(e).__name__}: {e}")

    _log("\n=== RISULTATO ===")
    for r in results:
        _log(str(r))

    if not ok_any:
        _log("Nessun aggiornamento effettuato (o Yahoo non ha risposto). Workflow comunque OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
