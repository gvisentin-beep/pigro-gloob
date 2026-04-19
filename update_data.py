from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import requests
import yfinance as yf

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

TWELVE_DATA_API_KEY = os.getenv("TWELVE_DATA_API_KEY", "").strip()
TWELVE_DATA_BASE_URL = "https://api.twelvedata.com/time_series"

MIN_VALID_ROWS = 100

ASSETS = {
    "ls80": {
        "symbol": "V80A.AS",
        "fallback_symbols": ["VNGA80.MI"],
        "path": DATA_DIR / "ls80.csv",
        "source": "yahoo",
        "update_enabled": True,
        "force_refresh": True,
    },
    "gold": {
        "symbol": os.getenv("GOLD_TICKER", "GLD").strip(),
        "path": DATA_DIR / "gold.csv",
        "source": "twelve",
        "update_enabled": True,
    },
    "btc": {
        "symbol": os.getenv("BTC_TICKER", "BTC/EUR").strip(),
        "path": DATA_DIR / "btc.csv",
        "source": "twelve",
        "update_enabled": True,
    },
    "world": {
        "symbol": os.getenv("WORLD_TICKER", "URTH").strip(),
        "path": DATA_DIR / "world.csv",
        "source": "yahoo",
        "update_enabled": True,
    },
    "eurostoxx": {
        "symbol": "CSSX5E.MI",
        "path": DATA_DIR / "mib.csv",
        "source": "yahoo",
        "update_enabled": True,
    },
    "sp500": {
        "symbol": os.getenv("SP500_TICKER", "SPY").strip(),
        "path": DATA_DIR / "sp500.csv",
        "source": "twelve",
        "update_enabled": True,
    },
}


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def log(msg: str) -> None:
    print(msg, flush=True)


def _to_naive_datetime(series: pd.Series, dayfirst: bool = False) -> pd.Series:
    s = pd.to_datetime(series, dayfirst=dayfirst, errors="coerce")
    try:
        if getattr(s.dt, "tz", None) is not None:
            s = s.dt.tz_convert(None)
    except Exception:
        pass
    try:
        s = s.dt.tz_localize(None)
    except Exception:
        pass
    return s


def read_existing_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None

    try:
        df = pd.read_csv(path, sep=";", dtype=str, encoding="utf-8-sig")
        df.columns = [str(c).strip().lower() for c in df.columns]
        if "date" not in df.columns or "close" not in df.columns:
            df = pd.read_csv(
                path,
                sep=";",
                header=None,
                names=["date", "close"],
                dtype=str,
                encoding="utf-8-sig",
            )
    except Exception:
        df = pd.read_csv(
            path,
            sep=";",
            header=None,
            names=["date", "close"],
            dtype=str,
            encoding="utf-8-sig",
        )

    df["date"] = _to_naive_datetime(df["date"], dayfirst=True)
    df["close"] = df["close"].astype(str).str.strip().str.replace(",", ".", regex=False)
    df["close"] = pd.to_numeric(df["close"], errors="coerce")

    df = (
        df.dropna(subset=["date", "close"])
        .sort_values("date")
        .drop_duplicates(subset=["date"], keep="last")
        .reset_index(drop=True)
    )

    return df if not df.empty else None


def write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    out = (
        df.copy()
        .sort_values("date")
        .drop_duplicates(subset=["date"], keep="last")
        .reset_index(drop=True)
    )

    out["date"] = _to_naive_datetime(out["date"]).dt.strftime("%d/%m/%Y")
    out.to_csv(path, sep=";", index=False)


def fetch_series_twelve(symbol: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    if not symbol:
        return None, "Ticker vuoto"

    if not TWELVE_DATA_API_KEY:
        return None, "TWELVE_DATA_API_KEY mancante"

    params = {
        "symbol": symbol,
        "interval": "1day",
        "outputsize": 5000,
        "format": "JSON",
        "order": "ASC",
        "apikey": TWELVE_DATA_API_KEY,
    }

    try:
        resp = requests.get(TWELVE_DATA_BASE_URL, params=params, timeout=40)
        resp.raise_for_status()
        data = resp.json()

        if not isinstance(data, dict):
            return None, "Risposta Twelve Data non valida"

        if data.get("status") == "error":
            return None, f"{data.get('code', '')} {data.get('message', 'errore sconosciuto')}".strip()

        values = data.get("values")
        if not values:
            return None, "Nessun valore restituito"

        df = pd.DataFrame(values)
        if "datetime" not in df.columns or "close" not in df.columns:
            return None, f"Colonne inattese: {list(df.columns)}"

        df["date"] = _to_naive_datetime(df["datetime"])
        df["close"] = pd.to_numeric(df["close"], errors="coerce")

        df = (
            df.dropna(subset=["date", "close"])
            .sort_values("date")
            .drop_duplicates(subset=["date"], keep="last")
            .reset_index(drop=True)
        )

        if df.empty:
            return None, "Serie vuota dopo pulizia"

        return df[["date", "close"]].copy(), None

    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def _normalize_yahoo_df(df: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    if df is None or df.empty:
        return None, "Nessun dato Yahoo restituito"

    df = df.reset_index()

    if "Date" in df.columns:
        df = df.rename(columns={"Date": "date"})
    elif "Datetime" in df.columns:
        df = df.rename(columns={"Datetime": "date"})
    else:
        return None, f"Colonna data Yahoo inattesa: {list(df.columns)}"

    close_col = None
    for candidate in ["Close", "Adj Close"]:
        if candidate in df.columns:
            close_col = candidate
            break

    if close_col is None:
        return None, f"Colonne Yahoo inattese: {list(df.columns)}"

    df = df.rename(columns={close_col: "close"})
    df["date"] = _to_naive_datetime(df["date"])
    df["close"] = pd.to_numeric(df["close"], errors="coerce")

    df = (
        df[["date", "close"]]
        .dropna(subset=["date", "close"])
        .sort_values("date")
        .drop_duplicates(subset=["date"], keep="last")
        .reset_index(drop=True)
    )

    if df.empty:
        return None, "Serie Yahoo vuota dopo pulizia"

    return df, None


def fetch_series_yahoo(symbol: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    if not symbol:
        return None, "Ticker Yahoo vuoto"

    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period="max", interval="1d", auto_adjust=False)
        if df is not None and not df.empty:
            log(f"  Yahoo history rows scaricate: {len(df)}")
            if len(df.index) > 0:
                log(f"  Yahoo history ultima data raw: {df.index.max()}")
            norm, err = _normalize_yahoo_df(df)
            if norm is not None:
                log(f"  Yahoo history pulite: {len(norm)} righe | ultima data {norm['date'].iloc[-1].date()}")
                return norm, None
    except Exception as e:
        log(f"  Yahoo history errore: {type(e).__name__}: {e}")

    try:
        df = yf.download(
            symbol,
            period="max",
            interval="1d",
            progress=False,
            auto_adjust=False,
            threads=False,
        )
        if df is not None and not df.empty:
            log(f"  Yahoo download rows scaricate: {len(df)}")
            if len(df.index) > 0:
                log(f"  Yahoo download ultima data raw: {df.index.max()}")
            norm, err = _normalize_yahoo_df(df)
            if norm is not None:
                log(f"  Yahoo download pulite: {len(norm)} righe | ultima data {norm['date'].iloc[-1].date()}")
                return norm, None
    except Exception as e:
        log(f"  Yahoo download errore: {type(e).__name__}: {e}")

    return None, "Yahoo non ha restituito dati utilizzabili"


def is_series_usable(df: Optional[pd.DataFrame]) -> bool:
    return df is not None and not df.empty and len(df) >= MIN_VALID_ROWS


def update_asset(name: str, cfg: dict) -> bool:
    symbol = cfg["symbol"]
    fallback_symbols = cfg.get("fallback_symbols", [])
    path = cfg["path"]
    source = cfg.get("source", "twelve")
    update_enabled = bool(cfg.get("update_enabled", True))
    force_refresh = bool(cfg.get("force_refresh", False))

    log(f"\n[{name.upper()}] {symbol or '(manuale)'} -> {path}")

    existing = None if force_refresh else read_existing_csv(path)
    if force_refresh:
        log("  Force refresh attivo: ignoro il CSV locale")
    elif existing is not None and not existing.empty:
        log(f"  CSV locale: {len(existing)} righe | ultima data {existing['date'].iloc[-1].date()}")
    else:
        log("  CSV locale assente o vuoto")

    if not update_enabled:
        log("  Aggiornamento automatico disattivato: mantengo il CSV esistente")
        return existing is not None and not existing.empty

    fresh = None
    err = None

    if source == "yahoo":
        symbols_to_try = [symbol] + [s for s in fallback_symbols if s and s != symbol]
        for sym in symbols_to_try:
            log(f"  Provo Yahoo con ticker: {sym}")
            fresh, err = fetch_series_yahoo(sym)
            if fresh is not None and not fresh.empty:
                symbol = sym
                log(f"  Yahoo OK con ticker: {sym}")
                break
            log(f"  Yahoo KO con ticker {sym}: {err}")
    elif source == "twelve":
        fresh, err = fetch_series_twelve(symbol)
    else:
        log("  Fonte non gestita")
        return existing is not None and not existing.empty

    if fresh is None or fresh.empty:
        log(f"  API non disponibile: {err}")
        if existing is not None and not existing.empty:
            log("  Fallback: mantengo dati esistenti (OK)")
            return True
        log("  Fallback fallito: nessun dato disponibile")
        return False

    if existing is not None and not existing.empty:
        existing["date"] = _to_naive_datetime(existing["date"])
    fresh["date"] = _to_naive_datetime(fresh["date"])

    if name != "ls80":
        if not is_series_usable(fresh):
            log(f"  Serie scaricata troppo corta o sospetta: {len(fresh)} righe")
            if existing is not None and not existing.empty:
                log("  Mantengo il CSV locale esistente")
                return True
            log("  Nessun fallback disponibile")
            return False
    else:
        log(f"  LS80: procedo con salvataggio forzato ({len(fresh)} righe)")

    merged = (
        pd.concat([existing, fresh], ignore_index=True)
        if existing is not None and not existing.empty
        else fresh
    )

    merged = (
        merged.drop_duplicates(subset=["date"], keep="last")
        .sort_values("date")
        .reset_index(drop=True)
    )

    if name != "ls80":
        if not is_series_usable(merged):
            log(f"  Serie finale troppo corta: {len(merged)} righe")
            if existing is not None and not existing.empty:
                log("  Mantengo il CSV locale esistente")
                return True
            return False

    write_csv(path, merged)
    log(f"  Salvato: {len(merged)} righe | ultima data {merged['date'].iloc[-1].date()} | ticker usato: {symbol}")
    return True


def main() -> None:
    log("=" * 72)
    log(f"Aggiornamento dati avviato: {now_utc()}")
    log("=" * 72)

    results = []
    for name, cfg in ASSETS.items():
        ok = update_asset(name, cfg)
        results.append(ok)

    log("=" * 72)
    log(f"Aggiornamento completato: {now_utc()}")
    log("=" * 72)

 if not all(results):
    log("ATTENZIONE: alcuni asset non aggiornati (probabile rate limit Yahoo)")
    log("Mantengo i dati esistenti → processo considerato OK")

if __name__ == "__main__":
    main()
