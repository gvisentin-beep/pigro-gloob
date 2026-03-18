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

LS80_FALLBACK_TICKERS = [
    t.strip()
    for t in [
        os.getenv("LS80_TICKER", "IE00BMVB5R75.MI"),
        "VNGA80.MI",
        "V80A.MI",
    ]
    if t and t.strip()
]

ASSETS = {
    "ls80": {"symbol": os.getenv("LS80_TICKER", "IE00BMVB5R75.MI").strip(), "path": DATA_DIR / "ls80.csv"},
    "gold": {"symbol": os.getenv("GOLD_TICKER", "GLD").strip(), "path": DATA_DIR / "gold.csv"},
    "btc": {"symbol": os.getenv("BTC_TICKER", "BTC/EUR").strip(), "path": DATA_DIR / "btc.csv"},
    "world": {"symbol": os.getenv("WORLD_TICKER", "URTH").strip(), "path": DATA_DIR / "world.csv"},
}


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def log(msg: str) -> None:
    print(msg, flush=True)


def read_existing_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None

    try:
        df = pd.read_csv(path, sep=";", dtype=str, encoding="utf-8-sig")
        df.columns = [str(c).strip().lower() for c in df.columns]
        if "date" not in df.columns or "close" not in df.columns:
            df = pd.read_csv(path, sep=";", header=None, names=["date", "close"], dtype=str, encoding="utf-8-sig")
    except Exception:
        df = pd.read_csv(path, sep=";", header=None, names=["date", "close"], dtype=str, encoding="utf-8-sig")

    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
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
    out = df.copy().sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%d/%m/%Y")
    out.to_csv(path, sep=";", index=False)


def fetch_series(symbol: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
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

        df["date"] = pd.to_datetime(df["datetime"], errors="coerce")
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


def _flatten_yf_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            "_".join([str(x) for x in tup if str(x) != ""]).strip("_")
            for tup in df.columns.to_flat_index()
        ]
    return df


def fetch_yahoo(symbol: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    if not symbol:
        return None, "Ticker vuoto"

    try:
        df = yf.download(
            symbol,
            period="max",
            interval="1d",
            progress=False,
            auto_adjust=False,
            group_by="column",
            threads=False,
        )

        if df is None or df.empty:
            return None, f"Yahoo vuoto per {symbol}"

        df = _flatten_yf_columns(df)
        df = df.reset_index()

        date_col = None
        for c in df.columns:
            if str(c).lower() in ("date", "datetime"):
                date_col = c
                break

        close_col = None
        for c in df.columns:
            if str(c).lower() == "close" or str(c).lower().startswith("close_"):
                close_col = c
                break

        if date_col is None or close_col is None:
            return None, f"Colonne Yahoo inattese per {symbol}: {list(df.columns)}"

        out = df[[date_col, close_col]].copy()
        out.columns = ["date", "close"]
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        out["close"] = pd.to_numeric(out["close"], errors="coerce")
        out = (
            out.dropna(subset=["date", "close"])
            .sort_values("date")
            .drop_duplicates(subset=["date"], keep="last")
            .reset_index(drop=True)
        )
        if out.empty:
            return None, f"Serie Yahoo vuota dopo pulizia per {symbol}"
        return out[["date", "close"]].copy(), None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def fetch_ls80_with_fallback() -> Tuple[Optional[pd.DataFrame], Optional[str], Optional[str]]:
    errors: list[str] = []

    for ticker in LS80_FALLBACK_TICKERS:
        df, err = fetch_yahoo(ticker)
        if df is not None and not df.empty:
            return df, None, ticker
        errors.append(f"{ticker}: {err}")

    return None, " | ".join(errors) if errors else "nessun ticker LS80 disponibile", None


def update_asset(name: str, symbol: str, path: Path) -> bool:
    log(f"\n[{name.upper()}] {symbol} -> {path}")

    existing = read_existing_csv(path)
    if existing is not None:
        log(f"  CSV locale: {len(existing)} righe | ultima data {existing['date'].iloc[-1].date()}")
    else:
        log("  CSV locale assente o vuoto")

    if name == "ls80":
        fresh, err, used_ticker = fetch_ls80_with_fallback()
        log(f"  Fonte: Yahoo | ticker usato: {used_ticker or symbol}")
    else:
        fresh, err = fetch_series(symbol)
        log(f"  Fonte: Twelve Data | ticker usato: {symbol}")

    if fresh is None or fresh.empty:
        log(f"  API non disponibile: {err}")
        if existing is not None and not existing.empty:
            log("  Mantengo il CSV locale esistente")
            return True
        log("  Nessun fallback disponibile")
        return False

    merged = pd.concat([existing, fresh], ignore_index=True) if existing is not None and not existing.empty else fresh
    merged = (
        merged.drop_duplicates(subset=["date"], keep="last")
        .sort_values("date")
        .reset_index(drop=True)
    )
    write_csv(path, merged)
    log(f"  Salvato: {len(merged)} righe | ultima data {merged['date'].iloc[-1].date()}")
    return True


def main() -> None:
    log("=" * 72)
    log(f"Aggiornamento dati avviato: {now_utc()}")
    log("=" * 72)

    results = []
    for name, cfg in ASSETS.items():
        ok = update_asset(name, cfg["symbol"], cfg["path"])
        results.append(ok)

    log("=" * 72)
    log(f"Aggiornamento completato: {now_utc()}")
    log("=" * 72)

    if not all(results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
