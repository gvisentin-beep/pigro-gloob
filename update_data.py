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
        "symbol": "",
        "path": DATA_DIR / "ls80.csv",
        "source": "manual",
        "update_enabled": False,
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
        "source": "twelve",
        "update_enabled": True,
    },
    "mib": {
        "symbol": "VGK",
        "path": DATA_DIR / "mib.csv",
        "source": "yahoo",
        "update_enabled": True,
    },
    "sp500": {
        "symbol": "",
        "path": DATA_DIR / "sp500.csv",
        "source": "manual",
        "update_enabled": False,
    },
}


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def log(msg: str) -> None:
    print(msg, flush=True)


def read_existing_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None

    try:
        text = path.read_text(encoding="utf-8-sig").strip()
    except Exception:
        text = ""

    if not text:
        return None

    try:
        first_line = text.splitlines()[0].strip().lower()
        if ";" in first_line:
            df = pd.read_csv(path, sep=";", dtype=str, encoding="utf-8-sig")
        elif "," in first_line:
            df = pd.read_csv(path, sep=",", dtype=str, encoding="utf-8-sig")
        else:
            df = pd.read_csv(
                path,
                sep=r"\s+",
                engine="python",
                dtype=str,
                encoding="utf-8-sig",
            )
    except Exception:
        return None

    df.columns = [str(c).strip().lower() for c in df.columns]

    if "date" not in df.columns or "close" not in df.columns:
        if len(df.columns) >= 2:
            df = df.iloc[:, :2].copy()
            df.columns = ["date", "close"]
        else:
            return None

    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    df["close"] = (
        df["close"]
        .astype(str)
        .str.strip()
        .str.replace(",", ".", regex=False)
    )
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

    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    out.to_csv(path, sep=",", index=False)


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

        df["date"] = pd.to_datetime(df["datetime"], errors="coerce")
        df["date"] = df["date"].dt.tz_localize(None)
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


def fetch_series_yahoo(symbol: str, start_date: str = "2022-06-02") -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    if not symbol:
        return None, "Ticker Yahoo vuoto"

    try:
        df = yf.download(
            symbol,
            start=start_date,
            interval="1d",
            progress=False,
            auto_adjust=False,
            threads=False,
        )

        if df is None or df.empty:
            return None, "Nessun dato Yahoo restituito"

        df = df.reset_index()

        if "Date" not in df.columns:
            return None, f"Colonne inattese: {list(df.columns)}"

        close_col = "Adj Close" if "Adj Close" in df.columns else "Close"
        if close_col not in df.columns:
            return None, f"Colonne inattese: {list(df.columns)}"

        df = df.rename(columns={"Date": "date", close_col: "close"})
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["date"] = df["date"].dt.tz_localize(None)
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

    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def is_series_usable(df: Optional[pd.DataFrame]) -> bool:
    return df is not None and not df.empty and len(df) >= MIN_VALID_ROWS


def update_asset(name: str, cfg: dict) -> bool:
    symbol = cfg["symbol"]
    path = cfg["path"]
    source = cfg.get("source", "twelve")
    update_enabled = bool(cfg.get("update_enabled", True))

    log(f"\n[{name.upper()}] {symbol or '(manuale)'} -> {path}")

    existing = read_existing_csv(path)
    if existing is not None and not existing.empty:
        log(f"  CSV locale: {len(existing)} righe | ultima data {existing['date'].iloc[-1].date()}")
    else:
        log("  CSV locale assente o vuoto")

    if not update_enabled:
        log("  Aggiornamento automatico disattivato: mantengo il CSV esistente")
        return existing is not None and not existing.empty

    if source == "twelve":
        fresh, err = fetch_series_twelve(symbol)
    elif source == "yahoo":
        fresh, err = fetch_series_yahoo(symbol, start_date="2022-06-02")
    else:
        log("  Fonte non gestita in questa versione")
        return existing is not None and not existing.empty

    if fresh is None or fresh.empty:
        log(f"  API non disponibile: {err}")
        if existing is not None and not existing.empty:
            log("  Fallback: mantengo dati esistenti (OK)")
            return True
        log("  Fallback fallito: nessun dato disponibile")
        return False

    if not is_series_usable(fresh):
        log(f"  Serie scaricata troppo corta o sospetta: {len(fresh)} righe")
        if existing is not None and not existing.empty:
            log("  Mantengo il CSV locale esistente")
            return True
        log("  Nessun fallback disponibile")
        return False

    if source == "yahoo":
        merged = fresh.copy()
    else:
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

    if not is_series_usable(merged):
        log(f"  Serie finale troppo corta: {len(merged)} righe")
        if existing is not None and not existing.empty:
            log("  Mantengo il CSV locale esistente")
            return True
        return False

    write_csv(path, merged)
    log(f"  Salvato: {len(merged)} righe | ultima data {merged['date'].iloc[-1].date()}")
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
        raise SystemExit(1)


if __name__ == "__main__":
    main()
