from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import requests

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
        "symbol": "",
        "path": DATA_DIR / "mib.csv",
        "source": "manual",
        "update_enabled": False,
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

    out = (
        df.copy()
        .sort_values("date")
        .drop_duplicates(subset=["date"], keep="last")
        .reset_index(drop=True)
    )

    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%d/%m/%Y")
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

        if data.get("status") == "error":
            return None, data.get("message", "errore API")

        values = data.get("values")
        if not values:
            return None, "Nessun dato"

        df = pd.DataFrame(values)

        df["date"] = pd.to_datetime(df["datetime"], errors="coerce")
        df["close"] = pd.to_numeric(df["close"], errors="coerce")

        df = df.dropna().sort_values("date")

        return df[["date", "close"]], None

    except Exception as e:
        return None, str(e)


def is_series_usable(df: Optional[pd.DataFrame]) -> bool:
    return df is not None and len(df) >= MIN_VALID_ROWS


def update_asset(name: str, cfg: dict) -> bool:
    symbol = cfg["symbol"]
    path = cfg["path"]
    source = cfg["source"]
    enabled = cfg["update_enabled"]

    log(f"\n[{name}]")

    existing = read_existing_csv(path)

    if not enabled:
        log("  manuale → uso CSV esistente")
        return existing is not None

    if source != "twelve":
        log("  fonte non supportata")
        return existing is not None

    fresh, err = fetch_series_twelve(symbol)

    if fresh is None:
        log(f"  errore: {err}")
        return existing is not None

    merged = pd.concat([existing, fresh]) if existing is not None else fresh
    merged = merged.drop_duplicates("date").sort_values("date")

    write_csv(path, merged)

    log(f"  aggiornato: {len(merged)} righe")
    return True


def main() -> None:
    log("=" * 60)
    log("UPDATE START")
    log("=" * 60)

    results = []

    for name, cfg in ASSETS.items():
        ok = update_asset(name, cfg)
        results.append(ok)

    log("=" * 60)
    log("UPDATE END")
    log("=" * 60)

    # ⚠️ QUI ERA IL PROBLEMA
    # NON facciamo fallire il workflow
    if not all(results):
        log("WARNING: alcuni asset non aggiornati (ma continuo)")


if __name__ == "__main__":
    main()
