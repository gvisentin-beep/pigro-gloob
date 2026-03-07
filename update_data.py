from __future__ import annotations

import os
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, Tuple

import pandas as pd
import requests

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

TWELVE_DATA_API_KEY = os.getenv("TWELVE_DATA_API_KEY", "").strip()
TWELVE_DATA_BASE_URL = "https://api.twelvedata.com/time_series"

ASSETS = {
    "ls80": {
        "symbol": "VNGA80.MI",
        "path": DATA_DIR / "ls80.csv",
    },
    "gold": {
        "symbol": "GLD",
        "path": DATA_DIR / "gold.csv",
    },
    "world": {
        "symbol": "SMSWLD.MI",
        "path": DATA_DIR / "world.csv",
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
                dtype=str,
                header=None,
                names=["date", "close"],
                encoding="utf-8-sig",
            )

        df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
        df["close"] = pd.to_numeric(
            df["close"].astype(str).str.strip().str.replace(",", ".", regex=False),
            errors="coerce",
        )

        df = df.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
        if df.empty:
            return None

        return df[["date", "close"]].copy()

    except Exception as e:
        log(f"Errore lettura CSV {path.name}: {e}")
        return None


def write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    out = df.copy().sort_values("date").reset_index(drop=True)
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%d/%m/%Y")
    out.to_csv(path, sep=";", index=False)


def fetch_twelve_data(symbol: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    if not TWELVE_DATA_API_KEY:
        return None, "TWELVE_DATA_API_KEY mancante"

    params = {
        "symbol": symbol,
        "interval": "1day",
        "outputsize": 5000,
        "format": "JSON",
        "apikey": TWELVE_DATA_API_KEY,
        "order": "ASC",
    }

    try:
        resp = requests.get(TWELVE_DATA_BASE_URL, params=params, timeout=40)
        resp.raise_for_status()
        data = resp.json()

        if not isinstance(data, dict):
            return None, "Risposta Twelve Data non valida"

        if data.get("status") == "error":
            return None, f"{data.get('code', '')} {data.get('message', 'errore sconosciuto')}"

        values = data.get("values")
        if not values:
            return None, "Nessun valore restituito"

        df = pd.DataFrame(values)

        if "datetime" not in df.columns or "close" not in df.columns:
            return None, f"Colonne inattese: {list(df.columns)}"

        df["date"] = pd.to_datetime(df["datetime"], errors="coerce")
        df["close"] = pd.to_numeric(df["close"], errors="coerce")

        df = df.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
        if df.empty:
            return None, "Serie vuota dopo pulizia"

        return df[["date", "close"]].copy(), None

    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def update_one_asset(name: str, symbol: str, path: Path) -> dict:
    result = {
        "asset": name,
        "symbol": symbol,
        "file": str(path),
        "updated": False,
    }

    old_df = read_existing_csv(path)
    new_df, err = fetch_twelve_data(symbol)

    if new_df is None or new_df.empty:
        result["reason"] = err or "no_data_from_twelve_data"
        return result

    if old_df is not None and not old_df.empty:
        merged = pd.concat([old_df, new_df], ignore_index=True)
    else:
        merged = new_df

    merged = (
        merged.drop_duplicates(subset=["date"], keep="last")
        .sort_values("date")
        .reset_index(drop=True)
    )

    write_csv(path, merged)

    result["updated"] = True
    result["rows"] = int(len(merged))
    result["first_date"] = str(merged["date"].iloc[0].date())
    result["last_date"] = str(merged["date"].iloc[-1].date())
    result["first_value"] = float(merged["close"].iloc[0])
    result["last_value"] = float(merged["close"].iloc[-1])
    return result


def main() -> None:
    log(f"[{now_utc()}] Aggiornamento dati con Twelve Data...")

    all_ok = True

    for name, cfg in ASSETS.items():
        res = update_one_asset(name, cfg["symbol"], cfg["path"])
        log(str(res))
        if not res.get("updated"):
            all_ok = False

    log(f"[{now_utc()}] Fine aggiornamento.")

    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
