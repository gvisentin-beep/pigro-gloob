from __future__ import annotations

import os
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
import yfinance as yf

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

LS80_TICKER = os.getenv("LS80_TICKER", "VNGA80.MI").strip()
GOLD_TICKER = os.getenv("GOLD_TICKER", "SGLD.MI").strip()
WORLD_TICKER = os.getenv("WORLD_TICKER", "SMSWLD.MI").strip()

LS80_FILE = DATA_DIR / "ls80.csv"
GOLD_FILE = DATA_DIR / "gold.csv"
WORLD_FILE = DATA_DIR / "world.csv"


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _read_existing(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
      return None

    try:
        df = pd.read_csv(path, sep=";", dtype=str, encoding="utf-8-sig")
        df.columns = [str(c).strip().lower() for c in df.columns]

        if "date" not in df.columns or "close" not in df.columns:
            df = pd.read_csv(path, sep=";", dtype=str, header=None, names=["date", "close"], encoding="utf-8-sig")

        df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
        df["close"] = pd.to_numeric(df["close"].astype(str).str.replace(",", ".", regex=False), errors="coerce")
        df = df.dropna(subset=["date", "close"]).sort_values("date")
        return df[["date", "close"]]
    except Exception:
        return None


def _download_asset(ticker: str) -> Optional[pd.DataFrame]:
    try:
        raw = yf.download(ticker, period="15y", interval="1d", auto_adjust=False, progress=False)
        if raw is None or raw.empty:
            return None

        tmp = raw.copy()
        if isinstance(tmp.index, pd.DatetimeIndex):
            tmp = tmp.reset_index()

        if "Date" in tmp.columns:
            tmp = tmp.rename(columns={"Date": "date"})

        if "Adj Close" in tmp.columns:
            tmp["close"] = tmp["Adj Close"]
        elif "Close" in tmp.columns:
            tmp["close"] = tmp["Close"]
        else:
            return None

        tmp["date"] = pd.to_datetime(tmp["date"], errors="coerce").dt.normalize()
        tmp["close"] = pd.to_numeric(tmp["close"], errors="coerce")
        tmp = tmp.dropna(subset=["date", "close"]).sort_values("date")
        return tmp[["date", "close"]]
    except Exception:
        return None


def _write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy().sort_values("date")
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%d/%m/%Y")
    out.to_csv(path, sep=";", index=False)


def _update_one(ticker: str, path: Path) -> dict:
    result = {"asset": path.stem, "ticker_used": ticker, "updated": False, "file": str(path)}

    old_df = _read_existing(path)
    new_df = _download_asset(ticker)

    if new_df is None or new_df.empty:
        result["reason"] = "no_data_from_yahoo"
        return result

    if old_df is not None and not old_df.empty:
        merged = pd.concat([old_df, new_df], ignore_index=True)
    else:
        merged = new_df

    merged = merged.drop_duplicates(subset=["date"], keep="last").sort_values("date").reset_index(drop=True)
    _write_csv(path, merged)

    result["updated"] = True
    result["rows"] = int(len(merged))
    result["last_date"] = str(merged["date"].iloc[-1].date())
    result["last_value"] = float(merged["close"].iloc[-1])
    return result


def main():
    print(f"[{_now_iso()}] Aggiornamento dati...")
    print(_update_one(LS80_TICKER, LS80_FILE))
    print(_update_one(GOLD_TICKER, GOLD_FILE))
    print(_update_one(WORLD_TICKER, WORLD_FILE))


if __name__ == "__main__":
    main()
