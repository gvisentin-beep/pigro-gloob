#!/usr/bin/env python3
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


def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index()
        df.rename(columns={"Date": "date"}, inplace=True)

    if "Adj Close" in df.columns:
        df["close"] = df["Adj Close"]
    elif "Close" in df.columns:
        df["close"] = df["Close"]
    elif "close" not in df.columns:
        raise ValueError("yfinance returned no Close/Adj Close column")

    out = df[["date", "close"]].copy()
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    out = out.dropna(subset=["date", "close"]).sort_values("date")
    out["date_str"] = out["date"].dt.strftime("%d/%m/%Y")
    return out[["date", "date_str", "close"]]


def _read_existing(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    df = pd.read_csv(path, sep=";", dtype=str)
    if "date" not in df.columns or "close" not in df.columns:
        return None
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["date", "close"]).sort_values("date")
    df["date_str"] = df["date"].dt.strftime("%d/%m/%Y")
    return df[["date", "date_str", "close"]]


def _write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy().sort_values("date")
    out2 = out[["date_str", "close"]].rename(columns={"date_str": "date"})
    out2.to_csv(path, sep=";", index=False, float_format="%.6g")


def _update_one(ticker: str, path: Path) -> bool:
    raw = yf.download(ticker, period="60d", interval="1d", auto_adjust=False, progress=False)
    if raw is None or raw.empty:
        return False

    new_df = _normalize_df(raw)
    old_df = _read_existing(path)

    if old_df is not None:
        merged = pd.concat([old_df, new_df], ignore_index=True)
    else:
        merged = new_df

    merged = merged.drop_duplicates(subset=["date"], keep="last").sort_values("date")
    _write_csv(path, merged)
    return True


def main() -> int:
    ok_ls = _update_one(LS80_TICKER, LS80_FILE)
    ok_gd = _update_one(GOLD_TICKER, GOLD_FILE)
    ok_wd = _update_one(WORLD_TICKER, WORLD_FILE)

    print(f"[{_now_iso()}] updated ls80={ok_ls} gold={ok_gd} world={ok_wd}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
