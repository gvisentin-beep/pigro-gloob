from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
import yfinance as yf

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

ASSETS = {
    "ls80": ("VNGA80.MI", DATA_DIR / "ls80.csv"),
    "gold": ("SGLD.MI", DATA_DIR / "gold.csv"),
    "world": ("SMSWLD.MI", DATA_DIR / "world.csv"),
}


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


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

    except Exception:
        return None


def download_asset_history(ticker: str) -> Optional[pd.DataFrame]:
    try:
        raw = yf.download(
            ticker,
            period="20y",
            interval="1d",
            auto_adjust=False,
            progress=False,
        )

        if raw is None or raw.empty:
            return None

        df = raw.copy()
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()

        if "Date" in df.columns:
            df = df.rename(columns={"Date": "date"})

        if "Adj Close" in df.columns:
            df["close"] = df["Adj Close"]
        elif "Close" in df.columns:
            df["close"] = df["Close"]
        else:
            return None

        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
        df["close"] = pd.to_numeric(df["close"], errors="coerce")

        df = df.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
        if df.empty:
            return None

        return df[["date", "close"]].copy()

    except Exception:
        return None


def write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    out = df.copy().sort_values("date").reset_index(drop=True)
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%d/%m/%Y")

    out.to_csv(path, sep=";", index=False)


def update_one_asset(name: str, ticker: str, path: Path) -> dict:
    result = {
        "asset": name,
        "ticker_used": ticker,
        "file": str(path),
        "updated": False,
    }

    old_df = read_existing_csv(path)
    new_df = download_asset_history(ticker)

    if new_df is None or new_df.empty:
        result["reason"] = "no_data_from_yahoo"
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
    print(f"[{now_utc()}] Aggiornamento dati...")

    for name, (ticker, path) in ASSETS.items():
        res = update_one_asset(name, ticker, path)
        print(res)

    print(f"[{now_utc()}] Fine aggiornamento.")


if __name__ == "__main__":
    main()
