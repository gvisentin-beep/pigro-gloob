from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

MIN_VALID_ROWS = 50

ASSETS = {
    "ls80": {
        "symbol": "",
        "path": DATA_DIR / "ls80.csv",
        "source": "manual",
        "update_enabled": False,
    },
    "gold": {
        "symbol": "GLD",
        "path": DATA_DIR / "gold.csv",
        "source": "yahoo",
        "update_enabled": True,
    },
    "btc": {
        "symbol": "BTC-EUR",
        "path": DATA_DIR / "btc.csv",
        "source": "yahoo",
        "update_enabled": True,
    },
    "world": {
        "symbol": "URTH",
        "path": DATA_DIR / "world.csv",
        "source": "yahoo",
        "update_enabled": True,
    },
    "mib": {
        "symbol": "MSE.PA",  # benchmark Europa
        "path": DATA_DIR / "mib.csv",
        "source": "yahoo",
        "update_enabled": True,
    },
    "sp500": {
        "symbol": "^GSPC",
        "path": DATA_DIR / "sp500.csv",
        "source": "yahoo",
        "update_enabled": True,
    },
}


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

    except Exception:
        return None


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


def extract_close_series(df: pd.DataFrame) -> Optional[pd.Series]:
    if df is None or df.empty:
        return None

    if isinstance(df.columns, pd.MultiIndex):
        close_cols = [col for col in df.columns if str(col[0]).lower() == "close"]
        if not close_cols:
            return None
        return df[close_cols[0]]

    if "Close" in df.columns:
        return df["Close"]

    return None


def clean_downloaded_series(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    close_series = extract_close_series(df)
    if close_series is None:
        return None

    out = pd.DataFrame(
        {
            "date": pd.to_datetime(df.index, errors="coerce"),
            "close": pd.to_numeric(close_series, errors="coerce"),
        }
    )

    out = (
        out.dropna(subset=["date", "close"])
        .sort_values("date")
        .drop_duplicates(subset=["date"], keep="last")
        .reset_index(drop=True)
    )

    if out.empty:
        return None

    # Filtro leggero anti-anomalie: elimina solo spike davvero estremi
    # e isola un singolo punto palesemente errato rispetto ai vicini.
    if len(out) >= 5:
        out["ret"] = out["close"].pct_change()
        spike_idx = []

        for i in range(1, len(out) - 1):
            r = out.at[i, "ret"]
            prev_close = out.at[i - 1, "close"]
            cur_close = out.at[i, "close"]
            next_close = out.at[i + 1, "close"]

            if not pd.notna(r):
                continue

            # Candidato spike: variazione enorme in un giorno
            if abs(r) > 0.35:
                # Se il punto successivo torna vicino al precedente,
                # probabile anomalia tecnica del dato.
                if prev_close > 0:
                    rebound = abs((next_close / prev_close) - 1)
                    if rebound < 0.08:
                        spike_idx.append(i)

            # Prezzi non positivi o nulli non devono passare
            if cur_close <= 0:
                spike_idx.append(i)

        if spike_idx:
            out = out.drop(index=sorted(set(spike_idx))).reset_index(drop=True)

        out = out.drop(columns=["ret"], errors="ignore")

    return out if not out.empty else None


def fetch_yahoo(symbol: str):
    try:
        time.sleep(2)

        df = yf.download(
            symbol,
            period="max",
            interval="1d",
            progress=False,
            threads=False,
            auto_adjust=True,
        )

        if df is None or df.empty:
            return None, "Yahoo vuoto"

        out = clean_downloaded_series(df)
        if out is None or out.empty:
            return None, "Serie Yahoo vuota dopo pulizia"

        return out, None

    except Exception as e:
        return None, str(e)


def update_asset(name: str, cfg: dict) -> bool:
    symbol = cfg["symbol"]
    path = cfg["path"]
    update_enabled = bool(cfg.get("update_enabled", True))

    log(f"\n[{name.upper()}] {symbol or '(manuale)'}")

    existing = read_existing_csv(path)
    if existing is not None and not existing.empty:
        log(f"CSV locale: {len(existing)} righe | ultima data {existing['date'].iloc[-1].date()}")
    else:
        log("CSV locale assente o vuoto")

    if not update_enabled:
        log("Aggiornamento disattivato: mantengo il CSV esistente")
        return True

    fresh, err = fetch_yahoo(symbol)

    if fresh is None:
        log(f"⚠️ Yahoo bloccato o errore: {err}")
        if existing is not None and not existing.empty:
            log("Uso dati esistenti (fallback OK)")
            return True
        log("Nessun dato disponibile, ma non blocco il workflow")
        return True

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

    if len(merged) < MIN_VALID_ROWS:
        log(f"Serie troppo corta ({len(merged)} righe), mantengo eventuali dati esistenti")
        if existing is not None and not existing.empty:
            return True
        return True

    write_csv(path, merged)
    log(f"Salvato: {len(merged)} righe | ultima data {merged['date'].iloc[-1].date()}")
    return True


def main() -> None:
    log("=== AGGIORNAMENTO DATI ===")

    for name, cfg in ASSETS.items():
        update_asset(name, cfg)

    log("=== FINE ===")


if __name__ == "__main__":
    main()
