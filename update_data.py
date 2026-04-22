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
        "symbol": "MSE.PA",
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

# soglie più severe
MAX_DAILY_MOVE_DEFAULT = 0.18   # 18%
MAX_DAILY_MOVE_BTC = 0.30       # BTC più volatile
REBOUND_TOLERANCE = 0.06        # 6%
LOCAL_WINDOW = 3                # controllo locale più robusto


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


def robust_zscore_flags(series: pd.Series, window: int = 15, z_thresh: float = 5.0) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    med = s.rolling(window=window, center=True, min_periods=5).median()
    mad = (s - med).abs().rolling(window=window, center=True, min_periods=5).median()

    # 1.4826 * MAD ≈ std robusta
    robust_sigma = 1.4826 * mad
    z = (s - med).abs() / robust_sigma.replace(0, pd.NA)
    flags = z > z_thresh
    return flags.fillna(False)


def detect_isolated_spikes(df: pd.DataFrame, symbol: str) -> list[int]:
    close = pd.to_numeric(df["close"], errors="coerce")
    ret = close.pct_change()
    n = len(df)

    max_daily = MAX_DAILY_MOVE_BTC if symbol.upper() == "BTC-EUR" else MAX_DAILY_MOVE_DEFAULT
    flags = set()

    robust_flags = robust_zscore_flags(close, window=15, z_thresh=5.0)

    for i in range(1, n - 1):
        prev_close = close.iloc[i - 1]
        cur_close = close.iloc[i]
        next_close = close.iloc[i + 1]

        if not pd.notna(prev_close) or not pd.notna(cur_close) or not pd.notna(next_close):
            continue

        if cur_close <= 0:
            flags.add(i)
            continue

        r = ret.iloc[i]
        if pd.notna(r) and abs(r) > max_daily:
            flags.add(i)
            continue

        # punto isolato: il giorno dopo torna vicino al giorno prima
        if prev_close > 0:
            rebound = abs((next_close / prev_close) - 1)
            one_day_jump = abs((cur_close / prev_close) - 1)
            next_day_jump = abs((next_close / cur_close) - 1)

            if one_day_jump > max_daily and rebound < REBOUND_TOLERANCE:
                flags.add(i)
                continue

            # spike a V molto evidente
            if one_day_jump > max_daily * 0.8 and next_day_jump > max_daily * 0.8 and rebound < REBOUND_TOLERANCE:
                flags.add(i)
                continue

        # controllo locale su finestra
        left = max(0, i - LOCAL_WINDOW)
        right = min(n, i + LOCAL_WINDOW + 1)
        local = close.iloc[left:right].drop(index=df.index[i], errors="ignore")

        if len(local) >= 3:
            local_med = local.median()
            if pd.notna(local_med) and local_med > 0:
                dev = abs((cur_close / local_med) - 1)
                if dev > max_daily * 1.2:
                    flags.add(i)
                    continue

        # robust z-score
        if bool(robust_flags.iloc[i]):
            flags.add(i)

    return sorted(flags)


def smooth_spikes_by_interpolation(df: pd.DataFrame, spike_idx: list[int]) -> pd.DataFrame:
    if not spike_idx:
        return df

    out = df.copy().reset_index(drop=True)
    out.loc[spike_idx, "close"] = pd.NA
    out["close"] = pd.to_numeric(out["close"], errors="coerce").interpolate(
        method="linear",
        limit_direction="both"
    )

    out = (
        out.dropna(subset=["date", "close"])
        .sort_values("date")
        .drop_duplicates(subset=["date"], keep="last")
        .reset_index(drop=True)
    )
    return out


def second_pass_return_filter(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if df.empty:
        return df

    close = pd.to_numeric(df["close"], errors="coerce")
    ret = close.pct_change().abs()

    max_daily = MAX_DAILY_MOVE_BTC if symbol.upper() == "BTC-EUR" else MAX_DAILY_MOVE_DEFAULT
    bad_idx = ret[ret > max_daily].index.tolist()

    if not bad_idx:
        return df

    out = df.copy()
    out.loc[bad_idx, "close"] = pd.NA
    out["close"] = pd.to_numeric(out["close"], errors="coerce").interpolate(
        method="linear",
        limit_direction="both"
    )

    out = (
        out.dropna(subset=["date", "close"])
        .sort_values("date")
        .drop_duplicates(subset=["date"], keep="last")
        .reset_index(drop=True)
    )
    return out


def clean_downloaded_series(df: pd.DataFrame, symbol: str) -> Optional[pd.DataFrame]:
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

    removed_total = 0

    # 1° passaggio: spike isolati
    spike_idx = detect_isolated_spikes(out, symbol)
    removed_total += len(spike_idx)
    out = smooth_spikes_by_interpolation(out, spike_idx)

    # 2° passaggio: eventuali residui con filtro su ritorni
    before_len = len(out)
    out = second_pass_return_filter(out, symbol)
    after_len = len(out)
    # qui non togliamo righe, ma “ripariamo” valori; il conteggio è indicativo
    if before_len == after_len:
        pass

    # 3° passaggio: pulizia finale
    out = (
        out.dropna(subset=["date", "close"])
        .sort_values("date")
        .drop_duplicates(subset=["date"], keep="last")
        .reset_index(drop=True)
    )

    if out.empty:
        return None

    log(f"  Pulizia {symbol}: spike isolati corretti = {removed_total}")
    return out


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

        out = clean_downloaded_series(df, symbol)
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
        log(f"  CSV locale: {len(existing)} righe | ultima data {existing['date'].iloc[-1].date()}")
    else:
        log("  CSV locale assente o vuoto")

    if not update_enabled:
        log("  Aggiornamento disattivato: mantengo il CSV esistente")
        return True

    fresh, err = fetch_yahoo(symbol)

    if fresh is None:
        log(f"  ⚠️ Yahoo bloccato o errore: {err}")
        if existing is not None and not existing.empty:
            log("  Uso dati esistenti (fallback OK)")
            return True
        log("  Nessun dato disponibile, ma non blocco il workflow")
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
        log(f"  Serie troppo corta ({len(merged)} righe), mantengo eventuali dati esistenti")
        return True

    write_csv(path, merged)
    log(f"  Salvato: {len(merged)} righe | ultima data {merged['date'].iloc[-1].date()}")
    return True


def main() -> None:
    log("=== AGGIORNAMENTO DATI ===")

    for name, cfg in ASSETS.items():
        update_asset(name, cfg)

    log("=== FINE ===")


if __name__ == "__main__":
    main()
