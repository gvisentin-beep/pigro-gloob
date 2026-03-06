from __future__ import annotations

import os
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import math
import pandas as pd
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

LS80_FILE = DATA_DIR / "ls80.csv"
GOLD_FILE = DATA_DIR / "gold.csv"
WORLD_FILE = DATA_DIR / "world.csv"

LS80_TICKER = os.getenv("LS80_TICKER", "VNGA80.MI").strip()
GOLD_TICKER = os.getenv("GOLD_TICKER", "SGLD.MI").strip()
WORLD_TICKER = os.getenv("WORLD_TICKER", "SMSWLD.MI").strip()


# ----------------------------------------------------------
# Helpers
# ----------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _json_error(msg: str, status: int = 400, **extra: Any):
    payload = {"ok": False, "error": msg}
    payload.update(extra)
    return jsonify(payload), status


# ----------------------------------------------------------
# CSV reader ROBUSTO
# Supporta:
# - separatore ;
# - intestazione date;close
# - spazi/maiuscole nelle colonne
# - numeri con virgola o punto
# ----------------------------------------------------------

def read_price_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(str(path))

    # Prima prova: con intestazione
    try:
        df = pd.read_csv(path, sep=";", dtype=str, encoding="utf-8-sig")
        df.columns = [str(c).strip().lower() for c in df.columns]

        if "date" in df.columns and "close" in df.columns:
            df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")

            df["close"] = (
                df["close"]
                .astype(str)
                .str.strip()
                .str.replace(",", ".", regex=False)
            )
            df["close"] = pd.to_numeric(df["close"], errors="coerce")

            df = df.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)

            if not df.empty:
                return df[["date", "close"]].copy()
    except Exception:
        pass

    # Seconda prova: senza intestazione
    df = pd.read_csv(
        path,
        sep=";",
        dtype=str,
        header=None,
        names=["date", "close"],
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

    df = df.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)

    if df.empty:
        raise ValueError(f"CSV non valido: {path}")

    return df[["date", "close"]].copy()


# ----------------------------------------------------------
# Metriche
# ----------------------------------------------------------

def compute_cagr(series: pd.Series, dates: pd.Series) -> float:
    if len(series) < 2:
        return 0.0

    years = (dates.iloc[-1] - dates.iloc[0]).days / 365.25
    if years <= 0:
        return 0.0

    start = float(series.iloc[0])
    end = float(series.iloc[-1])

    if start <= 0:
        return 0.0

    return (end / start) ** (1.0 / years) - 1.0


def compute_max_dd(series: pd.Series) -> float:
    peak = series.cummax()
    dd = series / peak - 1.0
    return float(dd.min())


def compute_drawdown_series_pct(series: pd.Series) -> pd.Series:
    peak = series.cummax()
    dd = series / peak - 1.0
    return dd * 100.0


def doubling_years(cagr: float) -> Optional[float]:
    if cagr <= 0:
        return None
    return math.log(2.0) / math.log(1.0 + cagr)


# ----------------------------------------------------------
# 3 peggiori drawdown reali
# ----------------------------------------------------------

def worst_drawdowns(series: pd.Series, dates: pd.Series, n: int = 3) -> list[dict]:
    peak = series.cummax()
    dd = series / peak - 1.0

    events = []
    in_dd = False
    start_idx = None

    for i in range(1, len(series)):
        if not in_dd and dd.iloc[i] < 0:
            start_idx = i - 1
            in_dd = True

        if in_dd and dd.iloc[i] == 0:
            sub = dd.iloc[start_idx:i]
            if len(sub) > 0:
                bottom = sub.idxmin()
                events.append(
                    {
                        "start": dates.iloc[start_idx].strftime("%Y-%m-%d"),
                        "bottom": dates.iloc[bottom].strftime("%Y-%m-%d"),
                        "end": dates.iloc[i].strftime("%Y-%m-%d"),
                        "depth_pct": float(sub.min() * 100.0),
                    }
                )
            in_dd = False
            start_idx = None

    if in_dd and start_idx is not None:
        sub = dd.iloc[start_idx:]
        if len(sub) > 0:
            bottom = sub.idxmin()
            events.append(
                {
                    "start": dates.iloc[start_idx].strftime("%Y-%m-%d"),
                    "bottom": dates.iloc[bottom].strftime("%Y-%m-%d"),
                    "end": "in corso",
                    "depth_pct": float(sub.min() * 100.0),
                }
            )

    events.sort(key=lambda x: x["depth_pct"])  # più negativo prima
    return events[:n]


# ----------------------------------------------------------
# Portafoglio
# ----------------------------------------------------------

def compute_portfolio(ls80: pd.Series, gold: pd.Series, w_gold: float, capital: float) -> pd.Series:
    w_gold = max(0.0, min(0.5, float(w_gold)))
    w_ls80 = 1.0 - w_gold

    shares_ls80 = (capital * w_ls80) / float(ls80.iloc[0])
    shares_gold = (capital * w_gold) / float(gold.iloc[0])

    return shares_ls80 * ls80 + shares_gold * gold


# ----------------------------------------------------------
# Routes
# ----------------------------------------------------------

@app.get("/")
def home():
    return render_template("index.html")


@app.get("/api/diag")
def api_diag():
    def info(path: Path) -> Dict[str, Any]:
        if not path.exists():
            return {"exists": False, "file": str(path)}
        try:
            df = read_price_csv(path)
            return {
                "exists": True,
                "file": str(path),
                "rows": int(len(df)),
                "first_date": str(df["date"].iloc[0].date()),
                "last_date": str(df["date"].iloc[-1].date()),
                "first_value": float(df["close"].iloc[0]),
                "last_value": float(df["close"].iloc[-1]),
            }
        except Exception as e:
            return {
                "exists": True,
                "file": str(path),
                "error": f"{type(e).__name__}: {e}",
            }

    payload: Dict[str, Any] = {
        "ok": True,
        "time_utc": _now_iso(),
        "files": {
            "ls80": info(LS80_FILE),
            "gold": info(GOLD_FILE),
            "world": info(WORLD_FILE),
        },
        "env": {
            "LS80_TICKER": LS80_TICKER,
            "GOLD_TICKER": GOLD_TICKER,
            "WORLD_TICKER": WORLD_TICKER,
        },
    }

    try:
        ls80 = read_price_csv(LS80_FILE)
        gold = read_price_csv(GOLD_FILE)
        world = read_price_csv(WORLD_FILE)

        merged = ls80.merge(gold, on="date", how="inner", suffixes=("_ls80", "_gold"))
        merged = merged.merge(world, on="date", how="inner")

        payload["merge"] = {
            "rows_inner": int(len(merged)),
            "first_date": str(merged["date"].iloc[0].date()) if len(merged) else None,
            "last_date": str(merged["date"].iloc[-1].date()) if len(merged) else None,
        }
    except Exception as e:
        payload["merge_error"] = f"{type(e).__name__}: {e}"

    return jsonify(payload)


@app.get("/api/compute")
def api_compute():
    try:
        w_gold = float(request.args.get("w_gold", 0.2))
        capital = float(request.args.get("capital", 10000))

        if capital <= 0:
            return _json_error("Capitale non valido.", 400)

        ls80 = read_price_csv(LS80_FILE)
        gold = read_price_csv(GOLD_FILE)
        world = read_price_csv(WORLD_FILE)

        # Date comuni a tutti e tre
        df = ls80.merge(gold, on="date", how="inner", suffixes=("_ls80", "_gold"))
        df = df.merge(world, on="date", how="inner")
        df.columns = ["date", "ls80", "gold", "world"]

        if len(df) < 20:
            return _json_error(
                "Serie troppo corta dopo il merge tra LS80, Oro e MSCI World.",
                400,
                rows_inner=int(len(df)),
            )

        dates = df["date"].reset_index(drop=True)

        portfolio = compute_portfolio(df["ls80"], df["gold"], w_gold, capital).reset_index(drop=True)

        # World normalizzato sullo stesso capitale iniziale
        world_scaled = (capital * (df["world"] / df["world"].iloc[0])).reset_index(drop=True)

        cagr_port = compute_cagr(portfolio, dates)
        maxdd_port = compute_max_dd(portfolio)
        dbl_years = doubling_years(cagr_port)

        cagr_world = compute_cagr(world_scaled, dates)
        maxdd_world = compute_max_dd(world_scaled)

        dd_port = compute_drawdown_series_pct(portfolio)
        dd_world = compute_drawdown_series_pct(world_scaled)

        worst_port = worst_drawdowns(portfolio, dates, n=3)
        worst_world = worst_drawdowns(world_scaled, dates, n=3)

        # Dazi Trump 2025 = peggior drawdown nel 2025
        mask_2025 = dates.dt.year == 2025
        dd_2025_port = float((dd_port[mask_2025] / 100.0).min()) if mask_2025.any() else None
        dd_2025_world = float((dd_world[mask_2025] / 100.0).min()) if mask_2025.any() else None

        w_ls80 = 1.0 - w_gold
        az = 0.80 * w_ls80
        ob = 0.20 * w_ls80

        years_period = (dates.iloc[-1] - dates.iloc[0]).days / 365.25

        return jsonify(
            {
                "ok": True,
                "dates": dates.dt.strftime("%Y-%m-%d").tolist(),
                "portfolio": portfolio.tolist(),
                "world": world_scaled.tolist(),
                "drawdown_portfolio_pct": dd_port.tolist(),
                "drawdown_world_pct": dd_world.tolist(),
                "composition": {
                    "azionario": az,
                    "obbligazionario": ob,
                    "oro": w_gold,
                },
                "metrics": {
                    "final_portfolio": float(portfolio.iloc[-1]),
                    "final_years": float(years_period),
                    "cagr_portfolio": float(cagr_port),
                    "max_dd_portfolio": float(maxdd_port),
                    "doubling_years_portfolio": float(dbl_years) if dbl_years is not None else None,
                    "cagr_world": float(cagr_world),
                    "max_dd_world": float(maxdd_world),
                    "dd_2025_portfolio": dd_2025_port,
                    "dd_2025_world": dd_2025_world,
                    "worst_episodes_portfolio": worst_port,
                    "worst_episodes_world": worst_world,
                },
            }
        )

    except Exception as e:
        return jsonify(
            {
                "ok": False,
                "error": f"Errore compute: {str(e)}",
            }
        )


# ----------------------------------------------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=True)
