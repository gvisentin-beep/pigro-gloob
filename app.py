from __future__ import annotations
import os
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

LS80_FILE = DATA_DIR / "ls80.csv"
GOLD_FILE = DATA_DIR / "gold.csv"
WORLD_FILE = DATA_DIR / "world.csv"


# ----------------------------------------------------------
# LETTURA CSV (compatibile con tutti i tuoi formati)
# ----------------------------------------------------------

def read_price_csv(path: Path) -> pd.DataFrame:

    if not path.exists():
        raise FileNotFoundError(str(path))

    # prova formato con intestazione
    try:
        df = pd.read_csv(path, sep=";", dtype=str)
        cols = [c.lower() for c in df.columns]

        if "date" in cols and "close" in cols:
            df.columns = cols

            df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")

            df["close"] = pd.to_numeric(
                df["close"].astype(str).str.replace(",", "."),
                errors="coerce"
            )

            df = df.dropna(subset=["date", "close"]).sort_values("date")

            return df[["date", "close"]]

    except:
        pass

    # prova formato senza intestazione
    df = pd.read_csv(
        path,
        sep=";",
        header=None,
        names=["date", "close"],
        dtype=str
    )

    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")

    df["close"] = pd.to_numeric(
        df["close"].astype(str).str.replace(",", "."),
        errors="coerce"
    )

    df = df.dropna(subset=["date", "close"]).sort_values("date")

    if df.empty:
        raise ValueError(f"CSV non valido: {path}")

    return df


# ----------------------------------------------------------
# CAGR
# ----------------------------------------------------------

def compute_cagr(series, dates):

    if len(series) < 2:
        return 0

    years = (dates.iloc[-1] - dates.iloc[0]).days / 365.25

    if years <= 0:
        return 0

    return (series.iloc[-1] / series.iloc[0]) ** (1 / years) - 1


# ----------------------------------------------------------
# MAX DRAWDOWN
# ----------------------------------------------------------

def compute_max_dd(series):

    peak = series.cummax()

    dd = series / peak - 1

    return dd.min()


# ----------------------------------------------------------
# DRAW DOWN SERIES
# ----------------------------------------------------------

def compute_drawdown_series(series):

    peak = series.cummax()

    dd = series / peak - 1

    return dd * 100


# ----------------------------------------------------------
# PEGGIORI DISCese
# ----------------------------------------------------------

def worst_drawdowns(series, dates, n=3):

    peak = series.cummax()

    dd = series / peak - 1

    events = []

    in_dd = False
    start = None

    for i in range(1, len(series)):

        if not in_dd and dd.iloc[i] < 0:
            start = i - 1
            in_dd = True

        if in_dd and dd.iloc[i] == 0:

            sub = dd.iloc[start:i]

            bottom = sub.idxmin()

            events.append({

                "start": dates.iloc[start].strftime("%Y-%m-%d"),
                "bottom": dates.iloc[bottom].strftime("%Y-%m-%d"),
                "end": dates.iloc[i].strftime("%Y-%m-%d"),
                "depth_pct": float(sub.min() * 100)

            })

            in_dd = False

    events.sort(key=lambda x: x["depth_pct"])

    return events[:n]


# ----------------------------------------------------------
# PORTAFOGLIO
# ----------------------------------------------------------

def compute_portfolio(ls80, gold, w_gold, capital):

    w_ls80 = 1 - w_gold

    shares_ls = (capital * w_ls80) / ls80.iloc[0]

    shares_gold = (capital * w_gold) / gold.iloc[0]

    values = shares_ls * ls80 + shares_gold * gold

    return values


# ----------------------------------------------------------
# API COMPUTE
# ----------------------------------------------------------

@app.get("/api/compute")
def api_compute():

    try:

        w_gold = float(request.args.get("w_gold", 0.2))

        capital = float(request.args.get("capital", 10000))

        ls80 = read_price_csv(LS80_FILE)

        gold = read_price_csv(GOLD_FILE)

        world = read_price_csv(WORLD_FILE)

        df = ls80.merge(gold, on="date", how="inner", suffixes=("_ls80", "_gold"))

        df = df.merge(world, on="date", how="inner")

        df.columns = ["date", "ls80", "gold", "world"]

        dates = df["date"]

        portfolio = compute_portfolio(df["ls80"], df["gold"], w_gold, capital)

        world_scaled = capital * (df["world"] / df["world"].iloc[0])

        cagr = compute_cagr(portfolio, dates)

        max_dd = compute_max_dd(portfolio)

        cagr_world = compute_cagr(world_scaled, dates)

        max_dd_world = compute_max_dd(world_scaled)

        dd_port = compute_drawdown_series(portfolio)

        dd_world = compute_drawdown_series(world_scaled)

        worst_port = worst_drawdowns(portfolio, dates)

        worst_world = worst_drawdowns(world_scaled, dates)

        mask_2025 = dates.dt.year == 2025

        dd_2025_port = float(dd_port[mask_2025].min()/100) if mask_2025.any() else None

        dd_2025_world = float(dd_world[mask_2025].min()/100) if mask_2025.any() else None

        return jsonify({

            "ok": True,

            "dates": dates.dt.strftime("%Y-%m-%d").tolist(),

            "portfolio": portfolio.tolist(),

            "world": world_scaled.tolist(),

            "drawdown_portfolio_pct": dd_port.tolist(),

            "drawdown_world_pct": dd_world.tolist(),

            "metrics": {

                "cagr_portfolio": cagr,
                "max_dd_portfolio": max_dd,

                "cagr_world": cagr_world,
                "max_dd_world": max_dd_world,

                "dd_2025_portfolio": dd_2025_port,
                "dd_2025_world": dd_2025_world,

                "worst_episodes_portfolio": worst_port,
                "worst_episodes_world": worst_world

            }

        })

    except Exception as e:

        return jsonify({

            "ok": False,
            "error": f"Errore compute: {str(e)}"

        })


# ----------------------------------------------------------
# DIAGNOSTICA
# ----------------------------------------------------------

@app.get("/api/diag")
def diag():

    def info(path):

        if not path.exists():
            return {"exists": False}

        df = read_price_csv(path)

        return {

            "exists": True,
            "rows": len(df),
            "first_date": str(df["date"].iloc[0].date()),
            "last_date": str(df["date"].iloc[-1].date())

        }

    return jsonify({

        "files": {

            "ls80": info(LS80_FILE),
            "gold": info(GOLD_FILE),
            "world": info(WORLD_FILE)

        },

        "time": datetime.now(timezone.utc).isoformat()

    })


# ----------------------------------------------------------
# HOME
# ----------------------------------------------------------

@app.route("/")
def home():

    return render_template("index.html")


# ----------------------------------------------------------

if __name__ == "__main__":

    app.run(host="0.0.0.0", port=5000)
