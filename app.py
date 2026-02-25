from __future__ import annotations

import os
import json
import math
import traceback
from datetime import datetime, date
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, render_template

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))

LS80_FILE = DATA_DIR / "ls80.csv"
GOLD_FILE = DATA_DIR / "gold.csv"

# ⚠️ FIX CRITICO: ticker corretti per Borsa Italiana
LS80_TICKER = os.getenv("LS80_TICKER", "VNGA80.MI")
GOLD_TICKER = os.getenv("GOLD_TICKER", "SGLD.MI")

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
UPDATE_TOKEN = os.getenv("UPDATE_TOKEN", "").strip()

BUILD_ID = "2026-02-26_GLOOB_STABLE_MI"

def _now_iso():
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")


def _read_price_csv(file_path: Path) -> pd.DataFrame:
    if not file_path.exists():
        raise FileNotFoundError(f"File non trovato: {file_path}")

    df = pd.read_csv(file_path, sep=";")
    df["date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df["close"] = pd.to_numeric(df["Close"], errors="coerce")

    df = df.dropna()
    df = df.sort_values("date")
    return df[["date", "close"]]


def _write_price_csv(file_path: Path, df: pd.DataFrame):
    save = pd.DataFrame({
        "Date": df["date"].dt.strftime("%d/%m/%Y"),
        "Close": df["close"].round(2)
    })
    save.to_csv(file_path, sep=";", index=False)


# ⭐ NUOVO updater robusto (risolve il tuo problema)
def _download_history(ticker: str) -> pd.DataFrame:
    import yfinance as yf

    t = yf.Ticker(ticker)
    hist = t.history(period="6mo", interval="1d")

    if hist.empty:
        return pd.DataFrame()

    df = hist[["Close"]].copy()
    df["date"] = pd.to_datetime(df.index)
    df["close"] = df["Close"]
    df = df[["date", "close"]]
    df = df.dropna()
    df["date"] = df["date"].dt.tz_localize(None)
    return df


def _update_one(ticker: str, file_path: Path):
    info = {"ticker": ticker, "updated": False}

    local = _read_price_csv(file_path)
    last_date = local["date"].iloc[-1].date()

    hist = _download_history(ticker)

    if hist.empty:
        info["reason"] = "no_data_from_yahoo"
        return info

    new_rows = hist[hist["date"].dt.date > last_date]

    if new_rows.empty:
        info["reason"] = "already_up_to_date"
        info["last_yahoo_date"] = str(hist["date"].iloc[-1].date())
        return info

    out = pd.concat([local, new_rows])
    out = out.drop_duplicates(subset=["date"]).sort_values("date")

    _write_price_csv(file_path, out)

    info["updated"] = True
    info["added_rows"] = len(new_rows)
    info["last_date"] = str(out["date"].iloc[-1].date())
    return info


@app.get("/")
def home():
    return render_template("index.html")


@app.get("/api/update_data")
def api_update_data():
    token = (request.args.get("token") or "").strip()
    if UPDATE_TOKEN and token != UPDATE_TOKEN:
        return jsonify({"ok": False, "error": "Token non valido"}), 401

    try:
        res_ls = _update_one(LS80_TICKER, LS80_FILE)
        res_gold = _update_one(GOLD_TICKER, GOLD_FILE)

        return jsonify({
            "ok": True,
            "ls80": res_ls,
            "gold": res_gold,
            "time": _now_iso()
        })

    except Exception as e:
        return jsonify({
            "ok": False,
            "error": str(e),
            "trace": traceback.format_exc()
        }), 500


@app.get("/api/diag")
def diag():
    ls = _read_price_csv(LS80_FILE)
    gold = _read_price_csv(GOLD_FILE)

    return jsonify({
        "ok": True,
        "build": BUILD_ID,
        "ls80_last_date": str(ls["date"].iloc[-1].date()),
        "gold_last_date": str(gold["date"].iloc[-1].date()),
        "LS80_TICKER": LS80_TICKER,
        "GOLD_TICKER": GOLD_TICKER,
        "time": _now_iso()
    })
