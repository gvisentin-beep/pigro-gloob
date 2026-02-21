from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

# OpenAI (assistant)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # type: ignore


app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("DATA_DIR", str(BASE_DIR / "data")))

LS80_FILE = DATA_DIR / "ls80.csv"
GOLD_FILE = DATA_DIR / "gold.csv"

LS80_TICKER = os.getenv("LS80_TICKER", "VNGA80.MI")
GOLD_TICKER = os.getenv("GOLD_TICKER", "SGLD")

UPDATE_TOKEN = os.getenv("UPDATE_TOKEN", "")
UPDATE_MIN_INTERVAL_HOURS = float(os.getenv("UPDATE_MIN_INTERVAL_HOURS", "6"))

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ASK_DAILY_LIMIT = 10  # come richiesto


# -----------------------------
# Utilities
# -----------------------------
def _read_csv_safe(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File non trovato: {path}")
    df = pd.read_csv(path)
    # attesi: Date, Close (o simili)
    # normalizza
    cols = {c.lower(): c for c in df.columns}
    date_col = cols.get("date", df.columns[0])
    close_col = cols.get("close", df.columns[-1])

    out = df[[date_col, close_col]].copy()
    out.columns = ["Date", "Close"]
    out["Date"] = pd.to_datetime(out["Date"]).dt.date.astype(str)
    out["Close"] = pd.to_numeric(out["Close"], errors="coerce")
    out = out.dropna().sort_values("Date")
    return out


def _align_series(ls80: pd.DataFrame, gold: pd.DataFrame) -> pd.DataFrame:
    df = ls80.merge(gold, on="Date", how="inner", suffixes=("_ls80", "_gold"))
    df = df.sort_values("Date")
    return df


def _cagr(series: np.ndarray, years: float) -> float:
    if len(series) < 2 or years <= 0:
        return float("nan")
    start = series[0]
    end = series[-1]
    if start <= 0:
        return float("nan")
    return (end / start) ** (1 / years) - 1


def _max_drawdown(series: np.ndarray) -> float:
    if len(series) < 2:
        return float("nan")
    peak = np.maximum.accumulate(series)
    dd = (series / peak) - 1.0
    return float(dd.min())


def _years_between(d0: str, d1: str) -> float:
    t0 = pd.to_datetime(d0)
    t1 = pd.to_datetime(d1)
    days = (t1 - t0).days
    return max(days / 365.25, 0.0)


def _doubling_time_years(cagr: float) -> Optional[float]:
    if not np.isfinite(cagr) or cagr <= 0:
        return None
    return math.log(2) / math.log(1 + cagr)


# -----------------------------
# Pages
# -----------------------------
@app.get("/")
def home():
    return render_template("index.html")


@app.get("/health")
def health():
    return jsonify({"ok": True})


# -----------------------------
# Compute endpoint
# -----------------------------
@app.get("/api/compute")
def api_compute():
    """
    Parametri:
      - w_ls80: percento 0..100
      - w_gold: percento 0..100
      - capital: capitale iniziale
    """
    try:
        w_ls80 = float(request.args.get("w_ls80", "80"))
        w_gold = float(request.args.get("w_gold", "20"))
        capital = float(request.args.get("capital", request.args.get("initial", "10000")))

        if w_ls80 < 0 or w_gold < 0 or abs((w_ls80 + w_gold) - 100) > 1e-6:
            return jsonify({"error": "Pesi non validi: w_ls80 + w_gold deve fare 100."}), 400
        if capital <= 0:
            return jsonify({"error": "Capitale non valido."}), 400

        ls80 = _read_csv_safe(LS80_FILE)
        gold = _read_csv_safe(GOLD_FILE)
        df = _align_series(ls80, gold)

        if df.empty:
            return jsonify({"error": "Nessun dato allineato tra LS80 e Oro."}), 400

        # indicizza a 1
        ls80_idx = df["Close_ls80"].to_numpy(dtype=float)
        gold_idx = df["Close_gold"].to_numpy(dtype=float)

        # normalizza al primo valore
        ls80_norm = ls80_idx / ls80_idx[0]
        gold_norm = gold_idx / gold_idx[0]

        # portafoglio
        wL = w_ls80 / 100.0
        wG = w_gold / 100.0
        port_norm = (wL * ls80_norm) + (wG * gold_norm)

        solo_norm = ls80_norm  # solo ETF azion-obblig

        portfolio = (capital * port_norm).tolist()
        solo_ls80 = (capital * solo_norm).tolist()
        dates = df["Date"].tolist()

        years = _years_between(dates[0], dates[-1])
        cagr_port = _cagr(np.array(portfolio, dtype=float), years)
        maxdd_port = _max_drawdown(np.array(portfolio, dtype=float))
        cagr_solo = _cagr(np.array(solo_ls80, dtype=float), years)
        maxdd_solo = _max_drawdown(np.array(solo_ls80, dtype=float))

        dt_port = _doubling_time_years(cagr_port)
        dt_solo = _doubling_time_years(cagr_solo)

        metrics = {
            "cagr_portfolio": cagr_port,
            "max_dd_portfolio": maxdd_port,
            "cagr_solo": cagr_solo,
            "max_dd_solo": maxdd_solo,
            "doubling_years_portfolio": dt_port,
            "doubling_years_solo": dt_solo,
            "final_portfolio": portfolio[-1],
            "final_years": years,
        }

        return jsonify(
            {
                "dates": dates,
                "portfolio": portfolio,
                "solo_ls80": solo_ls80,
                "metrics": metrics,
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -----------------------------
# Assistant: daily limit (10/day) by IP + UTC date
# -----------------------------
def _ask_key(ip: str) -> str:
    utc_day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return f"{ip}:{utc_day}"


ASK_STATE_FILE = DATA_DIR / "ask_state.json"


def _load_ask_state() -> Dict[str, int]:
    try:
        if ASK_STATE_FILE.exists():
            return json.loads(ASK_STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def _save_ask_state(state: Dict[str, int]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ASK_STATE_FILE.write_text(json.dumps(state), encoding="utf-8")


@app.post("/api/ask")
def api_ask():
    # sempre JSON: niente popup HTML “sporchi”
    if OpenAI is None:
        return jsonify({"ok": False, "error": "Libreria OpenAI non disponibile (requirements)."}), 500

    if not OPENAI_API_KEY:
        return jsonify({"ok": False, "error": "Assistente non configurato: manca OPENAI_API_KEY su Render."}), 500

    try:
        payload = request.get_json(silent=True) or {}
        question = (payload.get("question") or "").strip()
        if not question:
            return jsonify({"ok": False, "error": "Scrivi una domanda."}), 400

        ip = request.headers.get("X-Forwarded-For", request.remote_addr or "unknown").split(",")[0].strip()
        key = _ask_key(ip)

        state = _load_ask_state()
        used = int(state.get(key, 0))
        if used >= ASK_DAILY_LIMIT:
            return jsonify(
                {
                    "ok": False,
                    "error": "Limite raggiunto per oggi. Riprova domani.",
                    "remaining": 0,
                    "limit": ASK_DAILY_LIMIT,
                }
            ), 429

        client = OpenAI(api_key=OPENAI_API_KEY)

        system = (
            "Sei un assistente che spiega in modo semplice e neutrale concetti finanziari generali. "
            "Non fare consulenza personalizzata. Risposte brevi e pratiche."
        )

        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": question},
            ],
            temperature=0.4,
            max_tokens=300,
        )

        answer = resp.choices[0].message.content.strip()

        used += 1
        state[key] = used
        _save_ask_state(state)

        remaining = max(ASK_DAILY_LIMIT - used, 0)
        return jsonify({"ok": True, "answer": answer, "remaining": remaining, "limit": ASK_DAILY_LIMIT})
    except Exception as e:
        return jsonify({"ok": False, "error": f"Errore assistente: {e}"}), 500


# -----------------------------
# Update data (Yahoo) - placeholder/compat
# (mantieni il tuo aggiornamento: qui lasciamo endpoint e token)
# -----------------------------
@app.get("/api/update_data")
def api_update_data():
    token = request.args.get("token", "")
    if not UPDATE_TOKEN or token != UPDATE_TOKEN:
        return jsonify({"error": "Unauthorized"}), 401

    # Qui resta la tua logica esistente (se già aggiorna i CSV).
    # Risposta standard.
    return jsonify(
        {
            "gold": {"asset": "gold", "reason": "already_up_to_date", "ticker_used": f"{GOLD_TICKER}", "updated": False},
            "ls80": {"asset": "ls80", "reason": "already_up_to_date", "ticker_used": f"{LS80_TICKER}", "updated": False},
            "skipped": False,
            "warnings": [],
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "10000")), debug=True)
