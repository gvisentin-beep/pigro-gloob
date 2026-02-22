from __future__ import annotations

import json
import os
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

# -----------------------------
# App
# -----------------------------
app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("DATA_DIR", str(BASE_DIR / "data")))

LS80_TICKER = os.getenv("LS80_TICKER", "VNGA80.MI")
GOLD_TICKER = os.getenv("GOLD_TICKER", "SGLD")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

UPDATE_TOKEN = os.getenv("UPDATE_TOKEN", "")
UPDATE_MIN_INTERVAL_HOURS = float(os.getenv("UPDATE_MIN_INTERVAL_HOURS", "6"))

# mapping file assets
ASSET_FILES: Dict[str, str] = {
    "ls80": "ls80.csv",
    "gold": "gold.csv",
}

# cache in-memory per non rileggere sempre
_CACHE: Dict[str, Tuple[float, pd.Series]] = {}  # asset -> (mtime, series)


# -----------------------------
# Utilities
# -----------------------------
def _now_ts() -> float:
    return time.time()


def _parse_date_any(s: str) -> datetime:
    """
    Supporta:
      - YYYY-MM-DD
      - dd/mm/YYYY
      - dd-mm-YYYY
      - YYYY/MM/DD
      - mm/dd/YYYY (fallback)
      - stringhe con time (prende la parte data)
    """
    s = (s or "").strip()
    if not s:
        raise ValueError("empty date")

    # se contiene orario, prova a troncare (es: 2026-01-21 00:00:00)
    s0 = s.split(" ")[0].strip()

    fmts = [
        "%Y-%m-%d",
        "%d/%m/%Y",
        "%d-%m-%Y",
        "%Y/%m/%d",
        "%m/%d/%Y",
    ]

    # 1) fromisoformat (copre YYYY-MM-DD e YYYY-MM-DDTHH:MM:SS)
    try:
        return datetime.fromisoformat(s.replace("Z", "").replace("T", " ").strip())
    except Exception:
        pass

    # 2) prova con formati noti (s0 se serve)
    for f in fmts:
        try:
            return datetime.strptime(s0, f)
        except Exception:
            continue

    # 3) ultima spiaggia: pandas
    try:
        dt = pd.to_datetime(s, dayfirst=True, errors="raise")
        if isinstance(dt, pd.Timestamp):
            return dt.to_pydatetime()
    except Exception:
        pass

    raise ValueError(f"Unparseable date: {s}")


def _read_csv_flexible(path: Path) -> pd.DataFrame:
    """
    Legge CSV sia con separatore ';' (Date;Close) sia con ',' (Date,Close).
    Gestisce anche casi in cui GitHub preview segnala "No commas found".
    """
    # Tentativo 1: inferenza automatica
    try:
        df = pd.read_csv(path, sep=None, engine="python")
        if df.shape[1] == 1:
            # probabile sep sbagliato (tutto in una colonna tipo 'Date;Close')
            raise ValueError("single-column read; retry with ';'")
        return df
    except Exception:
        pass

    # Tentativo 2: separatore ';'
    try:
        df = pd.read_csv(path, sep=";")
        if df.shape[1] >= 2:
            return df
    except Exception:
        pass

    # Tentativo 3: separatore ','
    df = pd.read_csv(path, sep=",")
    return df


def _detect_and_read_csv(path: Path) -> pd.Series:
    """
    Ritorna una serie con index datetime e valori float.
    """
    if not path.exists():
        raise FileNotFoundError(str(path))

    df = _read_csv_flexible(path)

    # Normalizza colonne
    cols = [c.strip() for c in df.columns]
    df.columns = cols

    # casi tipici: Date, Close  oppure Date;Close
    date_col = None
    close_col = None
    for c in df.columns:
        cl = c.lower()
        if cl in ("date", "data"):
            date_col = c
        if cl in ("close", "chiusura", "prezzo"):
            close_col = c

    if date_col is None or close_col is None:
        # prova fallback: prime due colonne
        if df.shape[1] >= 2:
            date_col = df.columns[0]
            close_col = df.columns[1]
        else:
            raise ValueError(f"CSV columns not recognized in {path.name}: {df.columns.tolist()}")

    # Pulizia / parse
    dates_raw = df[date_col].astype(str).tolist()
    vals_raw = df[close_col].tolist()

    dates: List[datetime] = []
    vals: List[float] = []

    for d, v in zip(dates_raw, vals_raw):
        d = str(d).strip()
        if d.lower() in ("nan", "none", ""):
            continue

        # valori: possono arrivare come stringhe con virgola decimale
        try:
            vv = float(str(v).replace(",", ".").strip())
        except Exception:
            continue

        try:
            dt = _parse_date_any(d)
        except Exception:
            # caso tipico del tuo errore: riga letta male tipo "21/01/2026;39.37"
            # se capita, proviamo a splittare noi
            if ";" in d:
                a, b = d.split(";", 1)
                dt = _parse_date_any(a.strip())
                vv = float(b.strip().replace(",", "."))
            else:
                raise

        dates.append(dt)
        vals.append(vv)

    if not dates:
        raise ValueError(f"No valid rows parsed from {path.name}")

    s = pd.Series(vals, index=pd.to_datetime(dates)).sort_index()
    s = s[~s.index.duplicated(keep="last")]
    return s


def _load_asset_series(asset: str) -> pd.Series:
    """
    Carica la serie (da data/*.csv). Usa cache su mtime.
    """
    if asset not in ASSET_FILES:
        raise KeyError(f"Unknown asset: {asset}")

    path = DATA_DIR / ASSET_FILES[asset]
    mtime = path.stat().st_mtime if path.exists() else -1

    cached = _CACHE.get(asset)
    if cached and cached[0] == mtime:
        return cached[1].copy()

    s = _detect_and_read_csv(path)
    _CACHE[asset] = (mtime, s.copy())
    return s


def _align_two(a: pd.Series, b: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """
    Allinea due serie su date comuni (inner join).
    """
    df = pd.DataFrame({"a": a, "b": b}).dropna()
    return df["a"], df["b"]


def _compute_portfolio(ls80: pd.Series, gold: pd.Series, w_gold: float, capital: float) -> Dict:
    """
    Portafoglio = (1-w_gold)*LS80 + w_gold*GOLD
    w_gold in [0,1]
    """
    ls80_a, gold_a = _align_two(ls80, gold)

    w_gold = float(np.clip(w_gold, 0.0, 1.0))
    w_ls80 = 1.0 - w_gold

    # normalizza a capitale iniziale
    base_ls = ls80_a.iloc[0]
    base_g = gold_a.iloc[0]
    ls_norm = (ls80_a / base_ls) * capital
    g_norm = (gold_a / base_g) * capital

    port = (w_ls80 * ls_norm) + (w_gold * g_norm)

    # metriche base
    start = float(port.iloc[0])
    end = float(port.iloc[-1])
    dates = port.index.to_pydatetime().tolist()

    days = (dates[-1] - dates[0]).days
    years = days / 365.25 if days > 0 else 0.0

    cagr = (end / start) ** (1 / years) - 1 if years > 0 and start > 0 else np.nan

    running_max = port.cummax()
    dd = (port / running_max) - 1.0
    max_dd = float(dd.min()) if len(dd) else np.nan

    # raddoppio (anni) ~ ln(2)/ln(1+cagr)
    doubling_years = np.nan
    try:
        if np.isfinite(cagr) and cagr > -0.999:
            if cagr > 0:
                doubling_years = float(np.log(2) / np.log(1 + cagr))
    except Exception:
        pass

    # composizione “vera” (LS80 = 80% azioni, 20% obbligazioni)
    w_equity = w_ls80 * 0.80
    w_bond = w_ls80 * 0.20

    out = {
        "dates": [d.strftime("%Y-%m-%d") for d in port.index],
        "portfolio": [float(x) for x in port.values],
        "metrics": {
            "annualized_return": float(cagr) if np.isfinite(cagr) else None,
            "max_drawdown": float(max_dd) if np.isfinite(max_dd) else None,
            "doubling_years": float(doubling_years) if np.isfinite(doubling_years) else None,
            "final_value": float(end),
            "years_realized": float(years) if np.isfinite(years) else None,
            "weights": {
                "gold": float(w_gold),
                "equity": float(w_equity),
                "bond": float(w_bond),
            },
        },
    }
    return out


# -----------------------------
# Routes
# -----------------------------
@app.get("/health")
def health():
    return jsonify({"ok": True})


@app.get("/")
def home():
    # templates/index.html
    return render_template("index.html")


@app.get("/api/compute")
def api_compute():
    """
    Params:
      - w_gold (0..50 in percent, step 5) OR 0..1 (tolleriamo entrambi)
      - capital
    """
    try:
        w_gold = request.args.get("w_gold", default="20")
        capital = request.args.get("capital", default="10000")

        w_gold = float(w_gold)
        # se arriva in percento (0..100), converti
        if w_gold > 1.0:
            w_gold = w_gold / 100.0

        capital = float(str(capital).replace(".", "").replace(",", "."))
        if capital <= 0:
            capital = 10000.0

        ls80 = _load_asset_series("ls80")
        gold = _load_asset_series("gold")

        out = _compute_portfolio(ls80, gold, w_gold=w_gold, capital=capital)
        return jsonify(out)

    except Exception as e:
        return jsonify({"error": f"Compute error: {e}"}), 500


@app.get("/api/diag")
def api_diag():
    """
    Diagnostica generale: file presenti, parsing, prime/ultime date, righe.
    """
    def pack_series(name: str) -> Dict:
        path = DATA_DIR / ASSET_FILES[name]
        info = {
            "file": str(path),
            "exists": path.exists(),
        }
        if not path.exists():
            return info

        try:
            s = _detect_and_read_csv(path)
            info.update(
                {
                    "rows": int(len(s)),
                    "first_date": s.index.min().strftime("%Y-%m-%d"),
                    "last_date": s.index.max().strftime("%Y-%m-%d"),
                    "first_value": float(s.iloc[0]),
                    "last_value": float(s.iloc[-1]),
                }
            )
        except Exception as ex:
            info["parse_error"] = str(ex)
            info["trace"] = traceback.format_exc(limit=2)
        return info

    diag = {
        "ok": True,
        "time_utc": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "base_dir": str(BASE_DIR),
        "data_dir": str(DATA_DIR),
        "env": {
            "LS80_TICKER": LS80_TICKER,
            "GOLD_TICKER": GOLD_TICKER,
            "UPDATE_MIN_INTERVAL_HOURS": UPDATE_MIN_INTERVAL_HOURS,
            "OPENAI_API_KEY_present": bool(OPENAI_API_KEY),
        },
        "python": {
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "assets": {
            "ls80": pack_series("ls80"),
            "gold": pack_series("gold"),
        },
    }
    return jsonify(diag)


@app.get("/api/diag_compute")
def api_diag_compute():
    """
    Diagnostica sul compute con parametri correnti.
    """
    try:
        w_gold = request.args.get("w_gold", default="20")
        capital = request.args.get("capital", default="10000")

        w_gold_f = float(w_gold)
        if w_gold_f > 1.0:
            w_gold_f = w_gold_f / 100.0

        capital_f = float(str(capital).replace(".", "").replace(",", "."))
        ls80 = _load_asset_series("ls80")
        gold = _load_asset_series("gold")
        out = _compute_portfolio(ls80, gold, w_gold=w_gold_f, capital=capital_f)

        return jsonify(
            {
                "ok": True,
                "request": {"w_gold": w_gold, "capital": capital},
                "aligned_points": len(out["dates"]),
                "first_date": out["dates"][0],
                "last_date": out["dates"][-1],
                "final_value": out["metrics"]["final_value"],
                "annualized_return": out["metrics"]["annualized_return"],
                "max_drawdown": out["metrics"]["max_drawdown"],
            }
        )
    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "trace": traceback.format_exc(limit=3)}), 500


# -----------------------------
# (Se nel tuo progetto hai già /api/update_data e /api/ask, qui sotto puoi
#  incollare le tue versioni attuali. Io NON le tocco, così non rompi nulla.)
#  Se vuoi, nel prossimo step te le reintegro io “sane” dentro questo file.
# -----------------------------

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
