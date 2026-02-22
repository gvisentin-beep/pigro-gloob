from __future__ import annotations

import os
import json
import math
import time
from datetime import datetime, date
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, render_template

# OpenAI (nuovo SDK)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # gestiamo sotto


app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))

LS80_FILE = DATA_DIR / "ls80.csv"
GOLD_FILE = DATA_DIR / "gold.csv"

ASK_DAILY_LIMIT = int(os.getenv("ASK_DAILY_LIMIT", "10"))
ASK_STORE_FILE = DATA_DIR / "ask_limits.json"

UPDATE_TOKEN = os.getenv("UPDATE_TOKEN", "")
LS80_TICKER = os.getenv("LS80_TICKER", "VNGA80.MI")
GOLD_TICKER = os.getenv("GOLD_TICKER", "SGLD")


# ----------------------------
# Helpers
# ----------------------------
def _now_iso() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")


def _safe_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _detect_sep(file_path: Path) -> str:
    """
    I tuoi CSV su GitHub sono del tipo: Date;Close (quindi ';').
    Se un domani li cambi, questo si adatta.
    """
    try:
        with file_path.open("r", encoding="utf-8", errors="ignore") as f:
            head = f.readline()
        if ";" in head and "," not in head:
            return ";"
        if "," in head and ";" not in head:
            return ","
        # se ci sono entrambi, scegli quello più frequente nella prima riga
        return ";" if head.count(";") >= head.count(",") else ","
    except Exception:
        return ";"


def _read_price_csv(file_path: Path) -> pd.DataFrame:
    """
    Ritorna DataFrame con colonne: date (datetime), close (float)
    Gestisce:
      - separatore ; o ,
      - date in formato gg/mm/aaaa o yyyy-mm-dd
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File non trovato: {file_path}")

    sep = _detect_sep(file_path)
    df = pd.read_csv(file_path, sep=sep)

    # normalizza nomi
    cols = {c.strip().lower(): c for c in df.columns}
    if "date" not in cols or "close" not in cols:
        # prova a ripulire eventuali spazi
        df.columns = [c.strip() for c in df.columns]
        cols = {c.strip().lower(): c for c in df.columns}
        if "date" not in cols or "close" not in cols:
            raise ValueError(f"CSV {file_path.name}: colonne attese 'Date' e 'Close'. Trovate: {list(df.columns)}")

    dcol = cols["date"]
    ccol = cols["close"]

    # parse date (dayfirst=True fondamentale per 21/01/2026)
    dates = pd.to_datetime(df[dcol], errors="coerce", dayfirst=True, utc=False)

    # close a float
    close = pd.to_numeric(df[ccol], errors="coerce")

    out = pd.DataFrame({"date": dates, "close": close}).dropna()
    out = out.sort_values("date").drop_duplicates(subset=["date"], keep="last")

    if len(out) < 10:
        raise ValueError(f"CSV {file_path.name}: troppo poche righe valide dopo parsing ({len(out)}).")

    return out


def _compute_drawdown(series: pd.Series) -> float:
    peak = series.cummax()
    dd = (series / peak) - 1.0
    return float(dd.min())


def _compute_cagr(values: pd.Series, dates: pd.Series) -> Optional[float]:
    if len(values) < 2:
        return None
    t_years = (dates.iloc[-1] - dates.iloc[0]).days / 365.25
    if t_years <= 0:
        return None
    v0 = float(values.iloc[0])
    v1 = float(values.iloc[-1])
    if v0 <= 0 or v1 <= 0:
        return None
    return (v1 / v0) ** (1.0 / t_years) - 1.0


def _doubling_years(cagr: Optional[float]) -> Optional[float]:
    if cagr is None:
        return None
    if cagr <= 0:
        return None
    return math.log(2.0) / math.log(1.0 + cagr)


def _first_trading_day_each_year(dates: pd.Series) -> pd.Series:
    """
    Prende la prima data disponibile per ciascun anno (prima riga dell’anno nel dataset).
    """
    years = dates.dt.year
    first_idx = dates.groupby(years).head(1).index
    return pd.Series(True, index=dates.index).where(dates.index.isin(first_idx), False).fillna(False)


def _annual_rebalance_portfolio(
    ls80_close: pd.Series,
    gold_close: pd.Series,
    dates: pd.Series,
    w_gold: float,
    capital: float,
) -> pd.Series:
    """
    Portafoglio con ribilanciamento annuale (prima data disponibile di ogni anno).
    w_gold: peso oro (0..1)
    w_ls80: 1 - w_gold
    """
    w_gold = float(np.clip(w_gold, 0.0, 1.0))
    w_ls80 = 1.0 - w_gold

    # holdings iniziali
    v0 = float(capital)
    ls80_shares = (v0 * w_ls80) / float(ls80_close.iloc[0])
    gold_shares = (v0 * w_gold) / float(gold_close.iloc[0])

    rebalance_flags = _first_trading_day_each_year(dates)

    values = []
    for i in range(len(dates)):
        v = ls80_shares * float(ls80_close.iloc[i]) + gold_shares * float(gold_close.iloc[i])
        values.append(v)

        # ribilancia sul primo giorno dell'anno (escluso i=0 perché già allocato)
        if i != 0 and bool(rebalance_flags.iloc[i]):
            v_now = v
            ls80_shares = (v_now * w_ls80) / float(ls80_close.iloc[i])
            gold_shares = (v_now * w_gold) / float(gold_close.iloc[i])

    return pd.Series(values)


def _client_ip() -> str:
    # Render spesso mette X-Forwarded-For
    xff = request.headers.get("X-Forwarded-For", "")
    if xff:
        return xff.split(",")[0].strip()
    return request.remote_addr or "unknown"


def _load_ask_store() -> Dict[str, Any]:
    try:
        if ASK_STORE_FILE.exists():
            return json.loads(ASK_STORE_FILE.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def _save_ask_store(store: Dict[str, Any]) -> None:
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        ASK_STORE_FILE.write_text(json.dumps(store, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        # se non riesce a scrivere (filesystem read-only), pazienza
        pass


def _check_and_consume_quota(ip: str) -> Tuple[int, int]:
    """
    Ritorna (remaining, limit) dopo aver consumato 1.
    Se quota esaurita, remaining=0.
    """
    limit = ASK_DAILY_LIMIT
    today = date.today().isoformat()
    store = _load_ask_store()

    key = f"{today}:{ip}"
    used = int(store.get(key, 0))

    if used >= limit:
        return 0, limit

    used += 1
    store[key] = used
    _save_ask_store(store)

    remaining = max(0, limit - used)
    return remaining, limit


def _openai_client() -> Optional[Any]:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key or OpenAI is None:
        return None
    try:
        return OpenAI(api_key=api_key)
    except Exception:
        return None


# ----------------------------
# Pages
# ----------------------------
@app.get("/")
def home():
    return render_template("index.html")


@app.get("/health")
def health():
    return jsonify({"ok": True})


# ----------------------------
# API: diagnostics
# ----------------------------
@app.get("/api/diag")
def api_diag():
    diag: Dict[str, Any] = {
        "time_utc": _now_iso(),
        "data_dir": str(DATA_DIR),
        "env": {
            "LS80_TICKER": LS80_TICKER,
            "GOLD_TICKER": GOLD_TICKER,
            "OPENAI_API_KEY_present": bool(os.getenv("OPENAI_API_KEY", "").strip()),
            "UPDATE_TOKEN_present": bool(UPDATE_TOKEN.strip()),
            "ASK_DAILY_LIMIT": ASK_DAILY_LIMIT,
        },
        "files": {},
    }

    def file_info(fp: Path) -> Dict[str, Any]:
        info: Dict[str, Any] = {"exists": fp.exists(), "file": str(fp)}
        if not fp.exists():
            return info
        try:
            df = _read_price_csv(fp)
            info.update(
                {
                    "rows": int(len(df)),
                    "first_date": str(df["date"].iloc[0].date()),
                    "last_date": str(df["date"].iloc[-1].date()),
                    "first_value": float(df["close"].iloc[0]),
                    "last_value": float(df["close"].iloc[-1]),
                    "sep_detected": _detect_sep(fp),
                }
            )
        except Exception as e:
            info["error"] = f"{type(e).__name__}: {e}"
        return info

    diag["files"]["ls80"] = file_info(LS80_FILE)
    diag["files"]["gold"] = file_info(GOLD_FILE)

    # prova merge (per capire se combaciano le date)
    try:
        ls = _read_price_csv(LS80_FILE)
        gd = _read_price_csv(GOLD_FILE)
        merged = ls.merge(gd, on="date", how="inner", suffixes=("_ls80", "_gold"))
        diag["merge"] = {
            "rows_inner": int(len(merged)),
            "first_date": str(merged["date"].iloc[0].date()) if len(merged) else None,
            "last_date": str(merged["date"].iloc[-1].date()) if len(merged) else None,
        }
    except Exception as e:
        diag["merge"] = {"error": f"{type(e).__name__}: {e}"}

    return jsonify(diag)


# ----------------------------
# API: compute (grafico + metriche)
# ----------------------------
@app.get("/api/compute")
def api_compute():
    try:
        w_gold = _safe_float(request.args.get("w_gold"))
        if w_gold is None:
            # fallback: alcuni vecchi frontend mandavano w_ls80
            w_ls80 = _safe_float(request.args.get("w_ls80"))
            if w_ls80 is None:
                w_gold = 0.20
            else:
                w_gold = 1.0 - float(w_ls80)

        # accetta sia "20" che "0.20"
        if w_gold > 1.0:
            w_gold = w_gold / 100.0

        # clamp 0..0.50 (come slider attuale)
        w_gold = float(np.clip(w_gold, 0.0, 0.50))

        capital = _safe_float(request.args.get("capital")) or _safe_float(request.args.get("initial")) or 10000.0
        if capital <= 0:
            capital = 10000.0

        ls = _read_price_csv(LS80_FILE)
        gd = _read_price_csv(GOLD_FILE)

        df = ls.merge(gd, on="date", how="inner", suffixes=("_ls80", "_gold"))
        df = df.rename(columns={"close_ls80": "ls80", "close_gold": "gold"})

        if len(df) < 20:
            return jsonify({"error": "Poche date in comune tra LS80 e Oro (merge troppo corto)."}), 400

        dates = df["date"]
        port = _annual_rebalance_portfolio(df["ls80"], df["gold"], dates, w_gold=w_gold, capital=float(capital))

        # metriche
        cagr = _compute_cagr(port, dates)
        max_dd = _compute_drawdown(port)
        dbl = _doubling_years(cagr)

        # composizione (solo per testo)
        w_ls80 = 1.0 - w_gold
        az = 0.80 * w_ls80
        ob = 0.20 * w_ls80

        # anni del periodo (reali)
        years_period = (dates.iloc[-1] - dates.iloc[0]).days / 365.25

        payload: Dict[str, Any] = {
            "dates": [d.strftime("%Y-%m-%d") for d in dates],
            "portfolio": [float(x) for x in port],
            "metrics": {
                "annualized_return": cagr,
                "max_drawdown": max_dd,
                "years_period": years_period,
                "final_value": float(port.iloc[-1]),
                "doubling_years": dbl,
                "weights": {
                    "gold": w_gold,
                    "ls80": w_ls80,
                    "equity": az,
                    "bond": ob,
                },
            },
        }

        # compatibilità con vecchi frontend (chiavi alternative)
        payload["cagr_portfolio"] = cagr
        payload["max_dd_portfolio"] = max_dd
        payload["final_value"] = float(port.iloc[-1])
        payload["final_years"] = years_period
        payload["composition"] = {"azionario": az, "obbligazionario": ob, "oro": w_gold}

        return jsonify(payload)

    except Exception as e:
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500


# ----------------------------
# API: ask (assistente)
# ----------------------------
@app.post("/api/ask")
def api_ask():
    try:
        data = request.get_json(silent=True) or {}
        question = (data.get("question") or data.get("q") or "").strip()

        if not question:
            return jsonify({"error": "Scrivi una domanda."}), 400

        # quota per IP
        ip = _client_ip()
        remaining, limit = _check_and_consume_quota(ip)
        if remaining == 0 and limit > 0:
            return jsonify(
                {
                    "error": "Limite giornaliero raggiunto.",
                    "remaining": 0,
                    "limit": limit,
                }
            ), 429

        client = _openai_client()
        if client is None:
            # Risposta “di fallback” (non crasha il frontend)
            return jsonify(
                {
                    "answer": "Assistente non configurato: manca OPENAI_API_KEY su Render (Environment).",
                    "remaining": remaining,
                    "limit": limit,
                }
            )

        # Risposta breve e chiara, stile “educativo”
        resp = client.responses.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
            input=(
                "Rispondi in italiano, in modo semplice e pratico. "
                "Contesto: sito 'Metodo Pigro' (ETF azion-obblig + oro), nessuna consulenza personalizzata. "
                "Domanda utente:\n"
                f"{question}"
            ),
        )

        answer = ""
        try:
            answer = resp.output_text  # SDK recente
        except Exception:
            answer = str(resp)

        return jsonify(
            {
                "answer": answer.strip(),
                "remaining": remaining,
                "limit": limit,
            }
        )

    except Exception as e:
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500


# ----------------------------
# API: update data (opzionale, richiede yfinance)
# ----------------------------
@app.get("/api/update_data")
def api_update_data():
    """
    Aggiorna i CSV con l'ultima chiusura (Yahoo Finance) - opzionale.
    Protezione con token: /api/update_data?token=...
    """
    token = (request.args.get("token") or "").strip()
    if not UPDATE_TOKEN or token != UPDATE_TOKEN:
        return jsonify({"error": "Token non valido."}), 401

    try:
        import yfinance as yf  # richiede requirements: yfinance
    except Exception as e:
        return jsonify({"error": f"yfinance non disponibile (requirements). {type(e).__name__}: {e}"}), 500

    def update_one(ticker: str, file_path: Path) -> Dict[str, Any]:
        info: Dict[str, Any] = {"ticker": ticker, "file": str(file_path), "updated": False}
        df = _read_price_csv(file_path)

        last_date = df["date"].iloc[-1].date()

        # scarica ultimi ~10 giorni per sicurezza
        hist = yf.download(ticker, period="10d", interval="1d", auto_adjust=False, progress=False)
        if hist is None or hist.empty:
            info["reason"] = "no_data_from_yahoo"
            return info

        hist = hist.reset_index()
        # colonna data può essere 'Date' o 'Datetime'
        dcol = "Date" if "Date" in hist.columns else ("Datetime" if "Datetime" in hist.columns else None)
        if dcol is None or "Close" not in hist.columns:
            info["reason"] = "unexpected_columns"
            info["columns"] = list(hist.columns)
            return info

        hist["date"] = pd.to_datetime(hist[dcol], errors="coerce").dt.tz_localize(None)
        hist["close"] = pd.to_numeric(hist["Close"], errors="coerce")
        hist = hist[["date", "close"]].dropna().sort_values("date")

        new_rows = hist[hist["date"].dt.date > last_date]
        if new_rows.empty:
            info["reason"] = "already_up_to_date"
            return info

        # append
        out = pd.concat([df, new_rows], ignore_index=True)
        out = out.sort_values("date").drop_duplicates(subset=["date"], keep="last")

        # salva nel formato attuale: Date;Close e date gg/mm/aaaa
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        save = pd.DataFrame(
            {
                "Date": out["date"].dt.strftime("%d/%m/%Y"),
                "Close": out["close"].map(lambda x: f"{float(x):.2f}"),
            }
        )
        save.to_csv(file_path, sep=";", index=False)
        info["updated"] = True
        info["added_rows"] = int(len(new_rows))
        info["last_date"] = str(out["date"].iloc[-1].date())
        info["last_value"] = float(out["close"].iloc[-1])
        return info

    try:
        res_ls = update_one(LS80_TICKER, LS80_FILE)
        res_gd = update_one(GOLD_TICKER, GOLD_FILE)
        return jsonify({"ok": True, "ls80": res_ls, "gold": res_gd, "time_utc": _now_iso()})
    except Exception as e:
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500


if __name__ == "__main__":
    # locale
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "10000")), debug=True)
