from __future__ import annotations

import io
import json
import os
import time
import math
import traceback
from dataclasses import dataclass
from datetime import datetime, date
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import requests
from flask import Flask, jsonify, render_template, request, send_from_directory

# =========================
# Config / Paths
# =========================
app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

LS80_CSV = DATA_DIR / "ls80.csv"
GOLD_CSV = DATA_DIR / "gold.csv"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()

LS80_TICKER = os.getenv("LS80_TICKER", "VNGA80.MI").strip()
GOLD_TICKER = os.getenv("GOLD_TICKER", "SGLD").strip()

UPDATE_TOKEN = os.getenv("UPDATE_TOKEN", "").strip()
UPDATE_MIN_INTERVAL_HOURS = float(os.getenv("UPDATE_MIN_INTERVAL_HOURS", "6") or "6")

# rate limit assistente
ASK_DAILY_LIMIT = 10

# cache minima per limitare update_data
_LAST_UPDATE_TS_FILE = DATA_DIR / ".last_update_ts"

# =========================
# Helpers
# =========================
def _now_ts() -> float:
    return time.time()

def _safe_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default

def _parse_int(x, default=0):
    try:
        return int(float(x))
    except Exception:
        return default

def _is_truthy(x: str) -> bool:
    return str(x).strip().lower() in {"1", "true", "yes", "y", "ok"}

def _format_exc(e: Exception) -> str:
    return f"{type(e).__name__}: {e}"

def _read_last_update_ts() -> Optional[float]:
    try:
        if _LAST_UPDATE_TS_FILE.exists():
            return float(_LAST_UPDATE_TS_FILE.read_text(encoding="utf-8").strip())
    except Exception:
        return None
    return None

def _write_last_update_ts(ts: float) -> None:
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        _LAST_UPDATE_TS_FILE.write_text(str(ts), encoding="utf-8")
    except Exception:
        pass

def _detect_delimiter(sample: str) -> str:
    # i tuoi file sono con ;, ma facciamo robusto
    if sample.count(";") >= sample.count(","):
        return ";"
    return ","

def _load_price_csv(path: Path) -> pd.DataFrame:
    """
    Carica CSV con colonne Date;Close (oppure Date,Close).
    Supporta date tipo DD/MM/YYYY e YYYY-MM-DD.
    Ritorna df con index datetime (tz-naive) e colonna 'close' float.
    """
    if not path.exists():
        raise FileNotFoundError(f"File non trovato: {path}")

    raw = path.read_text(encoding="utf-8", errors="replace")
    # alcune volte GitHub aggiunge warning ma non modifica il contenuto; qui leggiamo normale.

    sample = raw[:2000]
    sep = _detect_delimiter(sample)

    df = pd.read_csv(io.StringIO(raw), sep=sep, engine="python")
    # normalizza nomi colonne
    cols = {c.strip().lower(): c for c in df.columns}
    if "date" not in cols or "close" not in cols:
        raise ValueError(f"CSV {path.name}: colonne attese Date/Close. Trovate: {list(df.columns)}")

    df = df.rename(columns={cols["date"]: "date", cols["close"]: "close"})
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")

    df = df.dropna(subset=["date", "close"]).copy()
    df = df.sort_values("date")
    df = df.set_index("date")
    df = df[~df.index.duplicated(keep="last")]
    return df

def _first_trading_day_each_year(idx: pd.DatetimeIndex) -> List[pd.Timestamp]:
    # prima data presente per ogni anno (nel tuo dataset è “primo giorno di mercato” disponibile)
    years = pd.Series(idx.year, index=idx)
    firsts = years.groupby(years).apply(lambda s: s.index[0])
    return list(firsts.values)

def _compute_portfolio_series(
    ls80: pd.Series,
    gold: pd.Series,
    w_gold: float,
    capital: float,
    rebalance: bool = True
) -> pd.Series:
    """
    Portfolio di 2 asset con ribilanciamento annuale (prima data di ogni anno presente nel dataset).
    w_gold in [0,1], resto su LS80.
    """
    w_gold = float(np.clip(w_gold, 0.0, 1.0))
    w_ls80 = 1.0 - w_gold

    df = pd.DataFrame({"ls80": ls80, "gold": gold}).dropna()
    if len(df) < 5:
        raise ValueError("Dati insufficienti per il calcolo (meno di 5 righe dopo allineamento).")

    # rebalance dates
    reb_dates = set(_first_trading_day_each_year(df.index)) if rebalance else set()

    # inizializza quote
    p0 = df.iloc[0]
    shares_ls80 = (capital * w_ls80) / p0["ls80"] if p0["ls80"] > 0 else 0.0
    shares_gold = (capital * w_gold) / p0["gold"] if p0["gold"] > 0 else 0.0

    values = []
    for dt, row in df.iterrows():
        total = shares_ls80 * row["ls80"] + shares_gold * row["gold"]
        values.append(total)

        if rebalance and (dt in reb_dates) and (dt != df.index[0]):
            # ribilancia alla chiusura del giorno (semplice e coerente)
            target_ls80 = total * w_ls80
            target_gold = total * w_gold
            shares_ls80 = target_ls80 / row["ls80"] if row["ls80"] > 0 else 0.0
            shares_gold = target_gold / row["gold"] if row["gold"] > 0 else 0.0

    return pd.Series(values, index=df.index, name="portfolio")

def _max_drawdown(series: pd.Series) -> float:
    s = series.astype(float)
    peak = s.cummax()
    dd = (s / peak) - 1.0
    return float(dd.min())

def _annualized_return(series: pd.Series) -> float:
    s = series.astype(float)
    if len(s) < 2:
        return float("nan")
    start = float(s.iloc[0])
    end = float(s.iloc[-1])
    if start <= 0:
        return float("nan")
    years = (s.index[-1] - s.index[0]).days / 365.25
    if years <= 0:
        return float("nan")
    return (end / start) ** (1.0 / years) - 1.0

def _years_between(idx: pd.DatetimeIndex) -> float:
    if len(idx) < 2:
        return 0.0
    return (idx[-1] - idx[0]).days / 365.25

def _doubling_years(ann_return: float) -> Optional[float]:
    try:
        if ann_return is None or not np.isfinite(ann_return) or ann_return <= 0:
            return None
        return float(math.log(2.0) / math.log(1.0 + ann_return))
    except Exception:
        return None

def _to_datestr_list(idx: pd.DatetimeIndex) -> List[str]:
    # lasciamo ISO (YYYY-MM-DD) e app.js le normalizza/ticka
    return [d.strftime("%Y-%m-%d") for d in idx.to_pydatetime()]

# =========================
# Rate limit assistente (in RAM)
# =========================
_ASK_STATE: Dict[str, Dict[str, int]] = {}
# struttura: _ASK_STATE[ip][YYYY-MM-DD] = count

def _client_ip() -> str:
    # Render / proxy: X-Forwarded-For
    xff = request.headers.get("X-Forwarded-For", "")
    if xff:
        return xff.split(",")[0].strip()
    return request.remote_addr or "unknown"

def _ask_check_and_increment() -> Tuple[bool, int, int]:
    ip = _client_ip()
    today = date.today().isoformat()
    per_ip = _ASK_STATE.setdefault(ip, {})
    used = int(per_ip.get(today, 0))
    if used >= ASK_DAILY_LIMIT:
        return False, 0, ASK_DAILY_LIMIT
    used += 1
    per_ip[today] = used
    remaining = ASK_DAILY_LIMIT - used
    return True, remaining, ASK_DAILY_LIMIT

def _ask_remaining() -> Tuple[int, int]:
    ip = _client_ip()
    today = date.today().isoformat()
    used = int(_ASK_STATE.get(ip, {}).get(today, 0))
    remaining = max(0, ASK_DAILY_LIMIT - used)
    return remaining, ASK_DAILY_LIMIT

# =========================
# Pages
# =========================
@app.get("/")
def home():
    return render_template("index.html")

@app.get("/health")
def health():
    return jsonify({"ok": True})

# =========================
# API: compute
# =========================
@app.get("/api/compute")
def api_compute():
    """
    Parametri attesi (come nel tuo app.js):
    - w_ls80: percentuale LS80 (0..100)
    - w_gold: percentuale Oro (0..100)
    - capital: capitale iniziale
    """
    try:
        w_ls80 = _safe_float(request.args.get("w_ls80"), None)
        w_gold = _safe_float(request.args.get("w_gold"), None)
        capital = _safe_float(request.args.get("capital"), 10000.0)

        if w_gold is None and w_ls80 is None:
            # fallback
            w_gold = 20.0
            w_ls80 = 80.0
        elif w_gold is None:
            w_gold = max(0.0, 100.0 - float(w_ls80))
        elif w_ls80 is None:
            w_ls80 = max(0.0, 100.0 - float(w_gold))

        # normalizza a 100 (per sicurezza)
        tot = float(w_ls80) + float(w_gold)
        if tot <= 0:
            w_ls80, w_gold = 80.0, 20.0
            tot = 100.0
        w_ls80 = float(w_ls80) / tot * 100.0
        w_gold = float(w_gold) / tot * 100.0

        # carica dati
        df_ls80 = _load_price_csv(LS80_CSV)
        df_gold = _load_price_csv(GOLD_CSV)

        # allinea
        joined = pd.concat([df_ls80["close"].rename("ls80"), df_gold["close"].rename("gold")], axis=1).dropna()
        if len(joined) < 10:
            return jsonify({"error": "Dati insufficienti dopo l'allineamento (ls80/gold)."}), 500

        # portfolio 2 strumenti
        w_gold_f = float(w_gold) / 100.0
        series = _compute_portfolio_series(joined["ls80"], joined["gold"], w_gold=w_gold_f, capital=float(capital), rebalance=True)

        ann = _annualized_return(series)
        mdd = _max_drawdown(series)
        years = _years_between(series.index)
        final_val = float(series.iloc[-1])

        # composizione risultante: LS80 si “spacca” in 80/20
        equity_pct = (w_ls80 * 0.80)
        bond_pct = (w_ls80 * 0.20)
        gold_pct = w_gold

        dbl = _doubling_years(ann)

        metrics = {
            "cagr_portfolio": ann,  # decimale (0.12 = 12%)
            "max_dd_portfolio": mdd,
            "final_portfolio": final_val,
            "final_years": years,
            "doubling_years_portfolio": dbl,
            "composition": {
                "equity_pct": equity_pct,
                "bond_pct": bond_pct,
                "gold_pct": gold_pct,
            },
        }

        return jsonify(
            {
                "dates": _to_datestr_list(series.index),
                "portfolio": [float(x) for x in series.values],
                "metrics": metrics,
            }
        )

    except Exception as e:
        app.logger.exception("Errore in /api/compute")
        return jsonify({"error": f"Errore compute: {_format_exc(e)}"}), 500

# =========================
# API: assistant
# =========================
@app.post("/api/ask")
def api_ask():
    """
    JSON input: { "question": "..." }
    JSON output:
      { ok: true, answer: "...", remaining: x, limit: 10 }
    or
      { ok: false, error: "...", remaining: x, limit: 10 }
    """
    try:
        payload = request.get_json(silent=True) or {}
        q = (payload.get("question") or "").strip()

        if not q:
            remaining, limit = _ask_remaining()
            return jsonify({"ok": False, "error": "Scrivi una domanda.", "remaining": remaining, "limit": limit}), 400

        allowed, remaining, limit = _ask_check_and_increment()
        if not allowed:
            return jsonify(
                {"ok": False, "error": f"Hai raggiunto il limite di {limit} domande per oggi. Riprova domani.", "remaining": 0, "limit": limit}
            ), 429

        if not OPENAI_API_KEY:
            return jsonify(
                {"ok": False, "error": "Assistente non configurato: manca OPENAI_API_KEY su Render.", "remaining": remaining, "limit": limit}
            ), 500

        # OpenAI SDK (nuovo stile)
        try:
            from openai import OpenAI  # type: ignore
            client = OpenAI(api_key=OPENAI_API_KEY)

            system = (
                "Rispondi in italiano, in modo chiaro e pratico. "
                "Contesto: sito educativo su una strategia 'Metodo Pigro' (ETF Azion-Obblig + ETC Oro). "
                "Non fornire consulenza finanziaria personalizzata; mantieni tono informativo."
            )

            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": q},
                ],
                temperature=0.4,
                max_tokens=300,
            )
            answer = (resp.choices[0].message.content or "").strip()

        except Exception as oe:
            # IMPORTANTISSIMO: sempre JSON, niente HTML
            app.logger.exception("Errore OpenAI in /api/ask")
            return jsonify({"ok": False, "error": f"Errore assistente: {_format_exc(oe)}", "remaining": remaining, "limit": limit}), 500

        return jsonify({"ok": True, "answer": answer, "remaining": remaining, "limit": limit})

    except Exception as e:
        app.logger.exception("Errore in /api/ask")
        remaining, limit = _ask_remaining()
        return jsonify({"ok": False, "error": f"Errore server: {_format_exc(e)}", "remaining": remaining, "limit": limit}), 500

# =========================
# API: update_data (cron)
# =========================
def _stooq_download_csv(symbol: str) -> Optional[str]:
    """
    Stooq endpoint: https://stooq.com/q/d/l/?s=xxx& i=d
    Per alcuni simboli europei funziona, per altri no.
    """
    url = f"https://stooq.com/q/d/l/?s={symbol.lower()}&i=d"
    r = requests.get(url, timeout=20)
    if r.status_code != 200 or not r.text or "Date" not in r.text:
        return None
    return r.text

def _try_download_prices(symbols: List[str]) -> Tuple[Optional[pd.DataFrame], str]:
    last_err = ""
    for sym in symbols:
        try:
            csv_text = _stooq_download_csv(sym)
            if not csv_text:
                last_err = f"stooq no data for {sym}"
                continue
            df = pd.read_csv(io.StringIO(csv_text))
            if "Date" not in df.columns or "Close" not in df.columns:
                last_err = f"stooq invalid columns for {sym}: {df.columns}"
                continue
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
            df = df.dropna(subset=["Date", "Close"]).sort_values("Date")
            return df[["Date", "Close"]].copy(), sym
        except Exception as e:
            last_err = _format_exc(e)
            continue
    return None, last_err

def _append_new_rows_semicolon(csv_path: Path, new_df: pd.DataFrame) -> Tuple[bool, str]:
    """
    Appende nuove righe in formato Date;Close (DD/MM/YYYY;valore)
    """
    try:
        old = _load_price_csv(csv_path)
        last_date = old.index.max()
    except Exception:
        old = None
        last_date = None

    new_df = new_df.copy()
    new_df["Date"] = pd.to_datetime(new_df["Date"], errors="coerce")
    new_df = new_df.dropna(subset=["Date"])
    if last_date is not None:
        new_df = new_df[new_df["Date"] > last_date]

    if new_df.empty:
        return False, "already_up_to_date"

    lines = []
    for _, r in new_df.iterrows():
        d = pd.Timestamp(r["Date"]).to_pydatetime().date()
        d_str = d.strftime("%d/%m/%Y")
        c = float(r["Close"])
        # manteniamo punto decimale, come i tuoi file
        lines.append(f"{d_str};{c:.2f}")

    # se file non esiste, crea header
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if not csv_path.exists():
        csv_path.write_text("Date;Close\n", encoding="utf-8")

    with csv_path.open("a", encoding="utf-8") as f:
        for ln in lines:
            f.write(ln + "\n")
    return True, f"appended_{len(lines)}"

@app.get("/api/update_data")
def api_update_data():
    """
    Protezione con token (querystring): /api/update_data?token=...
    Aggiorna data/ls80.csv e data/gold.csv.
    """
    try:
        token = (request.args.get("token") or "").strip()
        if not UPDATE_TOKEN or token != UPDATE_TOKEN:
            return jsonify({"error": "forbidden"}), 403

        last_ts = _read_last_update_ts()
        if last_ts is not None:
            hours = (_now_ts() - last_ts) / 3600.0
            if hours < UPDATE_MIN_INTERVAL_HOURS:
                return jsonify({"skipped": True, "reason": "min_interval", "hours_since_last": hours}), 200

        # Proviamo più simboli (come avevamo discusso)
        ls80_candidates = [LS80_TICKER, LS80_TICKER.replace(".MI", ""), LS80_TICKER.lower()]
        gold_candidates = [GOLD_TICKER, f"{GOLD_TICKER}.MI", f"{GOLD_TICKER}.L", GOLD_TICKER.lower()]

        out = {"ls80": None, "gold": None, "warnings": []}

        df_ls, used_ls = _try_download_prices(ls80_candidates)
        if df_ls is None:
            out["ls80"] = {"asset": "ls80", "updated": False, "skipped": True, "reason": "download_failed", "detail": "no data"}
        else:
            updated, reason = _append_new_rows_semicolon(LS80_CSV, df_ls)
            out["ls80"] = {"asset": "ls80", "ticker_used": used_ls, "updated": updated, "skipped": (not updated), "reason": reason}

        df_g, used_g = _try_download_prices(gold_candidates)
        if df_g is None:
            out["gold"] = {"asset": "gold", "updated": False, "skipped": True, "reason": "download_failed", "detail": "no data"}
        else:
            updated, reason = _append_new_rows_semicolon(GOLD_CSV, df_g)
            out["gold"] = {"asset": "gold", "ticker_used": used_g, "updated": updated, "skipped": (not updated), "reason": reason}

        _write_last_update_ts(_now_ts())
        return jsonify(out), 200

    except Exception as e:
        app.logger.exception("Errore in /api/update_data")
        return jsonify({"error": _format_exc(e)}), 500

# =========================
# Diagnostics
# =========================
@app.get("/api/diag")
def api_diag():
    """
    Diagnostica rapida: verifica env, presenza file, parsing CSV, prime/ultime date.
    """
    diag: Dict[str, object] = {
        "ok": True,
        "env": {
            "LS80_TICKER": LS80_TICKER,
            "GOLD_TICKER": GOLD_TICKER,
            "OPENAI_API_KEY_set": bool(OPENAI_API_KEY),
            "UPDATE_TOKEN_set": bool(UPDATE_TOKEN),
            "UPDATE_MIN_INTERVAL_HOURS": UPDATE_MIN_INTERVAL_HOURS,
        },
        "paths": {
            "base_dir": str(BASE_DIR),
            "data_dir": str(DATA_DIR),
            "ls80_csv": str(LS80_CSV),
            "gold_csv": str(GOLD_CSV),
            "ls80_exists": LS80_CSV.exists(),
            "gold_exists": GOLD_CSV.exists(),
        },
        "csv": {},
        "compute_smoke": {},
    }

    try:
        ls = _load_price_csv(LS80_CSV)
        gd = _load_price_csv(GOLD_CSV)
        diag["csv"]["ls80"] = {
            "rows": int(len(ls)),
            "first": ls.index.min().strftime("%Y-%m-%d"),
            "last": ls.index.max().strftime("%Y-%m-%d"),
            "head_close": float(ls["close"].iloc[0]),
            "tail_close": float(ls["close"].iloc[-1]),
        }
        diag["csv"]["gold"] = {
            "rows": int(len(gd)),
            "first": gd.index.min().strftime("%Y-%m-%d"),
            "last": gd.index.max().strftime("%Y-%m-%d"),
            "head_close": float(gd["close"].iloc[0]),
            "tail_close": float(gd["close"].iloc[-1]),
        }

        joined = pd.concat([ls["close"].rename("ls80"), gd["close"].rename("gold")], axis=1).dropna()
        diag["csv"]["joined_rows"] = int(len(joined))
        diag["csv"]["joined_first"] = joined.index.min().strftime("%Y-%m-%d")
        diag["csv"]["joined_last"] = joined.index.max().strftime("%Y-%m-%d")

        # smoke test compute
        series = _compute_portfolio_series(joined["ls80"], joined["gold"], w_gold=0.2, capital=10000.0, rebalance=True)
        diag["compute_smoke"] = {
            "rows": int(len(series)),
            "ann": _annualized_return(series),
            "mdd": _max_drawdown(series),
            "final": float(series.iloc[-1]),
        }

    except Exception as e:
        diag["ok"] = False
        diag["error"] = _format_exc(e)
        diag["trace"] = traceback.format_exc().splitlines()[-10:]  # ultime righe

    return jsonify(diag)

# =========================
# Static (optional fallback)
# =========================
@app.get("/static/<path:filename>")
def static_files(filename: str):
    return send_from_directory(str(BASE_DIR / "static"), filename)


if __name__ == "__main__":
    # utile in locale
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "10000")), debug=True)
