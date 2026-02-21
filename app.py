from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import time
import urllib.request
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

from flask import Flask, jsonify, make_response, render_template, request

app = Flask(__name__, template_folder="templates", static_folder="static")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

ASSET_FILES: Dict[str, str] = {
    "ls80": "ls80.csv",
    "gold": "gold.csv",
}

# =========================
# ENV / CONFIG
# =========================
LS80_TICKER = (os.getenv("LS80_TICKER") or "VNGA80.MI").strip()
GOLD_TICKER = (os.getenv("GOLD_TICKER") or "SGLD").strip()

UPDATE_TOKEN = (os.getenv("UPDATE_TOKEN") or "").strip()
UPDATE_MIN_INTERVAL_HOURS = float(os.getenv("UPDATE_MIN_INTERVAL_HOURS") or "6")

OPENAI_API_KEY = (os.getenv("OPENAI_API_KEY") or "").strip()
OPENAI_MODEL = (os.getenv("OPENAI_MODEL") or "gpt-4o-mini").strip()

ASK_DAILY_LIMIT = int(os.getenv("ASK_DAILY_LIMIT") or "10")
ASK_SALT = (os.getenv("ASK_SALT") or UPDATE_TOKEN or "gloob_salt").strip()
ASK_STORE_PATH = DATA_DIR / "ask_limits.json"

_last_update_ts_path = DATA_DIR / ".last_update_ts.json"


# =========================
# UTILS: CSV
# =========================
def _detect_and_read_csv(path: Path) -> Tuple[List[date], List[float]]:
    if not path.exists():
        raise FileNotFoundError(f"File dati mancante: {path.name}")

    # Supporta separatori comuni e header
    with path.open("r", encoding="utf-8", newline="") as f:
        sample = f.read(4096)
        f.seek(0)
        dialect = csv.Sniffer().sniff(sample, delimiters=";,")
        reader = csv.reader(f, dialect)

        rows = list(reader)

    # prova a saltare header se presente
    start_idx = 0
    if rows and rows[0] and ("date" in rows[0][0].lower() or "data" in rows[0][0].lower()):
        start_idx = 1

    dates: List[date] = []
    vals: List[float] = []

    for r in rows[start_idx:]:
        if not r or len(r) < 2:
            continue
        ds = r[0].strip()
        vs = r[1].strip().replace(",", ".")
        try:
            d = datetime.fromisoformat(ds).date()
            v = float(vs)
        except Exception:
            continue
        dates.append(d)
        vals.append(v)

    if len(dates) < 10:
        raise ValueError(f"Serie troppo corta in {path.name}")

    return dates, vals


# =========================
# UPDATE DATA (Yahoo via CSV download / fallback)
# =========================
def _read_last_update_ts() -> float:
    try:
        if _last_update_ts_path.exists():
            obj = json.loads(_last_update_ts_path.read_text(encoding="utf-8"))
            return float(obj.get("ts", 0.0))
    except Exception:
        pass
    return 0.0


def _write_last_update_ts(ts: float) -> None:
    try:
        _last_update_ts_path.write_text(json.dumps({"ts": ts}), encoding="utf-8")
    except Exception:
        pass


def _fetch_yahoo_csv(ticker: str, out_path: Path) -> Dict[str, str]:
    """
    Scarica storico giornaliero da Yahoo Finance (download CSV).
    Nota: Yahoo può cambiare formato; questo è un approccio pragmatico.
    """
    now = int(time.time())
    # periodo: ultimi ~20 anni
    period1 = now - int(60 * 60 * 24 * 365.25 * 20)
    period2 = now

    url = (
        "https://query1.finance.yahoo.com/v7/finance/download/"
        + urllib.request.quote(ticker)
        + f"?period1={period1}&period2={period2}&interval=1d&events=history&includeAdjustedClose=true"
    )

    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept": "text/csv",
        },
    )

    with urllib.request.urlopen(req, timeout=25) as resp:
        raw = resp.read().decode("utf-8", errors="ignore")

    # Yahoo header tipico: Date,Open,High,Low,Close,Adj Close,Volume
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    if len(lines) < 3 or not lines[0].lower().startswith("date"):
        raise ValueError("CSV Yahoo non valido (header inatteso).")

    # Convertiamo in formato semplice: date,value (usiamo Adj Close se presente)
    header = lines[0].split(",")
    try:
        date_i = header.index("Date")
        adj_i = header.index("Adj Close")
    except Exception:
        # fallback su Close
        date_i = 0
        adj_i = 4

    out_rows = [["date", "value"]]
    for ln in lines[1:]:
        parts = ln.split(",")
        if len(parts) <= max(date_i, adj_i):
            continue
        ds = parts[date_i].strip()
        vs = parts[adj_i].strip()
        if vs in ("null", "", "NaN"):
            continue
        out_rows.append([ds, vs])

    out_path.write_text("\n".join([",".join(r) for r in out_rows]) + "\n", encoding="utf-8")
    return {"ticker": ticker, "rows": str(len(out_rows) - 1)}


def _maybe_update_one(asset_key: str, ticker: str) -> Dict[str, object]:
    filename = ASSET_FILES[asset_key]
    path = DATA_DIR / filename

    # se file esiste, guardiamo l’ultima data
    last_date = None
    try:
        d, _v = _detect_and_read_csv(path)
        last_date = d[-1]
    except Exception:
        last_date = None

    tickers_to_try = [ticker]
    if asset_key == "gold":
        # euristica: se metti "SGLD" proviamo anche "SGLD.MI" e "SGLD.L"
        if "." not in ticker:
            tickers_to_try += [f"{ticker}.MI", f"{ticker}.L"]

    last_err = None
    for t in tickers_to_try:
        try:
            info = _fetch_yahoo_csv(t, path)
            # rilegge e verifica aggiornamento
            d2, _v2 = _detect_and_read_csv(path)
            updated = (last_date is None) or (d2[-1] != last_date)
            return {
                "asset": asset_key,
                "ticker_used": t,
                "updated": bool(updated),
                "reason": "ok" if updated else "already_up_to_date",
                "rows": info.get("rows"),
            }
        except Exception as e:
            last_err = e

    return {
        "asset": asset_key,
        "ticker_used": tickers_to_try[-1],
        "updated": False,
        "reason": f"failed: {last_err}",
    }


def update_assets_if_due(force: bool = False) -> Dict[str, object]:
    now = time.time()
    last = _read_last_update_ts()
    min_seconds = UPDATE_MIN_INTERVAL_HOURS * 3600.0

    if (not force) and last > 0 and (now - last) < min_seconds:
        return {"skipped": True, "reason": "too_soon", "seconds_since": now - last}

    res: Dict[str, object] = {"skipped": False, "warnings": []}
    try:
        res["ls80"] = _maybe_update_one("ls80", LS80_TICKER)
    except Exception as e:
        res["warnings"].append(f"Update ls80 fallito: {e}")

    try:
        res["gold"] = _maybe_update_one("gold", GOLD_TICKER)
    except Exception as e:
        res["warnings"].append(f"Update gold fallito: {e}")

    _write_last_update_ts(now)
    return res


@app.get("/api/update_data")
def api_update_data():
    token = (request.args.get("token") or "").strip()
    if not UPDATE_TOKEN or token != UPDATE_TOKEN:
        return jsonify({"error": "Unauthorized"}), 401
    return jsonify(update_assets_if_due(force=True))


# =========================
# PORTAFOGLIO / COMPUTE
# =========================
def _align_series(d1: List[date], v1: List[float], d2: List[date], v2: List[float]):
    m1 = dict(zip(d1, v1))
    m2 = dict(zip(d2, v2))
    common = sorted(set(m1.keys()) & set(m2.keys()))
    if len(common) < 10:
        raise ValueError("Serie troppo corte o poche date in comune.")
    a1 = [m1[d] for d in common]
    a2 = [m2[d] for d in common]
    return common, a1, a2


def _years_between(d0: date, d1: date) -> float:
    return max(0.00001, (d1 - d0).days / 365.25)


def _compute_cagr(v0: float, v1: float, years: float) -> float:
    if v0 <= 0 or v1 <= 0 or years <= 0:
        return float("nan")
    return (v1 / v0) ** (1 / years) - 1


def _max_drawdown(values: List[float]) -> float:
    peak = -1e30
    mdd = 0.0
    for v in values:
        peak = max(peak, v)
        if peak > 0:
            dd = (v / peak) - 1.0
            mdd = min(mdd, dd)
    return mdd


def _build_portfolio(
    dates: List[date],
    ls_vals: List[float],
    g_vals: List[float],
    w_ls80: float,
    w_gold: float,
    capital: float,
):
    w_ls80 = float(w_ls80)
    w_gold = float(w_gold)
    s = w_ls80 + w_gold
    if s <= 0:
        w_ls80, w_gold = 100.0, 0.0
        s = 100.0
    w_ls = (w_ls80 / s)
    w_g = (w_gold / s)

    # ribilanciamento annuale semplice
    ls_shares = (capital * w_ls) / ls_vals[0]
    g_shares = (capital * w_g) / g_vals[0]

    port: List[float] = []

    prev_year = dates[0].year
    for d, ls, g in zip(dates, ls_vals, g_vals):
        if d.year != prev_year:
            total = ls_shares * ls + g_shares * g
            ls_shares = (total * w_ls) / ls
            g_shares = (total * w_g) / g
            prev_year = d.year

        port.append(ls_shares * ls + g_shares * g)

    return port


@app.get("/")
def home():
    resp = make_response(render_template("index.html"))
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    return resp


@app.get("/health")
def health():
    return jsonify({"ok": True})


@app.get("/api/compute")
def api_compute():
    try:
        # update "soft": prova (al massimo ogni X ore)
        try:
            update_assets_if_due(force=False)
        except Exception:
            pass

        w_ls80 = float(request.args.get("w_ls80", "90"))
        w_gold = float(request.args.get("w_gold", "10"))
        capital = float(request.args.get("capital", "10000"))

        d_ls, v_ls = _detect_and_read_csv(DATA_DIR / ASSET_FILES["ls80"])
        d_g, v_g = _detect_and_read_csv(DATA_DIR / ASSET_FILES["gold"])
        dates, ls, g = _align_series(d_ls, v_ls, d_g, v_g)

        port_vals = _build_portfolio(dates, ls, g, w_ls80, w_gold, capital)

        years = _years_between(dates[0], dates[-1])
        cagr_port = _compute_cagr(port_vals[0], port_vals[-1], years)
        mdd_port = _max_drawdown(port_vals)

        yd = float("nan")
        if math.isfinite(cagr_port) and cagr_port > 0:
            yd = math.log(2) / math.log(1 + cagr_port)

        payload = {
            "dates": [d.isoformat() for d in dates],
            "portfolio": port_vals,
            "metrics": {
                "cagr_portfolio": cagr_port,
                "max_dd_portfolio": mdd_port,
                "years_to_double": yd,
                "start": dates[0].isoformat(),
                "end": dates[-1].isoformat(),
                "final_portfolio": port_vals[-1],
                "years_total": years,
            },
        }

        r = make_response(jsonify(payload))
        r.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        r.headers["Pragma"] = "no-cache"
        return r

    except Exception as e:
        return jsonify({"error": str(e)}), 400


# =========================
# ASSISTENTE /api/ask (nuovo)
# =========================
def _client_ip() -> str:
    xf = request.headers.get("X-Forwarded-For", "")
    if xf:
        # prima IP della lista
        return xf.split(",")[0].strip()
    return (request.remote_addr or "unknown").strip()


def _client_id_hash() -> str:
    raw = (_client_ip() + "|" + ASK_SALT).encode("utf-8", errors="ignore")
    return hashlib.sha256(raw).hexdigest()[:32]


def _today_key_utc() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def _load_ask_store() -> Dict[str, Dict[str, int]]:
    try:
        if ASK_STORE_PATH.exists():
            obj = json.loads(ASK_STORE_PATH.read_text(encoding="utf-8"))
            if isinstance(obj, dict):
                return obj  # type: ignore
    except Exception:
        pass
    return {}


def _save_ask_store(store: Dict[str, Dict[str, int]]) -> None:
    try:
        ASK_STORE_PATH.write_text(json.dumps(store), encoding="utf-8")
    except Exception:
        pass


def _ask_check_and_increment() -> Tuple[bool, int, int]:
    """
    Ritorna: (allowed, remaining, limit)
    """
    day = _today_key_utc()
    cid = _client_id_hash()
    store = _load_ask_store()

    if day not in store:
        store = {day: {}}  # reset giornaliero (semplice)

    used = int(store[day].get(cid, 0))
    limit = ASK_DAILY_LIMIT

    if used >= limit:
        remaining = 0
        return False, remaining, limit

    used += 1
    store[day][cid] = used
    _save_ask_store(store)

    remaining = max(0, limit - used)
    return True, remaining, limit


@app.post("/api/ask")
def api_ask():
    # Risposta SEMPRE JSON (evita “Unexpected token <”)
    try:
        payload = request.get_json(silent=True) or {}
        q = (payload.get("question") or payload.get("q") or "").strip()
        context = payload.get("context") or {}

        if not q:
            return jsonify({"error": "Scrivi una domanda prima di inviare.", "remaining": None, "limit": ASK_DAILY_LIMIT}), 400

        # micro-protezione base
        if len(q) > 600:
            q = q[:600].strip()

        allowed, remaining, limit = _ask_check_and_increment()
        if not allowed:
            return jsonify(
                {
                    "error": f"Hai raggiunto il limite di {limit} domande per oggi. Riprova domani.",
                    "remaining": 0,
                    "limit": limit,
                }
            ), 429

        if not OPENAI_API_KEY:
            return jsonify(
                {
                    "error": "Assistente non configurato: manca OPENAI_API_KEY su Render.",
                    "remaining": remaining,
                    "limit": limit,
                }
            ), 500

        # chiamata OpenAI
        try:
            from openai import OpenAI  # type: ignore
        except Exception:
            return jsonify(
                {
                    "error": "Server: libreria OpenAI non disponibile (requirements).",
                    "remaining": remaining,
                    "limit": limit,
                }
            ), 500

        client = OpenAI(api_key=OPENAI_API_KEY)

        # contesto “soft” (niente dati sensibili)
        ctx_str = ""
        if isinstance(context, dict) and context:
            parts = []
            for k, v in context.items():
                parts.append(f"{k}: {v}")
            ctx_str = "\nContesto:\n" + "\n".join(parts)

        system = (
            "Sei un assistente che spiega la finanza in modo semplice e prudente.\n"
            "Rispondi in italiano, in modo chiaro e sintetico.\n"
            "Niente consulenza personalizzata: solo informazioni generali.\n"
            "Se la domanda è ambigua, fai 1 domanda di chiarimento.\n"
        )

        user_msg = f"Domanda:\n{q}\n{ctx_str}".strip()

        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.4,
            max_tokens=350,
        )

        answer = (resp.choices[0].message.content or "").strip()
        if not answer:
            answer = "(nessuna risposta)"

        return jsonify({"answer": answer, "remaining": remaining, "limit": limit})

    except Exception as e:
        return jsonify({"error": f"Errore assistente: {e}"}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
