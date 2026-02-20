from __future__ import annotations

import csv
import json
import math
import os
import time
import urllib.error
import urllib.request
from collections import deque
from datetime import date, datetime
from pathlib import Path
from typing import Deque, Dict, List, Tuple

from flask import Flask, jsonify, make_response, render_template, request

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0  # no cache per static

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

ASSET_FILES: Dict[str, str] = {
    "ls80": "ls80.csv",
    "gold": "gold.csv",
}

# -----------------------------
# Rate limit assistente (in-memory)
# -----------------------------
# Limite: 10 richieste / 24h / IP (rolling)
ASK_LIMIT_PER_DAY = int(os.getenv("ASK_LIMIT_PER_DAY", "10"))
ASK_COOLDOWN_SECONDS = int(os.getenv("ASK_COOLDOWN_SECONDS", "8"))

_ask_hits_day: Dict[str, Deque[float]] = {}
_last_ask_at: Dict[str, float] = {}


def _client_ip() -> str:
    # Render spesso mette X-Forwarded-For
    xff = request.headers.get("X-Forwarded-For", "")
    if xff:
        return xff.split(",")[0].strip()
    return request.remote_addr or "unknown"


def _check_ask_limits(ip: str) -> Tuple[bool, str, int]:
    """
    Ritorna: (ok, messaggio_errore, remaining_today)
    remaining_today è sempre 0..ASK_LIMIT_PER_DAY
    """
    now = time.time()

    # Cooldown anti-click spam
    last = _last_ask_at.get(ip, 0.0)
    if (now - last) < ASK_COOLDOWN_SECONDS:
        wait = int(math.ceil(ASK_COOLDOWN_SECONDS - (now - last)))
        remaining = _remaining_today(ip, now)
        return False, f"Troppo veloce 🙂 Riprova tra {wait} secondi.", remaining

    # Limite rolling 24h
    window = 86400.0  # 24 ore
    q = _ask_hits_day.get(ip)
    if q is None:
        q = deque()
        _ask_hits_day[ip] = q

    while q and (now - q[0]) > window:
        q.popleft()

    if len(q) >= ASK_LIMIT_PER_DAY:
        return False, "Hai raggiunto il limite di 10 domande oggi. Riprova domani.", 0

    # Consuma una richiesta
    q.append(now)
    _last_ask_at[ip] = now

    remaining = max(0, ASK_LIMIT_PER_DAY - len(q))
    return True, "", remaining


def _remaining_today(ip: str, now: float | None = None) -> int:
    if now is None:
        now = time.time()
    window = 86400.0
    q = _ask_hits_day.get(ip)
    if not q:
        return ASK_LIMIT_PER_DAY
    while q and (now - q[0]) > window:
        q.popleft()
    return max(0, ASK_LIMIT_PER_DAY - len(q))


def _parse_float(s: str) -> float:
    s = (s or "").strip()
    if s == "":
        return float("nan")
    if s.count(",") and s.count("."):
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    else:
        if s.count(",") and not s.count("."):
            s = s.replace(",", ".")
    return float(s)


def _parse_date(s: str) -> date:
    s = (s or "").strip()
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%m/%d/%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            pass
    raise ValueError(f"Formato data non riconosciuto: {s}")


def _detect_and_read_csv(path: Path) -> Tuple[List[date], List[float]]:
    if not path.exists():
        raise FileNotFoundError(f"File mancante: {path}")

    dates: List[date] = []
    values: List[float] = []

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        sample = f.read(2048)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=";,|\t,")
        except Exception:
            dialect = csv.excel

        reader = csv.reader(f, dialect)
        rows = list(reader)

    # header?
    start_i = 0
    if rows and any("date" in (c or "").lower() for c in rows[0]):
        start_i = 1

    for r in rows[start_i:]:
        if not r or len(r) < 2:
            continue
        d = _parse_date(r[0])
        v = _parse_float(r[1])
        if not math.isfinite(v):
            continue
        dates.append(d)
        values.append(float(v))

    if not dates:
        raise ValueError(f"Nessun dato valido in {path}")

    return dates, values


def _align_series(
    d1: List[date], v1: List[float], d2: List[date], v2: List[float]
) -> Tuple[List[date], List[float], List[float]]:
    m1 = dict(zip(d1, v1))
    m2 = dict(zip(d2, v2))
    common = sorted(set(m1.keys()) & set(m2.keys()))
    if len(common) < 10:
        raise ValueError("Serie troppo corte o poche date in comune.")
    a1 = [m1[d] for d in common]
    a2 = [m2[d] for d in common]
    return common, a1, a2


def _compute_cagr(start_val: float, end_val: float, years: float) -> float:
    if start_val <= 0 or end_val <= 0 or years <= 0:
        return float("nan")
    return (end_val / start_val) ** (1 / years) - 1


def _max_drawdown(values: List[float]) -> float:
    peak = -1e18
    mdd = 0.0
    for x in values:
        if x > peak:
            peak = x
        if peak > 0:
            dd = (x / peak) - 1.0
            if dd < mdd:
                mdd = dd
    return mdd


def _years_between(d0: date, d1: date) -> float:
    return (d1.toordinal() - d0.toordinal()) / 365.25


def _build_portfolio(
    dates: List[date],
    ls80_vals: List[float],
    gold_vals: List[float],
    w_ls80_pct: float,
    w_gold_pct: float,
    capital: float,
) -> Tuple[List[float], List[float]]:
    w_ls = w_ls80_pct / 100.0
    w_g = w_gold_pct / 100.0

    if w_ls < 0 or w_g < 0 or (w_ls + w_g) <= 0:
        raise ValueError("Pesi non validi")

    # normalizza per sicurezza
    s = w_ls + w_g
    w_ls /= s
    w_g /= s

    ls0 = ls80_vals[0]
    g0 = gold_vals[0]
    if ls0 <= 0 or g0 <= 0:
        raise ValueError("Valori iniziali non validi")

    ls_eur = [capital * w_ls * (x / ls0) for x in ls80_vals]
    g_eur = [capital * w_g * (x / g0) for x in gold_vals]

    port = [ls_eur[i] + g_eur[i] for i in range(len(dates))]
    solo = [capital * (x / ls0) for x in ls80_vals]
    return port, solo


@app.get("/health")
def health():
    return jsonify({"ok": True})


@app.get("/")
def home():
    resp = make_response(render_template("index.html"))
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp


@app.get("/api/compute")
def api_compute():
    try:
        w_ls80 = float(request.args.get("w_ls80", "90"))
        w_gold = float(request.args.get("w_gold", "10"))
        capital = float(request.args.get("capital", "10000"))

        d_ls, v_ls = _detect_and_read_csv(DATA_DIR / ASSET_FILES["ls80"])
        d_g, v_g = _detect_and_read_csv(DATA_DIR / ASSET_FILES["gold"])

        dates, ls, g = _align_series(d_ls, v_ls, d_g, v_g)

        port_vals, solo_vals = _build_portfolio(dates, ls, g, w_ls80, w_gold, capital)

        years = _years_between(dates[0], dates[-1])
        cagr_port = _compute_cagr(port_vals[0], port_vals[-1], years)
        cagr_solo = _compute_cagr(solo_vals[0], solo_vals[-1], years)

        mdd_port = _max_drawdown(port_vals)
        mdd_solo = _max_drawdown(solo_vals)

        yd = float("nan")
        if math.isfinite(cagr_port) and cagr_port > 0:
            yd = math.log(2) / math.log(1 + cagr_port)

        payload = {
            "dates": [d.isoformat() for d in dates],
            "portfolio": port_vals,
            "solo_ls80": solo_vals,  # compatibilità
            "metrics": {
                "cagr_portfolio": cagr_port,
                "cagr_solo": cagr_solo,
                "max_dd_portfolio": mdd_port,
                "max_dd_solo": mdd_solo,
                "years_to_double": yd,
                "start": dates[0].isoformat(),
                "end": dates[-1].isoformat(),
                "final_portfolio": port_vals[-1],
                "final_solo": solo_vals[-1],
            },
        }

        resp = make_response(jsonify(payload))
        resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        resp.headers["Pragma"] = "no-cache"
        resp.headers["Expires"] = "0"
        return resp

    except Exception as e:
        return jsonify({"error": str(e)}), 400


# -----------------------------
# CHAT ASSISTANT (OpenAI)
# -----------------------------
def _extract_text_from_responses_api(payload: dict) -> str:
    out = payload.get("output", [])
    for item in out:
        content = item.get("content", [])
        for c in content:
            if c.get("type") == "output_text" and c.get("text"):
                return str(c["text"]).strip()
            if c.get("type") == "text" and c.get("text"):
                return str(c["text"]).strip()
    if "text" in payload and payload["text"]:
        return str(payload["text"]).strip()
    return ""


def _openai_answer(question: str, context: dict) -> str:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Assistente non configurato: manca OPENAI_API_KEY su Render.")

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"

    system = (
        "Sei l'assistente del sito Gloob (Metodo Pigro). "
        "Rispondi in italiano, in modo chiaro e pratico, senza fare consulenza personalizzata. "
        "Non dare istruzioni su strumenti complessi, non fare promesse di rendimento. "
        "Se la domanda è ambigua, proponi 1-2 chiarimenti. "
        "In chiusura aggiungi sempre una riga di disclaimer: "
        "'Nota: risposta informativa, non consulenza finanziaria.'"
    )

    ctx_lines = []
    if isinstance(context, dict):
        oro = context.get("oro_pct")
        az = context.get("azionario_pct")
        ob = context.get("obbligazionario_pct")
        cap = context.get("capitale_eur")
        if oro is not None and az is not None and ob is not None:
            ctx_lines.append(
                f"Composizione attuale: Azionario {az}% | Obbligazionario {ob}% | Oro {oro}%."
            )
        if cap is not None:
            ctx_lines.append(f"Capitale iniziale indicato: {cap} €.")
    ctx_text = "\n".join(ctx_lines)

    user = (
        f"Domanda utente:\n{question}\n\n"
        f"Contesto:\n{ctx_text}\n"
        "Rispondi con massimo 10 righe, se possibile."
    )

    body = {
        "model": model,
        "input": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }

    req = urllib.request.Request(
        "https://api.openai.com/v1/responses",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read().decode("utf-8")
            data = json.loads(raw)
    except urllib.error.HTTPError as e:
        msg = e.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"Errore OpenAI HTTP {e.code}: {msg[:400]}")
    except Exception as e:
        raise RuntimeError(f"Errore chiamata OpenAI: {e}")

    text = _extract_text_from_responses_api(data)
    if not text:
        raise RuntimeError("Risposta vuota dall'assistente.")
    return text


@app.post("/api/ask")
def api_ask():
    try:
        ip = _client_ip()

        ok, msg, remaining = _check_ask_limits(ip)
        if not ok:
            return jsonify(
                {
                    "error": msg,
                    "remaining_today": remaining,
                    "limit_per_day": ASK_LIMIT_PER_DAY,
                }
            ), 429

        payload = request.get_json(silent=True) or {}
        q = (payload.get("question") or "").strip()
        if not q:
            # non consumiamo ulteriormente: abbiamo già “consumato” per semplicità,
            # ma è un caso rarissimo perché il client non manda vuoto.
            return jsonify({"error": "Scrivi una domanda.", "remaining_today": remaining, "limit_per_day": ASK_LIMIT_PER_DAY}), 400
        if len(q) > 800:
            return jsonify({"error": "Domanda troppo lunga (max 800 caratteri).", "remaining_today": remaining, "limit_per_day": ASK_LIMIT_PER_DAY}), 400

        context = payload.get("context") or {}
        answer = _openai_answer(q, context)

        return jsonify(
            {
                "answer": answer,
                "remaining_today": remaining,
                "limit_per_day": ASK_LIMIT_PER_DAY,
            }
        )

    except Exception as e:
        # in caso di errore “vero”, proviamo comunque a mostrare quota residua se possibile
        ip = _client_ip()
        rem = _remaining_today(ip)
        return jsonify({"error": str(e), "remaining_today": rem, "limit_per_day": ASK_LIMIT_PER_DAY}), 400


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
