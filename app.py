from __future__ import annotations

import base64
import csv
import io
import json
import math
import os
import time
import urllib.error
import urllib.request
from collections import deque
from datetime import date, datetime
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

from flask import Flask, jsonify, make_response, render_template, request, send_file

# PDF (ReportLab)
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0  # no cache per static

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

ASSET_FILES: Dict[str, str] = {
    "ls80": "ls80.csv",
    "gold": "gold.csv",
}

# -----------------------------
# Rate limit semplice (in-memory)
# -----------------------------
# 20 richieste / ora / IP
ASK_LIMIT_PER_HOUR = int(os.getenv("ASK_LIMIT_PER_HOUR", "20"))
_ask_hits: Dict[str, Deque[float]] = {}


def _client_ip() -> str:
    # Render spesso mette X-Forwarded-For
    xff = request.headers.get("X-Forwarded-For", "")
    if xff:
        return xff.split(",")[0].strip()
    return request.remote_addr or "unknown"


def _rate_limit_ok(ip: str) -> Tuple[bool, int]:
    now = time.time()
    window = 3600.0

    q = _ask_hits.get(ip)
    if q is None:
        q = deque()
        _ask_hits[ip] = q

    while q and (now - q[0]) > window:
        q.popleft()

    if len(q) >= ASK_LIMIT_PER_HOUR:
        return False, 0

    q.append(now)
    remaining = max(0, ASK_LIMIT_PER_HOUR - len(q))
    return True, remaining


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

    # Prova a capire se c’è header
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
    # Normalizzazione in € partendo da capital
    w_ls = w_ls80_pct / 100.0
    w_g = w_gold_pct / 100.0

    if w_ls < 0 or w_g < 0 or (w_ls + w_g) <= 0:
        raise ValueError("Pesi non validi")

    # normalizza per sicurezza (se arriva 95+10 ecc)
    s = w_ls + w_g
    w_ls /= s
    w_g /= s

    # Serie indice -> euro
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

        # Legge CSV
        d_ls, v_ls = _detect_and_read_csv(DATA_DIR / ASSET_FILES["ls80"])
        d_g, v_g = _detect_and_read_csv(DATA_DIR / ASSET_FILES["gold"])

        # Allinea
        dates, ls, g = _align_series(d_ls, v_ls, d_g, v_g)

        # Portafoglio
        port_vals, solo_vals = _build_portfolio(dates, ls, g, w_ls80, w_gold, capital)

        years = _years_between(dates[0], dates[-1])
        cagr_port = _compute_cagr(port_vals[0], port_vals[-1], years)
        cagr_solo = _compute_cagr(solo_vals[0], solo_vals[-1], years)

        mdd_port = _max_drawdown(port_vals)
        mdd_solo = _max_drawdown(solo_vals)

        # anni per raddoppio (se cagr>0)
        yd = float("nan")
        if math.isfinite(cagr_port) and cagr_port > 0:
            yd = math.log(2) / math.log(1 + cagr_port)

        payload = {
            "dates": [d.isoformat() for d in dates],
            "portfolio": port_vals,
            "solo_ls80": solo_vals,  # lasciamo compatibilità (anche se non lo mostri)
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
# PDF (stampa)
# -----------------------------
def _pct_str(x: Optional[float]) -> str:
    try:
        if x is None:
            return "—"
        if not math.isfinite(float(x)):
            return "—"
        return f"{float(x) * 100:.1f}%"
    except Exception:
        return "—"


def _euro_str(x: Optional[float]) -> str:
    try:
        if x is None:
            return "—"
        v = float(x)
        if not math.isfinite(v):
            return "—"
        return f"{v:,.0f} €".replace(",", "X").replace(".", ",").replace("X", ".")
    except Exception:
        return "—"


def _clean_data_url_png(data_url: str) -> bytes:
    if not data_url:
        raise ValueError("Manca immagine del grafico.")
    s = data_url.strip()
    prefix = "data:image/png;base64,"
    if s.startswith(prefix):
        s = s[len(prefix):]
    return base64.b64decode(s, validate=True)


@app.post("/api/pdf")
def api_pdf():
    """
    Riceve:
      - chart_png: dataURL PNG del canvas Chart.js
      - meta: titolo/sottotitolo e campi numerici (opzionali)
    Ritorna un PDF pronto per stampa (A4).
    """
    try:
        payload = request.get_json(silent=True) or {}
        chart_png = payload.get("chart_png", "")
        meta = payload.get("meta", {}) or {}

        img_bytes = _clean_data_url_png(chart_png)
        img = ImageReader(io.BytesIO(img_bytes))

        # A4 portrait
        page_w, page_h = A4
        buf = io.BytesIO()
        c = canvas.Canvas(buf, pagesize=A4)

        # Margini
        ml = 36
        mr = 36
        mt = 40
        mb = 36

        y = page_h - mt

        # Header con logo + brand
        logo_path = BASE_DIR / "static" / "logo.png"
        if logo_path.exists():
            try:
                logo = ImageReader(str(logo_path))
                c.drawImage(logo, ml, y - 44, width=40, height=40, mask="auto")
                x_text = ml + 48
            except Exception:
                x_text = ml
        else:
            x_text = ml

        c.setFont("Helvetica-Bold", 18)
        c.drawString(x_text, y - 18, "Gloob")
        c.setFont("Helvetica-Bold", 12)
        c.drawString(x_text, y - 36, "Metodo Pigro")

        # Data stampa
        c.setFont("Helvetica", 9)
        c.drawRightString(page_w - mr, y - 20, f"Stampato il {datetime.now().strftime('%d/%m/%Y')}")

        y -= 58

        # Meta / riepilogo
        oro = meta.get("oro_pct")
        az = meta.get("azionario_pct")
        ob = meta.get("obbligazionario_pct")
        cap0 = meta.get("capitale_iniziale")
        capf = meta.get("capitale_finale")
        anni = meta.get("anni_periodo")
        cagr = meta.get("cagr")
        mdd = meta.get("max_dd")
        ytd = meta.get("years_to_double")
        start = meta.get("start")
        end = meta.get("end")

        c.setFont("Helvetica-Bold", 11)
        c.drawString(ml, y, "Riepilogo")
        y -= 14
        c.setFont("Helvetica", 10)

        lines = []
        if start and end:
            lines.append(f"Periodo: {start} → {end}")
        if oro is not None and az is not None and ob is not None:
            lines.append(f"Composizione: Azionario {az}% | Obbligazionario {ob}% | Oro {oro}%")
        if cap0 is not None or capf is not None:
            lines.append(f"Capitale: iniziale {_euro_str(cap0)}  →  finale {_euro_str(capf)}")
        if anni is not None:
            try:
                lines.append(f"Durata: {float(anni):.1f} anni".replace(".", ","))
            except Exception:
                pass
        lines.append(f"Rendimento annualizzato: {_pct_str(cagr)}")
        lines.append(f"Max ribasso nel periodo: {_pct_str(mdd)}")
        try:
            if ytd is not None and math.isfinite(float(ytd)) and float(ytd) > 0:
                lines.append(f"Raddoppio del portafoglio in anni: {float(ytd):.1f}".replace(".", ","))
        except Exception:
            pass

        for ln in lines:
            c.drawString(ml, y, ln)
            y -= 13

        y -= 8

        # Grafico (riempie la pagina in modo “print-friendly”)
        # area disponibile:
        img_x = ml
        img_w = page_w - ml - mr
        img_h = min(360, y - mb)  # lascia spazio al footer
        img_y = y - img_h

        c.setLineWidth(0.6)
        c.rect(img_x, img_y, img_w, img_h)

        # Mantieni aspect ratio dell'immagine ma fallo stare nella box
        iw, ih = img.getSize()
        scale = min(img_w / iw, img_h / ih)
        draw_w = iw * scale
        draw_h = ih * scale
        dx = img_x + (img_w - draw_w) / 2
        dy = img_y + (img_h - draw_h) / 2
        c.drawImage(img, dx, dy, width=draw_w, height=draw_h, mask="auto")

        # Footer / disclaimer
        c.setFont("Helvetica", 8.5)
        c.drawString(ml, mb - 12, "Nota: documento informativo, non consulenza finanziaria.")

        c.showPage()
        c.save()

        buf.seek(0)
        return send_file(
            buf,
            mimetype="application/pdf",
            as_attachment=True,
            download_name="gloob_metodo_pigro.pdf",
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 400


# -----------------------------
# CHAT ASSISTANT (OpenAI)
# -----------------------------
def _extract_text_from_responses_api(payload: dict) -> str:
    # Responses API: payload["output"] è una lista; cerchiamo il primo testo
    out = payload.get("output", [])
    for item in out:
        content = item.get("content", [])
        for c in content:
            if c.get("type") == "output_text" and c.get("text"):
                return str(c["text"]).strip()
            if c.get("type") == "text" and c.get("text"):
                return str(c["text"]).strip()
    # fallback
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

    # contesto utile e “sicuro”
    ctx_lines = []
    if isinstance(context, dict):
        oro = context.get("oro_pct")
        az = context.get("azionario_pct")
        ob = context.get("obbligazionario_pct")
        cap = context.get("capitale_eur")
        if oro is not None and az is not None and ob is not None:
            ctx_lines.append(f"Composizione attuale: Azionario {az}% | Obbligazionario {ob}% | Oro {oro}%.")
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
        ok, remaining = _rate_limit_ok(ip)
        if not ok:
            return jsonify({"error": "Troppe richieste. Riprova più tardi."}), 429

        payload = request.get_json(silent=True) or {}
        q = (payload.get("question") or "").strip()
        if not q:
            return jsonify({"error": "Scrivi una domanda."}), 400
        if len(q) > 800:
            return jsonify({"error": "Domanda troppo lunga (max 800 caratteri)."}), 400

        context = payload.get("context") or {}
        answer = _openai_answer(q, context)

        return jsonify({"answer": answer, "remaining": remaining})

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
