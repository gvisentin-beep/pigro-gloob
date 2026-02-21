from __future__ import annotations

import csv
import json
import math
import os
import time
import urllib.parse
import urllib.request
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

from flask import Flask, jsonify, make_response, render_template, request

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0  # anti-cache static

BASE_DIR = Path(__file__).resolve().parent

# Se vuoi usare Persistent Disk su Render, imposta DATA_DIR=/var/data (o simile)
DATA_DIR = Path(os.getenv("DATA_DIR", str(BASE_DIR / "data"))).resolve()
DATA_DIR.mkdir(parents=True, exist_ok=True)

ASSET_FILES: Dict[str, str] = {
    "ls80": "ls80.csv",
    "gold": "gold.csv",
}

# =========================
# CONFIG UPDATE QUOTAZIONI
# =========================
# Puoi impostarle su Render -> Environment, ma ho messo default sensati.
# (VNGA80 su Yahoo è VNGA80.MI) :contentReference[oaicite:2]{index=2}
LS80_TICKER = os.getenv("LS80_TICKER", "VNGA80.MI").strip()

# Per SGLD possiamo provare automaticamente varie piazze.
# (Esempi Yahoo: SGLD.L, SGLD.MI) :contentReference[oaicite:3]{index=3}
GOLD_TICKER = os.getenv("GOLD_TICKER", "SGLD").strip()

UPDATE_MIN_INTERVAL_HOURS = float(os.getenv("UPDATE_MIN_INTERVAL_HOURS", "6"))  # max 1 tentativo/6h
_update_state = {"last_try_ts": 0.0}

UPDATE_TOKEN = os.getenv("UPDATE_TOKEN", "").strip()  # per cron job /api/update_data?token=...


def _fmt_ddmmyyyy(d: date) -> str:
    return d.strftime("%d/%m/%Y")


def _parse_float(s: str) -> float:
    s = (s or "").strip()
    if s == "":
        return float("nan")
    # supporto "1.234,56" e "1234.56"
    if "," in s and "." in s:
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    elif "," in s and "." not in s:
        s = s.replace(",", ".")
    return float(s)


def _parse_date(s: str) -> date:
    s = (s or "").strip()
    for fmt in ("%Y-%m-%d", "%d/%m/%Y"):
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
        # i tuoi file sono in stile: Date;Close e righe "gg/mm/aaaa;123.45"
        reader = csv.reader(f, delimiter=";")
        header = next(reader, None)
        for row in reader:
            if not row or len(row) < 2:
                continue
            d = _parse_date(row[0])
            v = _parse_float(row[1])
            if math.isfinite(v):
                dates.append(d)
                values.append(float(v))

    if not dates:
        raise ValueError(f"Nessun dato valido in {path}")

    return dates, values


def _write_csv_same_style(path: Path, rows_desc: List[Tuple[date, float]]) -> None:
    # Manteniamo lo stesso stile originale: Date;Close e ordine decrescente (più recente in alto)
    with path.open("w", encoding="utf-8", newline="") as f:
        f.write("Date;Close\n")
        for d, v in rows_desc:
            f.write(f"{_fmt_ddmmyyyy(d)};{v:.2f}\n")


def _yahoo_chart_daily_closes(ticker: str, range_days: int = 30) -> List[Tuple[date, float]]:
    """
    Scarica close giornalieri da Yahoo Finance (endpoint pubblico chart).
    Ritorna [(date_utc, close), ...]
    """
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{urllib.parse.quote(ticker)}"
        f"?range={range_days}d&interval=1d&includePrePost=false&events=div%7Csplit"
    )
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"},
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    chart = (data or {}).get("chart", {})
    if chart.get("error"):
        raise RuntimeError(str(chart["error"]))

    result = (chart.get("result") or [None])[0]
    if not result:
        return []

    ts = result.get("timestamp") or []
    quote = ((result.get("indicators") or {}).get("quote") or [{}])[0]
    closes = quote.get("close") or []

    out: List[Tuple[date, float]] = []
    for t, c in zip(ts, closes):
        if c is None:
            continue
        d = datetime.fromtimestamp(int(t), tz=timezone.utc).date()
        out.append((d, float(c)))
    return out


def _resolve_tickers(user_ticker: str, asset_key: str) -> List[str]:
    """
    Se l’utente passa "SGLD" senza suffisso, proviamo alcune piazze.
    Per VNGA80, se arriva senza .MI, proviamo .MI.
    """
    t = (user_ticker or "").strip()
    if not t:
        return []

    if "." in t:
        return [t]

    if asset_key == "ls80":
        # tipicamente Borsa Italiana
        return [f"{t}.MI", t]
    if asset_key == "gold":
        # oro: prova Milano e Londra
        return [f"{t}.MI", f"{t}.L", t]

    return [t]


def _maybe_update_one(asset_key: str, user_ticker: str) -> Dict[str, object]:
    path = DATA_DIR / ASSET_FILES[asset_key]
    d_old, v_old = _detect_and_read_csv(path)
    old_map = {d: v for d, v in zip(d_old, v_old)}
    latest_old = max(old_map.keys())

    tickers_to_try = _resolve_tickers(user_ticker, asset_key)
    last_err = None
    downloaded: List[Tuple[date, float]] = []

    for tk in tickers_to_try:
        try:
            downloaded = _yahoo_chart_daily_closes(tk, range_days=30)
            if downloaded:
                used = tk
                break
        except Exception as e:
            last_err = e
            downloaded = []
            continue
    else:
        raise RuntimeError(f"Impossibile scaricare dati per {asset_key}. Ultimo errore: {last_err}")

    # aggiungi solo nuove date
    new_items = [(d, v) for d, v in downloaded if d > latest_old]
    if not new_items:
        return {"asset": asset_key, "updated": False, "reason": "already_up_to_date", "ticker_used": used}

    for d, v in new_items:
        old_map[d] = v

    rows_desc = sorted(old_map.items(), key=lambda x: x[0], reverse=True)
    _write_csv_same_style(path, rows_desc)

    return {
        "asset": asset_key,
        "updated": True,
        "added_rows": len(new_items),
        "latest_date": rows_desc[0][0].isoformat(),
        "ticker_used": used,
    }


def update_assets_if_due(force: bool = False) -> Dict[str, object]:
    now = time.time()
    min_interval = UPDATE_MIN_INTERVAL_HOURS * 3600.0
    if not force and (now - float(_update_state["last_try_ts"])) < min_interval:
        return {"skipped": True, "reason": "too_soon"}

    _update_state["last_try_ts"] = now

    res: Dict[str, object] = {"skipped": False, "ls80": None, "gold": None, "warnings": []}

    try:
        res["ls80"] = _maybe_update_one("ls80", LS80_TICKER)
    except Exception as e:
        res["warnings"].append(f"Update ls80 fallito: {e}")

    try:
        res["gold"] = _maybe_update_one("gold", GOLD_TICKER)
    except Exception as e:
        res["warnings"].append(f"Update gold fallito: {e}")

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


def _build_portfolio(dates: List[date], ls_vals: List[float], g_vals: List[float], w_ls80: float, w_gold: float, capital: float):
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
    solo_shares = capital / ls_vals[0]

    port: List[float] = []
    solo: List[float] = []

    prev_year = dates[0].year
    for d, ls, g in zip(dates, ls_vals, g_vals):
        if d.year != prev_year:
            total = ls_shares * ls + g_shares * g
            ls_shares = (total * w_ls) / ls
            g_shares = (total * w_g) / g
            prev_year = d.year

        port.append(ls_shares * ls + g_shares * g)
        solo.append(solo_shares * ls)

    return port, solo


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

        port_vals, solo_vals = _build_portfolio(dates, ls, g, w_ls80, w_gold, capital)

        years = _years_between(dates[0], dates[-1])
        cagr_port = _compute_cagr(port_vals[0], port_vals[-1], years)
        mdd_port = _max_drawdown(port_vals)

        yd = float("nan")
        if math.isfinite(cagr_port) and cagr_port > 0:
            yd = math.log(2) / math.log(1 + cagr_port)

        payload = {
            "dates": [d.isoformat() for d in dates],
            "portfolio": port_vals,
            "solo_ls80": solo_vals,
            "metrics": {
                "cagr_portfolio": cagr_port,
                "max_dd_portfolio": mdd_port,
                "years_to_double": yd,
                "start": dates[0].isoformat(),
                "end": dates[-1].isoformat(),
                "final_portfolio": port_vals[-1],
            },
        }

        r = make_response(jsonify(payload))
        r.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        r.headers["Pragma"] = "no-cache"
        return r

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
