from __future__ import annotations

import csv
import math
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from flask import Flask, jsonify, render_template, request, make_response

# IMPORTANTISSIMO: fissiamo esplicitamente le cartelle
# così Render/Flask non si "confondono" e / torna sempre HTML.
app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0  # evita cache dei file statici (app.js)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

ASSET_FILES: Dict[str, str] = {
    "ls80": "ls80.csv",
    "gold": "gold.csv",
}


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
    raise ValueError(f"Data non riconosciuta: {s!r}")


def _detect_delimiter(sample: str) -> str:
    return ";" if sample.count(";") > sample.count(",") else ","


def _read_price_series(path: Path) -> Tuple[List[date], List[float]]:
    if not path.exists():
        raise FileNotFoundError(f"File non trovato: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        head = f.read(4096)
        delim = _detect_delimiter(head)

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f, delimiter=delim)
        rows = list(reader)

    if not rows or len(rows) < 2:
        raise ValueError(f"CSV vuoto o troppo corto: {path.name}")

    header = [c.strip() for c in rows[0]]
    data_rows = rows[1:]

    date_idx = 0
    price_idx = 1 if len(header) > 1 else 0

    for i, c in enumerate(header):
        cl = c.lower()
        if "date" in cl or "data" in cl:
            date_idx = i
            break

    for i, c in enumerate(header):
        cl = c.lower()
        if any(k in cl for k in ("close", "prezzo", "price", "nav")):
            price_idx = i
            break

    tmp: List[Tuple[date, float]] = []
    for r in data_rows:
        if not r or len(r) <= max(date_idx, price_idx):
            continue
        ds = r[date_idx].strip()
        ps = r[price_idx].strip()
        if not ds or not ps:
            continue
        try:
            d = _parse_date(ds)
            p = _parse_float(ps)
        except Exception:
            continue
        if math.isnan(p):
            continue
        tmp.append((d, p))

    if not tmp:
        raise ValueError(f"Nel CSV {path.name} non trovo righe valide (serve colonna data + prezzo).")

    tmp.sort(key=lambda x: x[0])
    return [d for d, _ in tmp], [p for _, p in tmp]


def _align_by_date(
    a_dates: List[date], a_vals: List[float],
    b_dates: List[date], b_vals: List[float],
) -> Tuple[List[date], List[float], List[float]]:
    ad = {d: v for d, v in zip(a_dates, a_vals)}
    bd = {d: v for d, v in zip(b_dates, b_vals)}
    common = sorted(set(ad.keys()) & set(bd.keys()))
    if len(common) < 2:
        raise ValueError("Serie con poche date comuni tra LS80 e Oro.")
    return common, [ad[d] for d in common], [bd[d] for d in common]


def _max_drawdown(values: List[float]) -> float:
    peak = -float("inf")
    mdd = 0.0
    for v in values:
        if v > peak:
            peak = v
        if peak > 0:
            dd = (v / peak) - 1.0
            if dd < mdd:
                mdd = dd
    return mdd


def _cagr(values: List[float], dates: List[date]) -> float:
    if len(values) < 2:
        return 0.0
    start = values[0]
    end = values[-1]
    days = (dates[-1] - dates[0]).days
    years = days / 365.25 if days > 0 else 0.0
    if years <= 0 or start <= 0:
        return 0.0
    return (end / start) ** (1 / years) - 1


def _years_to_double(cagr: float) -> Optional[float]:
    if cagr <= 0:
        return None
    return math.log(2) / math.log(1 + cagr)


def _backtest_two_assets(
    dates: List[date],
    p_ls: List[float],
    p_gold: List[float],
    w_ls: float,
    w_gold: float,
    capital: float,
) -> Tuple[List[float], List[float]]:
    ls_shares = (capital * w_ls) / p_ls[0]
    gold_shares = (capital * w_gold) / p_gold[0]
    solo_shares = capital / p_ls[0]

    port_vals: List[float] = []
    solo_vals: List[float] = []

    last_year = dates[0].year
    for d, ls_price, g_price in zip(dates, p_ls, p_gold):
        total = ls_shares * ls_price + gold_shares * g_price
        solo_total = solo_shares * ls_price

        # ribilanciamento annuale
        if d.year != last_year:
            ls_shares = (total * w_ls) / ls_price
            gold_shares = (total * w_gold) / g_price
            last_year = d.year
            total = ls_shares * ls_price + gold_shares * g_price

        port_vals.append(total)
        solo_vals.append(solo_total)

    return port_vals, solo_vals


# ✅ HOME: deve tornare HTML sempre
@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/compute")
def api_compute():
    try:
        w_ls80 = float(request.args.get("w_ls80", "80"))
        w_gold = float(request.args.get("w_gold", "20"))

        capital = request.args.get("capital")
        if capital is None:
            capital = request.args.get("initial", "10000")
        capital = float(capital)

        if capital <= 0:
            return jsonify({"error": "Capitale deve essere > 0"}), 400
        if w_ls80 < 0 or w_gold < 0:
            return jsonify({"error": "I pesi devono essere >= 0"}), 400

        # percentuali -> quote
        s = w_ls80 + w_gold
        if s <= 0:
            return jsonify({"error": "Somma pesi deve essere > 0"}), 400

        w_ls = w_ls80 / s
        w_g = w_gold / s

        d_ls, p_ls = _read_price_series(DATA_DIR / ASSET_FILES["ls80"])
        d_g, p_g = _read_price_series(DATA_DIR / ASSET_FILES["gold"])
        dates, p_ls_al, p_g_al = _align_by_date(d_ls, p_ls, d_g, p_g)

        port_vals, solo_vals = _backtest_two_assets(dates, p_ls_al, p_g_al, w_ls, w_g, capital)

        cagr_port = _cagr(port_vals, dates)
        cagr_solo = _cagr(solo_vals, dates)
        mdd_port = _max_drawdown(port_vals)
        mdd_solo = _max_drawdown(solo_vals)
        yd = _years_to_double(cagr_port)

        payload = {
            "dates": [d.isoformat() for d in dates],
            "portfolio": port_vals,
            "solo_ls80": solo_vals,
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


if __name__ == "__main__":
    # su Render non serve, ma in locale sì
    app.run(host="127.0.0.1", port=5000, debug=True)
