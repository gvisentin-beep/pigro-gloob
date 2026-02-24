from __future__ import annotations

import os
import json
import math
import traceback
from datetime import datetime, date, timezone
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, render_template

# OpenAI SDK 1.x
try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # gestiamo sotto


app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))

# File dati
LS80_FILE = DATA_DIR / "ls80.csv"
GOLD_FILE = DATA_DIR / "gold.csv"

# Ticker (Yahoo)
LS80_TICKER = os.getenv("LS80_TICKER", "VNGA80.MI")
GOLD_TICKER = os.getenv("GOLD_TICKER", "SGLD")

# Token update manuale (resta disponibile, ma NON è più necessario per l'auto-update)
UPDATE_TOKEN = os.getenv("UPDATE_TOKEN", "").strip()

# ✅ Auto update: 1 = attivo, 0 = spento
AUTO_UPDATE = os.getenv("AUTO_UPDATE", "1").strip()
AUTO_UPDATE_MINUTES = int(os.getenv("AUTO_UPDATE_MINUTES", "60"))
_LAST_AUTO_UPDATE_UTC: Optional[datetime] = None


# ----------------------------
# Helpers
# ----------------------------
def _now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _json_error(message: str, code: int = 400, **extra: Any):
    payload = {"ok": False, "error": message, "time_utc": _now_iso()}
    payload.update(extra)
    return jsonify(payload), code


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        s = str(x).strip().replace(",", ".")
        if s == "":
            return None
        return float(s)
    except Exception:
        return None


def _detect_sep(file_path: Path) -> str:
    # euristica semplice: se la prima riga ha più ';' che ',' -> ';'
    sample = file_path.read_text(encoding="utf-8", errors="ignore").splitlines()[:3]
    if not sample:
        return ";"
    line = sample[0]
    return ";" if line.count(";") >= line.count(",") else ","


def _read_price_csv(file_path: Path) -> pd.DataFrame:
    """
    Ritorna DataFrame con colonne:
      - date (datetime naive)
      - close (float)

    Gestisce:
      - separatore ; o ,
      - date gg/mm/aaaa (dayfirst=True) oppure yyyy-mm-dd
      - eventuali spazi/righe sporche
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File non trovato: {file_path}")

    sep = _detect_sep(file_path)
    df = pd.read_csv(file_path, sep=sep)
    df.columns = [c.strip() for c in df.columns]

    cols = {c.lower(): c for c in df.columns}
    if "date" not in cols or "close" not in cols:
        raise ValueError(
            f"CSV {file_path.name}: colonne attese 'Date' e 'Close'. Trovate: {list(df.columns)}"
        )

    dcol = cols["date"]
    ccol = cols["close"]

    dates = pd.to_datetime(df[dcol], errors="coerce", dayfirst=True)
    close = pd.to_numeric(df[ccol], errors="coerce")

    out = pd.DataFrame({"date": dates, "close": close}).dropna()
    out = out.sort_values("date").drop_duplicates(subset=["date"], keep="last")

    if len(out) < 10:
        raise ValueError(
            f"CSV {file_path.name}: troppo poche righe valide dopo parsing ({len(out)})."
        )

    # forza naive (senza timezone)
    try:
        out["date"] = out["date"].dt.tz_localize(None)
    except Exception:
        pass

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
    if cagr is None or cagr <= 0:
        return None
    return math.log(2.0) / math.log(1.0 + cagr)


def _first_trading_day_each_year(dates: pd.Series) -> pd.Series:
    years = dates.dt.year
    first_idx = dates.groupby(years).head(1).index
    flags = dates.index.isin(first_idx)
    return pd.Series(flags, index=dates.index)


def _annual_rebalance_portfolio(
    ls80_close: pd.Series,
    gold_close: pd.Series,
    dates: pd.Series,
    w_gold: float,
    capital: float,
) -> pd.Series:
    """
    Portafoglio 2 strumenti ribilanciato 1 volta l'anno
    (sul primo giorno di borsa dell'anno).
    """
    w_gold = float(np.clip(w_gold, 0.0, 1.0))
    w_ls80 = 1.0 - w_gold

    v0 = float(capital)
    ls80_shares = (v0 * w_ls80) / float(ls80_close.iloc[0])
    gold_shares = (v0 * w_gold) / float(gold_close.iloc[0])

    rebalance_flags = _first_trading_day_each_year(dates)

    values = []
    for i in range(len(dates)):
        v = ls80_shares * float(ls80_close.iloc[i]) + gold_shares * float(gold_close.iloc[i])
        values.append(float(v))

        if rebalance_flags.iloc[i]:
            v = float(values[-1])
            ls80_shares = (v * w_ls80) / float(ls80_close.iloc[i])
            gold_shares = (v * w_gold) / float(gold_close.iloc[i])

    return pd.Series(values, index=dates.index, name="portfolio")


# ----------------------------
# ✅ Auto-update dati (automatico su /api/compute)
# ----------------------------
def _maybe_auto_update_data() -> Dict[str, Any]:
    """
    Aggiorna automaticamente i CSV (ls80/gold) usando yfinance **solo se**:
      - AUTO_UPDATE == "1"
      - e non abbiamo già tentato un update negli ultimi AUTO_UPDATE_MINUTES
    Non solleva eccezioni: in caso di problemi restituisce info diagnostiche.
    """
    global _LAST_AUTO_UPDATE_UTC

    info: Dict[str, Any] = {"attempted": False, "updated_any": False, "details": {}, "time_utc": _now_iso()}

    if AUTO_UPDATE != "1":
        info["skipped_reason"] = "AUTO_UPDATE_disabled"
        return info

    now = datetime.now(timezone.utc)
    if _LAST_AUTO_UPDATE_UTC is not None:
        delta_min = (now - _LAST_AUTO_UPDATE_UTC).total_seconds() / 60.0
        if delta_min < float(AUTO_UPDATE_MINUTES):
            info["skipped_reason"] = f"throttled_{AUTO_UPDATE_MINUTES}m"
            info["last_attempt_utc"] = _LAST_AUTO_UPDATE_UTC.isoformat().replace("+00:00", "Z")
            return info

    _LAST_AUTO_UPDATE_UTC = now
    info["attempted"] = True

    try:
        import yfinance as yf  # type: ignore
    except Exception as e:
        info["error"] = f"yfinance_missing: {type(e).__name__}: {e}"
        return info

    def update_one(ticker: str, file_path: Path) -> Dict[str, Any]:
        res: Dict[str, Any] = {"ticker": ticker, "file": str(file_path), "updated": False}
        try:
            df = _read_price_csv(file_path)
            last_date = df["date"].iloc[-1].date()

            # prende gli ultimi giorni: se c'è una nuova chiusura, la vede qui
            hist = yf.download(ticker, period="10d", interval="1d", auto_adjust=False, progress=False)
            if hist is None or hist.empty:
                res["reason"] = "no_data_from_yahoo"
                return res

            hist = hist.reset_index()
            dcol = "Date" if "Date" in hist.columns else ("Datetime" if "Datetime" in hist.columns else None)
            if dcol is None or "Close" not in hist.columns:
                res["reason"] = "unexpected_columns"
                res["columns"] = list(hist.columns)
                return res

            hist["date"] = pd.to_datetime(hist[dcol], errors="coerce").dt.tz_localize(None)
            hist["close"] = pd.to_numeric(hist["Close"], errors="coerce")
            hist = hist[["date", "close"]].dropna().sort_values("date")

            new_rows = hist[hist["date"].dt.date > last_date]
            if new_rows.empty:
                res["reason"] = "already_up_to_date"
                return res

            out = pd.concat([df, new_rows], ignore_index=True)
            out = out.sort_values("date").drop_duplicates(subset=["date"], keep="last")

            DATA_DIR.mkdir(parents=True, exist_ok=True)
            save = pd.DataFrame(
                {
                    "Date": out["date"].dt.strftime("%d/%m/%Y"),
                    "Close": out["close"].map(lambda x: f"{float(x):.2f}"),
                }
            )
            save.to_csv(file_path, sep=";", index=False)

            res["updated"] = True
            res["added_rows"] = int(len(new_rows))
            res["last_date"] = str(out["date"].iloc[-1].date())
            res["last_value"] = float(out["close"].iloc[-1])
            return res
        except Exception as e:
            res["error"] = f"{type(e).__name__}: {e}"
            return res

    res_ls = update_one(LS80_TICKER, LS80_FILE)
    res_gd = update_one(GOLD_TICKER, GOLD_FILE)
    info["details"] = {"ls80": res_ls, "gold": res_gd}
    info["updated_any"] = bool(res_ls.get("updated")) or bool(res_gd.get("updated"))
    return info


# ----------------------------
# Routes base
# ----------------------------
@app.get("/")
def home():
    return render_template("index.html")


@app.get("/health")
def health():
    return jsonify({"ok": True, "time_utc": _now_iso()})


def _routes_list() -> list[dict[str, str]]:
    routes = []
    for r in app.url_map.iter_rules():
        methods = ",".join(sorted([m for m in r.methods if m not in ("HEAD", "OPTIONS")]))
        routes.append({"rule": str(r), "methods": methods, "endpoint": r.endpoint})
    routes.sort(key=lambda x: (x["rule"], x["methods"]))
    return routes


@app.get("/api/diag/routes")
def api_diag_routes():
    return jsonify({"ok": True, "routes": _routes_list(), "time_utc": _now_iso()})


@app.get("/api/diag")
def api_diag():
    # diagnostica rapida: presenza file e ultime date
    info: Dict[str, Any] = {"ok": True, "time_utc": _now_iso(), "data_dir": str(DATA_DIR)}
    try:
        df_ls = _read_price_csv(LS80_FILE)
        df_gd = _read_price_csv(GOLD_FILE)
        info["ls80_last_date"] = str(df_ls["date"].iloc[-1].date())
        info["gold_last_date"] = str(df_gd["date"].iloc[-1].date())
        info["routes_count"] = len(_routes_list())
    except Exception as e:
        info["ok"] = False
        info["error"] = f"{type(e).__name__}: {e}"
    return jsonify(info)


@app.get("/api/diag/compute_smoke")
def api_diag_compute_smoke():
    # esegue compute con parametri default e restituisce solo metriche
    try:
        w_gold = 0.20
        capital = 10000.0

        df_ls = _read_price_csv(LS80_FILE)
        df_gd = _read_price_csv(GOLD_FILE)
        df = pd.merge(df_ls, df_gd, on="date", how="inner", suffixes=("_ls80", "_gold")).sort_values("date")
        dates = df["date"]
        port = _annual_rebalance_portfolio(df["close_ls80"], df["close_gold"], dates, w_gold=w_gold, capital=capital)

        cagr = _compute_cagr(port, dates)
        dd = _compute_drawdown(port)
        dy = _doubling_years(cagr)

        return jsonify(
            {
                "ok": True,
                "time_utc": _now_iso(),
                "len": int(len(df)),
                "start": str(dates.iloc[0].date()),
                "end": str(dates.iloc[-1].date()),
                "cagr": cagr,
                "max_drawdown": dd,
                "doubling_years": dy,
            }
        )
    except Exception as e:
        return _json_error(f"{type(e).__name__}: {e}", 500, traceback=traceback.format_exc())


# ----------------------------
# API: compute
# ----------------------------
@app.get("/api/compute")
def api_compute():
    try:
        # ✅ prima di calcolare: prova ad aggiornare i CSV all’ultima chiusura (throttled)
        auto_update_info = _maybe_auto_update_data()

        w_gold = _safe_float(request.args.get("w_gold"))
        if w_gold is None:
            w_ls80 = _safe_float(request.args.get("w_ls80"))
            if w_ls80 is None:
                w_gold = 0.20
            else:
                w_gold = 1.0 - float(np.clip(w_ls80, 0.0, 1.0))
        else:
            w_gold = float(np.clip(w_gold, 0.0, 1.0))

        capital = _safe_float(request.args.get("capital")) or 10000.0
        capital = max(1.0, float(capital))

        df_ls = _read_price_csv(LS80_FILE)
        df_gd = _read_price_csv(GOLD_FILE)

        df = pd.merge(df_ls, df_gd, on="date", how="inner", suffixes=("_ls80", "_gold")).sort_values("date")
        if df.empty:
            return _json_error("Dati insufficienti dopo merge (nessuna data in comune).", 500)

        dates = df["date"]
        port = _annual_rebalance_portfolio(df["close_ls80"], df["close_gold"], dates, w_gold=w_gold, capital=capital)

        cagr = _compute_cagr(port, dates)
        dd = _compute_drawdown(port)
        dy = _doubling_years(cagr)

        # composizione “azionaria/obbligazionaria” fissa per LS80 (80/20) * (1 - oro)
        w_ls = 1.0 - w_gold
        az = round(w_ls * 0.80, 4)
        ob = round(w_ls * 0.20, 4)

        metrics: Dict[str, Any] = {
            "cagr": cagr,
            "max_drawdown": dd,
            "doubling_years": dy,
            "final_value": float(port.iloc[-1]),
            "years": (dates.iloc[-1] - dates.iloc[0]).days / 365.25,
        }

        payload: Dict[str, Any] = {
            "ok": True,
            "dates": [d.strftime("%Y-%m-%d") for d in dates],
            "portfolio": [float(x) for x in port],
            "metrics": metrics,
            "composition": {"azionario": az, "obbligazionario": ob, "oro": w_gold},
            # ✅ info diagnostica (il frontend la ignora, ma ci serve se qualcosa non aggiorna)
            "auto_update": auto_update_info,
        }

        return jsonify(payload)

    except Exception as e:
        return _json_error(f"{type(e).__name__}: {e}", 500, traceback=traceback.format_exc())


# ----------------------------
# API: ask (assistente)
# ----------------------------
@app.post("/api/ask")
def api_ask():
    try:
        data = request.get_json(silent=True) or {}
        question = (data.get("question") or "").strip()
        if not question:
            return _json_error("Domanda vuota.", 400)

        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key or OpenAI is None:
            return _json_error("Assistente non configurato (OPENAI_API_KEY mancante).", 500)

        client = OpenAI(api_key=api_key)

        system = (
            "Sei un assistente informativo. Non fornire consulenza finanziaria personalizzata. "
            "Rispondi in italiano, in modo chiaro e concreto."
        )

        resp = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": question},
            ],
            temperature=0.4,
        )

        answer = resp.choices[0].message.content.strip() if resp.choices else ""
        return jsonify({"ok": True, "answer": answer, "time_utc": _now_iso()})

    except Exception as e:
        return _json_error(f"{type(e).__name__}: {e}", 500, traceback=traceback.format_exc())


# ----------------------------
# API: update data (manuale - opzionale)
# ----------------------------
@app.get("/api/update_data")
def api_update_data():
    token = (request.args.get("token") or "").strip()
    if not UPDATE_TOKEN or token != UPDATE_TOKEN:
        return _json_error("Token non valido.", 401)

    try:
        import yfinance as yf
    except Exception as e:
        return _json_error(f"yfinance non disponibile (requirements). {type(e).__name__}: {e}", 500)

    def update_one(ticker: str, file_path: Path) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "asset": file_path.stem,
            "ticker_used": ticker,
            "file": str(file_path),
            "updated": False,
        }

        df = _read_price_csv(file_path)
        last_date = df["date"].iloc[-1].date()

        hist = yf.download(ticker, period="10d", interval="1d", auto_adjust=False, progress=False)
        if hist is None or hist.empty:
            info["reason"] = "no_data_from_yahoo"
            return info

        hist = hist.reset_index()
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

        out = pd.concat([df, new_rows], ignore_index=True)
        out = out.sort_values("date").drop_duplicates(subset=["date"], keep="last")

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
        return _json_error(f"{type(e).__name__}: {e}", 500, traceback=traceback.format_exc())


# ----------------------------
# Main (local)
# ----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "10000")), debug=True)
