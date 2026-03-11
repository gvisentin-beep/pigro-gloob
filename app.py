from __future__ import annotations

import math
import os
from io import BytesIO
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import requests
from flask import Flask, jsonify, render_template, request, send_file

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

LS80_FILE = DATA_DIR / "ls80.csv"
GOLD_FILE = DATA_DIR / "gold.csv"
WORLD_FILE = DATA_DIR / "world.csv"

# CSV locali: il sito legge SEMPRE questi file
LS80_TICKER = os.getenv("LS80_TICKER", "IWDA").strip()
GOLD_TICKER = os.getenv("GOLD_TICKER", "GLD").strip()
WORLD_TICKER = os.getenv("WORLD_TICKER", "URTH").strip()

# Update lato server
UPDATE_TOKEN = os.getenv("UPDATE_TOKEN", "").strip()
TWELVE_DATA_API_KEY = os.getenv("TWELVE_DATA_API_KEY", "").strip()
TWELVE_DATA_BASE_URL = "https://api.twelvedata.com/time_series"

# Assistente
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini").strip()

# Limite semplice assistente
ASK_DAILY_LIMIT = int(os.getenv("ASK_DAILY_LIMIT", "10"))


# ----------------------------------------------------------
# Helpers generali
# ----------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _json_error(msg: str, status: int = 400, **extra: Any):
    payload = {"ok": False, "error": msg}
    payload.update(extra)
    return jsonify(payload), status


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(str(x).replace(",", "."))
    except Exception:
        return None


def _require_token() -> Optional[Tuple[Any, int]]:
    if not UPDATE_TOKEN:
        return None
    token = (request.args.get("token") or "").strip()
    if token != UPDATE_TOKEN:
        return _json_error("Token non valido.", 401)
    return None


# ----------------------------------------------------------
# CSV robusto
# ----------------------------------------------------------

def read_price_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(str(path))

    # Tentativo 1: con intestazione
    try:
        df = pd.read_csv(path, sep=";", dtype=str, encoding="utf-8-sig")
        df.columns = [str(c).strip().lower() for c in df.columns]

        if "date" in df.columns and "close" in df.columns:
            df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
            df["close"] = (
                df["close"]
                .astype(str)
                .str.strip()
                .str.replace(",", ".", regex=False)
            )
            df["close"] = pd.to_numeric(df["close"], errors="coerce")
            df = df.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
            if not df.empty:
                return df[["date", "close"]].copy()
    except Exception:
        pass

    # Tentativo 2: senza intestazione
    df = pd.read_csv(
        path,
        sep=";",
        dtype=str,
        header=None,
        names=["date", "close"],
        encoding="utf-8-sig",
    )

    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    df["close"] = (
        df["close"]
        .astype(str)
        .str.strip()
        .str.replace(",", ".", regex=False)
    )
    df["close"] = pd.to_numeric(df["close"], errors="coerce")

    df = df.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)

    if df.empty:
        raise ValueError(f"CSV non valido: {path}")

    return df[["date", "close"]].copy()


def write_price_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    out = df.copy().sort_values("date").reset_index(drop=True)
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%d/%m/%Y")
    out.to_csv(path, sep=";", index=False)


# ----------------------------------------------------------
# Twelve Data update
# ----------------------------------------------------------

def fetch_twelve_data(symbol: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    if not TWELVE_DATA_API_KEY:
        return None, "TWELVE_DATA_API_KEY mancante"

    params = {
        "symbol": symbol,
        "interval": "1day",
        "outputsize": 5000,
        "format": "JSON",
        "order": "ASC",
        "apikey": TWELVE_DATA_API_KEY,
    }

    try:
        resp = requests.get(TWELVE_DATA_BASE_URL, params=params, timeout=40)
        resp.raise_for_status()
        data = resp.json()

        if not isinstance(data, dict):
            return None, "Risposta Twelve Data non valida"

        if data.get("status") == "error":
            code = data.get("code", "")
            message = data.get("message", "errore sconosciuto")
            return None, f"{code} {message}".strip()

        values = data.get("values")
        if not values:
            return None, "Nessun valore restituito"

        df = pd.DataFrame(values)

        if "datetime" not in df.columns or "close" not in df.columns:
            return None, f"Colonne inattese: {list(df.columns)}"

        df["date"] = pd.to_datetime(df["datetime"], errors="coerce")
        df["close"] = pd.to_numeric(df["close"], errors="coerce")

        df = df.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
        if df.empty:
            return None, "Serie vuota dopo pulizia"

        return df[["date", "close"]].copy(), None

    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def update_one_asset(path: Path, symbol: str) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "asset": path.stem,
        "symbol": symbol,
        "file": str(path),
        "updated": False,
        "usable": False,
    }

    old_df = None
    try:
        old_df = read_price_csv(path)
    except Exception as e:
        info["read_error"] = f"{type(e).__name__}: {e}"

    new_df, err = fetch_twelve_data(symbol)

    if new_df is None or new_df.empty:
        info["reason"] = err or "no_data_from_twelve_data"

        if old_df is not None and not old_df.empty:
            info["usable"] = True
            info["fallback_rows"] = int(len(old_df))
            info["fallback_first_date"] = str(old_df["date"].iloc[0].date())
            info["fallback_last_date"] = str(old_df["date"].iloc[-1].date())
            info["fallback_last_value"] = float(old_df["close"].iloc[-1])

        return info

    if old_df is not None and not old_df.empty:
        merged = pd.concat([old_df, new_df], ignore_index=True)
    else:
        merged = new_df

    merged = (
        merged.drop_duplicates(subset=["date"], keep="last")
        .sort_values("date")
        .reset_index(drop=True)
    )

    write_price_csv(path, merged)

    info["updated"] = True
    info["usable"] = True
    info["rows"] = int(len(merged))
    info["first_date"] = str(merged["date"].iloc[0].date())
    info["last_date"] = str(merged["date"].iloc[-1].date())
    info["first_value"] = float(merged["close"].iloc[0])
    info["last_value"] = float(merged["close"].iloc[-1])
    return info


# ----------------------------------------------------------
# Metriche
# ----------------------------------------------------------

def compute_cagr(series: pd.Series, dates: pd.Series) -> float:
    if len(series) < 2:
        return 0.0

    years = (dates.iloc[-1] - dates.iloc[0]).days / 365.25
    if years <= 0:
        return 0.0

    start = float(series.iloc[0])
    end = float(series.iloc[-1])
    if start <= 0:
        return 0.0

    return (end / start) ** (1.0 / years) - 1.0


def compute_max_dd(series: pd.Series) -> float:
    peak = series.cummax()
    dd = series / peak - 1.0
    return float(dd.min())


def compute_drawdown_series_pct(series: pd.Series) -> pd.Series:
    peak = series.cummax()
    dd = series / peak - 1.0
    return dd * 100.0


def doubling_years(cagr: float) -> Optional[float]:
    if cagr <= 0:
        return None
    return math.log(2.0) / math.log(1.0 + cagr)


def worst_drawdowns(series: pd.Series, dates: pd.Series, n: int = 3) -> list[dict]:
    peak = series.cummax()
    dd = series / peak - 1.0

    events = []
    in_dd = False
    start_idx = None

    for i in range(1, len(series)):
        if not in_dd and dd.iloc[i] < 0:
            start_idx = i - 1
            in_dd = True

        if in_dd and dd.iloc[i] == 0:
            sub = dd.iloc[start_idx:i]
            if len(sub) > 0:
                bottom = sub.idxmin()
                events.append(
                    {
                        "start": dates.iloc[start_idx].strftime("%Y-%m-%d"),
                        "bottom": dates.iloc[bottom].strftime("%Y-%m-%d"),
                        "end": dates.iloc[i].strftime("%Y-%m-%d"),
                        "depth_pct": float(sub.min() * 100.0),
                    }
                )
            in_dd = False
            start_idx = None

    if in_dd and start_idx is not None:
        sub = dd.iloc[start_idx:]
        if len(sub) > 0:
            bottom = sub.idxmin()
            events.append(
                {
                    "start": dates.iloc[start_idx].strftime("%Y-%m-%d"),
                    "bottom": dates.iloc[bottom].strftime("%Y-%m-%d"),
                    "end": "in corso",
                    "depth_pct": float(sub.min() * 100.0),
                }
            )

    events.sort(key=lambda x: x["depth_pct"])
    return events[:n]


def compute_portfolio(ls80: pd.Series, gold: pd.Series, w_gold: float, capital: float) -> pd.Series:
    w_gold = max(0.0, min(0.5, float(w_gold)))
    w_ls80 = 1.0 - w_gold

    shares_ls80 = (capital * w_ls80) / float(ls80.iloc[0])
    shares_gold = (capital * w_gold) / float(gold.iloc[0])

    return shares_ls80 * ls80 + shares_gold * gold


# ----------------------------------------------------------
# Assistente minimale
# ----------------------------------------------------------

def _openai_client():
    if not OPENAI_API_KEY:
        return None
    try:
        from openai import OpenAI
        return OpenAI(api_key=OPENAI_API_KEY)
    except Exception:
        return None


# ----------------------------------------------------------
# Pages
# ----------------------------------------------------------

@app.get("/")
def home():
    return render_template("index.html")


@app.get("/health")
def health():
    return jsonify({"ok": True, "time_utc": _now_iso()})


# ----------------------------------------------------------
# PDF
# ----------------------------------------------------------

@app.get("/faxsimile_execution_only.pdf")
def faxsimile_execution_only():
    static_pdf = BASE_DIR / "static" / "faxsimile_execution_only.pdf"
    if static_pdf.exists():
        return send_file(static_pdf, mimetype="application/pdf")

    return _json_error("PDF non trovato.", 404)


@app.get("/api/pdf")
def api_pdf():
    try:
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import A4

        title = request.args.get("title", "Gloob - Metodo Pigro")
        cagr = request.args.get("cagr", "")
        maxdd = request.args.get("maxdd", "")
        finalv = request.args.get("final", "")
        years = request.args.get("years", "")

        buf = BytesIO()
        c = canvas.Canvas(buf, pagesize=A4)
        w, h = A4

        c.setFont("Helvetica-Bold", 16)
        c.drawString(50, h - 60, title)

        c.setFont("Helvetica", 12)
        y = h - 100
        c.drawString(50, y, f"Rendimento annualizzato: {cagr}")
        y -= 20
        c.drawString(50, y, f"Max Ribasso: {maxdd}")
        y -= 20
        c.drawString(50, y, f"Finale: {finalv} (in anni {years})")

        c.setFont("Helvetica", 10)
        c.drawString(50, 40, "Documento informativo - non consulenza finanziaria.")
        c.showPage()
        c.save()

        buf.seek(0)
        return send_file(
            buf,
            mimetype="application/pdf",
            as_attachment=True,
            download_name="gloob_pigro.pdf",
        )

    except Exception as e:
        return _json_error(f"Errore PDF: {type(e).__name__}: {e}", 500)


# ----------------------------------------------------------
# Diagnostics
# ----------------------------------------------------------

@app.get("/api/diag")
def api_diag():
    def info(path: Path) -> Dict[str, Any]:
        if not path.exists():
            return {"exists": False, "file": str(path)}
        try:
            df = read_price_csv(path)
            return {
                "exists": True,
                "file": str(path),
                "rows": int(len(df)),
                "first_date": str(df["date"].iloc[0].date()),
                "last_date": str(df["date"].iloc[-1].date()),
                "first_value": float(df["close"].iloc[0]),
                "last_value": float(df["close"].iloc[-1]),
            }
        except Exception as e:
            return {
                "exists": True,
                "file": str(path),
                "error": f"{type(e).__name__}: {e}",
            }

    payload: Dict[str, Any] = {
        "ok": True,
        "time_utc": _now_iso(),
        "files": {
            "ls80": info(LS80_FILE),
            "gold": info(GOLD_FILE),
            "world": info(WORLD_FILE),
        },
        "env": {
            "LS80_TICKER": LS80_TICKER,
            "GOLD_TICKER": GOLD_TICKER,
            "WORLD_TICKER": WORLD_TICKER,
            "TWELVE_DATA_API_KEY_present": bool(TWELVE_DATA_API_KEY),
            "UPDATE_TOKEN_present": bool(UPDATE_TOKEN),
            "OPENAI_API_KEY_present": bool(OPENAI_API_KEY),
            "OPENAI_MODEL": OPENAI_MODEL,
        },
    }

    try:
        ls80 = read_price_csv(LS80_FILE)
        gold = read_price_csv(GOLD_FILE)
        world = read_price_csv(WORLD_FILE)

        merged = ls80.merge(gold, on="date", how="inner", suffixes=("_ls80", "_gold"))
        merged = merged.merge(world, on="date", how="inner")

        payload["merge"] = {
            "rows_inner": int(len(merged)),
            "first_date": str(merged["date"].iloc[0].date()) if len(merged) else None,
            "last_date": str(merged["date"].iloc[-1].date()) if len(merged) else None,
        }
    except Exception as e:
        payload["merge_error"] = f"{type(e).__name__}: {e}"

    return jsonify(payload)


# ----------------------------------------------------------
# Compute
# ----------------------------------------------------------

@app.get("/api/compute")
def api_compute():
    try:
        w_gold = float(request.args.get("w_gold", 0.2))
        capital = float(request.args.get("capital", 10000))

        if capital <= 0:
            return _json_error("Capitale non valido.", 400)

        ls80 = read_price_csv(LS80_FILE)
        gold = read_price_csv(GOLD_FILE)
        world = read_price_csv(WORLD_FILE)

        df = ls80.merge(gold, on="date", how="inner", suffixes=("_ls80", "_gold"))
        df = df.merge(world, on="date", how="inner")
        df.columns = ["date", "ls80", "gold", "world"]

        if len(df) < 20:
            return _json_error(
                "Serie troppo corta dopo il merge tra LS80, Oro e MSCI World.",
                400,
                rows_inner=int(len(df)),
            )

        dates = df["date"].reset_index(drop=True)

        portfolio = compute_portfolio(df["ls80"], df["gold"], w_gold, capital).reset_index(drop=True)
        world_scaled = (capital * (df["world"] / df["world"].iloc[0])).reset_index(drop=True)

        cagr_port = compute_cagr(portfolio, dates)
        maxdd_port = compute_max_dd(portfolio)
        dbl_years = doubling_years(cagr_port)

        cagr_world = compute_cagr(world_scaled, dates)
        maxdd_world = compute_max_dd(world_scaled)

        dd_port = compute_drawdown_series_pct(portfolio)
        dd_world = compute_drawdown_series_pct(world_scaled)

        worst_port = worst_drawdowns(portfolio, dates, n=3)
        worst_world = worst_drawdowns(world_scaled, dates, n=3)

        mask_2025 = dates.dt.year == 2025
        dd_2025_port = float((dd_port[mask_2025] / 100.0).min()) if mask_2025.any() else None
        dd_2025_world = float((dd_world[mask_2025] / 100.0).min()) if mask_2025.any() else None

        w_ls80 = 1.0 - w_gold
        az = 0.80 * w_ls80
        ob = 0.20 * w_ls80
        years_period = (dates.iloc[-1] - dates.iloc[0]).days / 365.25

        return jsonify(
            {
                "ok": True,
                "dates": dates.dt.strftime("%Y-%m-%d").tolist(),
                "portfolio": portfolio.tolist(),
                "world": world_scaled.tolist(),
                "drawdown_portfolio_pct": dd_port.tolist(),
                "drawdown_world_pct": dd_world.tolist(),
                "composition": {
                    "azionario": az,
                    "obbligazionario": ob,
                    "oro": w_gold,
                },
                "metrics": {
                    "final_portfolio": float(portfolio.iloc[-1]),
                    "final_years": float(years_period),
                    "cagr_portfolio": float(cagr_port),
                    "max_dd_portfolio": float(maxdd_port),
                    "doubling_years_portfolio": float(dbl_years) if dbl_years is not None else None,
                    "cagr_world": float(cagr_world),
                    "max_dd_world": float(maxdd_world),
                    "dd_2025_portfolio": dd_2025_port,
                    "dd_2025_world": dd_2025_world,
                    "worst_episodes_portfolio": worst_port,
                    "worst_episodes_world": worst_world,
                },
            }
        )

    except Exception as e:
        return jsonify({"ok": False, "error": f"Errore compute: {str(e)}"})


# ----------------------------------------------------------
# Assistente
# ----------------------------------------------------------

@app.post("/api/ask")
def api_ask():
    try:
        data = request.get_json(silent=True) or {}
        question = (data.get("question") or data.get("q") or "").strip()
        if not question:
            return _json_error("Scrivi una domanda.", 400)

        client = _openai_client()
        if client is None:
            return jsonify(
                {
                    "ok": True,
                    "answer": "Assistente non configurato: manca OPENAI_API_KEY su Render.",
                }
            )

        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Rispondi in italiano, in modo semplice e pratico. "
                        "Contesto: sito Metodo Pigro. Informazione generale, non consulenza personalizzata."
                    ),
                },
                {"role": "user", "content": question},
            ],
        )

        answer = ""
        try:
            if resp and resp.choices and resp.choices[0].message and resp.choices[0].message.content:
                answer = resp.choices[0].message.content
        except Exception:
            answer = ""

        if not answer:
            answer = "Nessuna risposta disponibile."

        return jsonify({"ok": True, "answer": answer.strip()})

    except Exception as e:
        return _json_error(f"{type(e).__name__}: {e}", 500)


# ----------------------------------------------------------
# Update lato server con Twelve Data
# ----------------------------------------------------------

@app.get("/api/update_data")
def api_update_data():
    err = _require_token()
    if err is not None:
        return err

    try:
        res_ls = update_one_asset(LS80_FILE, LS80_TICKER)
        res_gd = update_one_asset(GOLD_FILE, GOLD_TICKER)
        res_wd = update_one_asset(WORLD_FILE, WORLD_TICKER)

        usable_all = all([
            bool(res_ls.get("usable")),
            bool(res_gd.get("usable")),
            bool(res_wd.get("usable")),
        ])

        return jsonify(
            {
                "ok": usable_all,
                "ls80": res_ls,
                "gold": res_gd,
                "world": res_wd,
                "time_utc": _now_iso(),
            }
        )

    except Exception as e:
        return _json_error(f"{type(e).__name__}: {e}", 500)


@app.get("/api/force_update")
def api_force_update():
    return api_update_data()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=True)
