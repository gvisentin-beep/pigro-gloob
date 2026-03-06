from __future__ import annotations

import os
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, Tuple, Optional, List

import pandas as pd
from flask import Flask, jsonify, request, send_file, render_template
import yfinance as yf

# ----------------------------
# Flask app
# ----------------------------
app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
TEMPLATES_DIR = BASE_DIR / "templates"

# ----------------------------
# Config & env
# ----------------------------
UPDATE_TOKEN = os.getenv("UPDATE_TOKEN", "").strip()
ASK_DAILY_LIMIT = int(os.getenv("ASK_DAILY_LIMIT", "10"))

LS80_TICKER = os.getenv("LS80_TICKER", "VNGA80.MI").strip()
GOLD_TICKER = os.getenv("GOLD_TICKER", "SGLD.MI").strip()
WORLD_TICKER = os.getenv("WORLD_TICKER", "SMSWLD.MI").strip()

LS80_FILE = DATA_DIR / "ls80.csv"
GOLD_FILE = DATA_DIR / "gold.csv"
WORLD_FILE = DATA_DIR / "world.csv"

# ----------------------------
# Helpers
# ----------------------------
def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _json_error(msg: str, status: int = 400, **extra: Any):
    payload = {"ok": False, "error": msg}
    if extra:
        payload.update(extra)
    return jsonify(payload), status


def _read_price_csv(path: Path) -> pd.DataFrame:
    """
    CSV formato: date;close
    date in dd/mm/YYYY
    """
    if not path.exists():
        raise FileNotFoundError(str(path))
    df = pd.read_csv(path, sep=";", dtype=str)
    if "date" not in df.columns or "close" not in df.columns:
        raise ValueError(f"CSV non valido (mancano colonne date/close): {path}")

    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
    return df[["date", "close"]].copy()


def _merge_on_date(df1: pd.DataFrame, df2: pd.DataFrame, how: str = "inner") -> pd.DataFrame:
    return df1.merge(df2, on="date", how=how)


def _compute_cagr(series: pd.Series, dates: pd.Series) -> float:
    if len(series) < 2:
        return 0.0
    years = (dates.iloc[-1] - dates.iloc[0]).days / 365.25
    if years <= 0:
        return 0.0
    return float((series.iloc[-1] / series.iloc[0]) ** (1 / years) - 1.0)


def _compute_drawdown(series: pd.Series) -> float:
    peak = series.cummax()
    dd = (series / peak) - 1.0
    return float(dd.min())


def _compute_drawdown_series_pct(series: pd.Series) -> pd.Series:
    """Drawdown in percent (negative), indexed like input series."""
    peak = series.cummax()
    dd = (series / peak) - 1.0
    return dd * 100.0


def _worst_drawdown_episodes(series: pd.Series, dates: pd.Series, top_n: int = 3) -> list[dict]:
    """Return top-N worst drawdown episodes with start/bottom/end dates and depth_pct (negative)."""
    if len(series) < 3:
        return []

    values = pd.Series(series).reset_index(drop=True)
    dts = pd.to_datetime(dates).reset_index(drop=True)

    peak = values.cummax()
    dd = (values / peak) - 1.0  # negative or 0
    dd = dd.fillna(0.0)

    episodes: list[dict] = []
    in_dd = False
    start_idx = 0

    def _close_episode(s_idx: int, e_idx: int):
        sub = dd.iloc[s_idx : e_idx + 1]
        if sub.empty:
            return
        bottom_rel = int(sub.values.argmin())
        bottom_idx = s_idx + bottom_rel
        depth = float(sub.min() * 100.0)  # percent (negative)
        episodes.append(
            {
                "start": dts.iloc[s_idx].strftime("%Y-%m-%d"),
                "bottom": dts.iloc[bottom_idx].strftime("%Y-%m-%d"),
                "end": dts.iloc[e_idx].strftime("%Y-%m-%d"),
                "depth_pct": depth,
            }
        )

    for i in range(1, len(dd)):
        if not in_dd and dd.iloc[i] < 0:
            start_idx = i - 1
            while start_idx > 0 and dd.iloc[start_idx] != 0:
                start_idx -= 1
            in_dd = True
        elif in_dd and dd.iloc[i] == 0:
            _close_episode(start_idx, i)
            in_dd = False

    if in_dd:
        _close_episode(start_idx, len(dd) - 1)

    if not episodes:
        return []

    episodes.sort(key=lambda x: x["depth_pct"])  # più negativo prima
    return episodes[:top_n]


def _doubling_years(cagr: float) -> float:
    if cagr <= 0:
        return float("inf")
    # ln(2)/ln(1+r)
    import math
    return float(math.log(2.0) / math.log(1.0 + cagr))


def _annual_rebalance_portfolio(ls80: pd.Series, gold: pd.Series, dates: pd.Series, w_gold: float, capital: float) -> pd.Series:
    """
    Ribilanciamento annuale (a fine anno / primo giorno del nuovo anno disponibile).
    """
    w_gold = float(max(0.0, min(0.5, w_gold)))
    w_ls80 = 1.0 - w_gold

    # Start: investo capital secondo pesi
    shares_ls = (capital * w_ls80) / float(ls80.iloc[0])
    shares_gd = (capital * w_gold) / float(gold.iloc[0])

    out = []
    last_year = pd.to_datetime(dates.iloc[0]).year

    for i in range(len(dates)):
        d = pd.to_datetime(dates.iloc[i])
        value = shares_ls * float(ls80.iloc[i]) + shares_gd * float(gold.iloc[i])
        out.append(value)

        # se cambia anno, ribilancio al primo giorno dell'anno nuovo (cioè qui)
        if d.year != last_year:
            # ribilancio a questa data (i)
            value_now = value
            shares_ls = (value_now * w_ls80) / float(ls80.iloc[i])
            shares_gd = (value_now * w_gold) / float(gold.iloc[i])
            last_year = d.year

    return pd.Series(out)


def _write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy().sort_values("date")
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%d/%m/%Y")
    out = out[["date", "close"]]
    out.to_csv(path, sep=";", index=False, float_format="%.6g")


def _update_one_asset(yf_mod, ticker: str, out_file: Path) -> Dict[str, Any]:
    """
    Scarica dati recenti (60d) e li fonde con lo storico locale.
    """
    result = {"asset": ticker, "file": str(out_file), "updated": False, "reason": ""}

    try:
        raw = yf_mod.download(ticker, period="60d", interval="1d", progress=False, auto_adjust=False)
        if raw is None or raw.empty:
            result["reason"] = "no_data_from_yahoo"
            return result

        if isinstance(raw.index, pd.DatetimeIndex):
            raw = raw.reset_index()

        if "Adj Close" in raw.columns:
            raw["close"] = raw["Adj Close"]
        elif "Close" in raw.columns:
            raw["close"] = raw["Close"]
        else:
            result["reason"] = "no_close_column"
            return result

        tmp = raw[["Date", "close"]].rename(columns={"Date": "date"}).copy()
        tmp["date"] = pd.to_datetime(tmp["date"]).dt.normalize()
        tmp["close"] = pd.to_numeric(tmp["close"], errors="coerce")
        tmp = tmp.dropna(subset=["date", "close"]).sort_values("date")

        if out_file.exists():
            old = _read_price_csv(out_file)
            merged = pd.concat([old, tmp[["date", "close"]]], ignore_index=True)
            merged = merged.drop_duplicates(subset=["date"], keep="last").sort_values("date")
        else:
            merged = tmp[["date", "close"]]

        _write_csv(out_file, merged)
        result["updated"] = True
        return result

    except Exception as e:
        result["reason"] = f"exception: {type(e).__name__}: {e}"
        return result


# ----------------------------
# Web: homepage
# ----------------------------
@app.get("/")
def home():
    return render_template("index.html")


# ----------------------------
# API: diag
# ----------------------------
@app.get("/api/diag")
def api_diag():
    diag: Dict[str, Any] = {
        "base_dir": str(BASE_DIR),
        "build_id": os.getenv("RENDER_GIT_COMMIT", "") or os.getenv("BUILD_ID", "") or "local",
        "data_dir": str(DATA_DIR),
        "env": {
            "ASK_DAILY_LIMIT": ASK_DAILY_LIMIT,
            "GOLD_TICKER": GOLD_TICKER,
            "WORLD_TICKER": WORLD_TICKER,
            "LS80_TICKER": LS80_TICKER,
            "OPENAI_API_KEY_present": bool(os.getenv("OPENAI_API_KEY")),
            "OPENAI_MODEL": os.getenv("OPENAI_MODEL", ""),
            "UPDATE_TOKEN_present": bool(UPDATE_TOKEN),
        },
        "files": {},
        "ok": True,
        "time_utc": _now_iso(),
        "versions": {
            "python": os.getenv("PYTHON_VERSION", ""),
        },
    }

    def file_info(p: Path) -> Dict[str, Any]:
        if not p.exists():
            return {"exists": False, "file": str(p)}
        try:
            df = _read_price_csv(p)
            return {
                "exists": True,
                "file": str(p),
                "first_date": str(df["date"].iloc[0].date()),
                "last_date": str(df["date"].iloc[-1].date()),
                "first_value": float(df["close"].iloc[0]),
                "last_value": float(df["close"].iloc[-1]),
                "rows": int(len(df)),
            }
        except Exception as e:
            return {"exists": True, "file": str(p), "error": f"{type(e).__name__}: {e}"}

    diag["files"]["gold"] = file_info(GOLD_FILE)
    diag["files"]["world"] = file_info(WORLD_FILE)
    diag["files"]["ls80"] = file_info(LS80_FILE)

    try:
        ls = _read_price_csv(LS80_FILE)
        gd = _read_price_csv(GOLD_FILE)
        wd = _read_price_csv(WORLD_FILE)

        merged_lg = ls.merge(gd, on="date", how="inner", suffixes=("_ls80", "_gold"))
        merged_all = merged_lg.merge(wd, on="date", how="inner")

        diag["merge"] = {
            "rows_inner_ls80_gold": int(len(merged_lg)),
            "rows_inner_all3": int(len(merged_all)),
            "first_date": str(merged_all["date"].iloc[0].date()) if len(merged_all) else (str(merged_lg["date"].iloc[0].date()) if len(merged_lg) else None),
            "last_date": str(merged_all["date"].iloc[-1].date()) if len(merged_all) else (str(merged_lg["date"].iloc[-1].date()) if len(merged_lg) else None),
        }
    except Exception as e:
        diag["merge_error"] = f"{type(e).__name__}: {e}"

    return jsonify(diag)


# ----------------------------
# API: update data (Render endpoint)
# ----------------------------
@app.get("/api/update_data")
def api_update_data():
    token = request.args.get("token", "").strip()
    if not UPDATE_TOKEN or token != UPDATE_TOKEN:
        return _json_error("Token non valido.", 403)

    try:
        res_ls = _update_one_asset(yf, LS80_TICKER, LS80_FILE)
        res_gd = _update_one_asset(yf, GOLD_TICKER, GOLD_FILE)
        res_wd = _update_one_asset(yf, WORLD_TICKER, WORLD_FILE)
        return jsonify({"ok": True, "ls80": res_ls, "gold": res_gd, "world": res_wd, "time_utc": _now_iso()})
    except Exception as e:
        return _json_error(f"Errore update_data: {type(e).__name__}: {e}", 500)


@app.get("/api/force_update")
def api_force_update():
    token = request.args.get("token", "").strip()
    if not UPDATE_TOKEN or token != UPDATE_TOKEN:
        return _json_error("Token non valido.", 403)

    try:
        res_ls = _update_one_asset(yf, LS80_TICKER, LS80_FILE)
        res_gd = _update_one_asset(yf, GOLD_TICKER, GOLD_FILE)
        res_wd = _update_one_asset(yf, WORLD_TICKER, WORLD_FILE)

        after = {}
        for name, p in [("ls80", LS80_FILE), ("gold", GOLD_FILE), ("world", WORLD_FILE)]:
            if p.exists():
                df = _read_price_csv(p)
                after[name] = {
                    "rows": int(len(df)),
                    "last_date": str(df["date"].iloc[-1].date()),
                    "last_value": float(df["close"].iloc[-1]),
                }
            else:
                after[name] = {"exists": False}

        return jsonify({"ok": True, "ls80": res_ls, "gold": res_gd, "world": res_wd, "after_file": after, "time_utc": _now_iso()})
    except Exception as e:
        return _json_error(f"Errore force_update: {type(e).__name__}: {e}", 500)


# ----------------------------
# API: compute portfolio & world
# ----------------------------
@app.get("/api/compute")
def api_compute():
    try:
        w_gold = float(request.args.get("w_gold", "0.20"))
        w_gold = max(0.0, min(0.50, w_gold))

        capital = float(request.args.get("capital", "10000"))
        if capital <= 0:
            return _json_error("Capitale non valido.", 400)

        ls = _read_price_csv(LS80_FILE)
        gd = _read_price_csv(GOLD_FILE)
        wd = _read_price_csv(WORLD_FILE)

        # ✅ A: rigoroso → solo date comuni tra LS80 e Oro
        df = ls.merge(gd, on="date", how="inner", suffixes=("_ls80", "_gold")).rename(
            columns={"close_ls80": "ls80", "close_gold": "gold"}
        )

        if len(df) < 20:
            return _json_error(
                "Poche date in comune tra LS80 e Oro (merge troppo corto).",
                400,
                rows_inner=int(len(df)),
            )

        dates = df["date"]

        # World: allinea sullo stesso asse temporale del portafoglio (ffill sulle date mancanti)
        wd2 = wd.rename(columns={"close": "world"}).copy()
        wd2 = wd2[["date", "world"]].sort_values("date")
        base = pd.DataFrame({"date": dates})
        world_aligned = base.merge(wd2, on="date", how="left").sort_values("date")
        world_aligned["world"] = world_aligned["world"].ffill()
        world_aligned = world_aligned.dropna(subset=["world"])

        # Se per qualche ragione l'allineamento taglia troppo, ricalibra su date comuni
        if len(world_aligned) < 20:
            df_all = df.merge(wd2, on="date", how="inner")
            dates = df_all["date"]
            df = df_all[["date", "ls80", "gold"]]
            world_series = df_all["world"]
        else:
            # Mantieni le date del portafoglio, ma taglia l'inizio fino a quando World è disponibile
            first_ok = int(world_aligned.index.min())
            df = df.iloc[first_ok:].reset_index(drop=True)
            dates = df["date"]
            world_series = world_aligned["world"].iloc[first_ok:].reset_index(drop=True)

        # Portafoglio ribilanciamento annuale
        port = _annual_rebalance_portfolio(df["ls80"], df["gold"], dates, w_gold=w_gold, capital=float(capital))

        # World normalizzato sul capitale iniziale (buy&hold)
        world_scaled = float(capital) * (world_series / float(world_series.iloc[0]))

        cagr = _compute_cagr(port, dates)
        max_dd = _compute_drawdown(port)
        dbl = _doubling_years(cagr)

        cagr_w = _compute_cagr(world_scaled, dates)
        max_dd_w = _compute_drawdown(world_scaled)

        # Drawdown serie (%)
        dd_port_pct = _compute_drawdown_series_pct(port)
        dd_world_pct = _compute_drawdown_series_pct(world_scaled)

        # "Dazi Trump 2025" = peggiore drawdown dentro il 2025
        try:
            mask_2025 = pd.to_datetime(dates).dt.year == 2025
            dd_2025_port = float((dd_port_pct[mask_2025] / 100.0).min()) if mask_2025.any() else None
            dd_2025_world = float((dd_world_pct[mask_2025] / 100.0).min()) if mask_2025.any() else None
        except Exception:
            dd_2025_port = None
            dd_2025_world = None

        worst3_port = _worst_drawdown_episodes(port, dates, top_n=3)
        worst3_world = _worst_drawdown_episodes(world_scaled, dates, top_n=3)

        w_ls80 = 1.0 - w_gold
        az = 0.80 * w_ls80
        ob = 0.20 * w_ls80

        years_period = (dates.iloc[-1] - dates.iloc[0]).days / 365.25
        final_value = float(port.iloc[-1])

        metrics = {
            "cagr_portfolio": cagr,
            "max_dd_portfolio": max_dd,
            "doubling_years_portfolio": dbl,
            "final_portfolio": final_value,
            "final_years": years_period,
            "weights": {"gold": w_gold, "ls80": w_ls80, "equity": az, "bond": ob},
            "last_data_date": str(dates.iloc[-1].date()),
            # World
            "cagr_world": cagr_w,
            "max_dd_world": max_dd_w,
            "dd_2025_portfolio": dd_2025_port,
            "dd_2025_world": dd_2025_world,
            "worst_episodes_portfolio": worst3_port,
            "worst_episodes_world": worst3_world,
        }

        payload: Dict[str, Any] = {
            "ok": True,
            "dates": [d.strftime("%Y-%m-%d") for d in dates],
            "portfolio": [float(x) for x in port],
            "world": [float(x) for x in world_scaled],
            "drawdown_portfolio_pct": [float(x) for x in dd_port_pct],
            "drawdown_world_pct": [float(x) for x in dd_world_pct],
            "metrics": metrics,
            "composition": {"azionario": az, "obbligazionario": ob, "oro": w_gold},
        }
        return jsonify(payload)

    except Exception as e:
        return _json_error(f"Errore compute: {type(e).__name__}: {e}", 500)


# ----------------------------
# PDF endpoint (grafico)
# ----------------------------
@app.get("/api/pdf")
def api_pdf():
    # (mantengo la tua logica PDF già presente)
    try:
        from io import BytesIO
        from reportlab.pdfgen import canvas as rl_canvas
        from reportlab.lib.pagesizes import A4

        title = request.args.get("title", "Gloob - Metodo Pigro")
        cagr = request.args.get("cagr", "")
        maxdd = request.args.get("maxdd", "")
        finalv = request.args.get("final", "")
        years = request.args.get("years", "")

        buffer = BytesIO()
        c = rl_canvas.Canvas(buffer, pagesize=A4)
        width, height = A4

        c.setFont("Helvetica-Bold", 16)
        c.drawString(50, height - 60, title)

        c.setFont("Helvetica", 12)
        y = height - 100
        for line in [
            f"Rendimento annualizzato: {cagr}",
            f"Max Ribasso nel periodo: {maxdd}",
            f"Finale: {finalv} (in anni {years})",
        ]:
            c.drawString(50, y, line)
            y -= 18

        c.setFont("Helvetica", 10)
        c.drawString(50, 40, "Nota: documento informativo, non consulenza finanziaria.")
        c.showPage()
        c.save()

        buffer.seek(0)
        return send_file(buffer, mimetype="application/pdf", as_attachment=True, download_name="gloob_pigro.pdf")

    except Exception as e:
        return _json_error(f"Errore pdf: {type(e).__name__}: {e}", 500)


if __name__ == "__main__":
    # Local debug
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=True)
