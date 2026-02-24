from __future__ import annotations

import os
import json
import math
import traceback
from datetime import datetime, date, timedelta
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


# ----------------------------
# App / Paths / Env
# ----------------------------
app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("DATA_DIR", str(BASE_DIR / "data")))

LS80_FILE = DATA_DIR / "ls80.csv"
GOLD_FILE = DATA_DIR / "gold.csv"

ASK_DAILY_LIMIT = int(os.getenv("ASK_DAILY_LIMIT", "10"))
ASK_STORE_FILE = DATA_DIR / "ask_limits.json"

UPDATE_TOKEN = os.getenv("UPDATE_TOKEN", "").strip()

LS80_TICKER = os.getenv("LS80_TICKER", "VNGA80.MI").strip()
GOLD_TICKER = os.getenv("GOLD_TICKER", "SGLD.MI").strip()

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini").strip()

# Auto update (safe)
AUTO_UPDATE = os.getenv("AUTO_UPDATE", "1").strip().lower() not in {"0", "false", "no"}
AUTO_UPDATE_MINUTES = int(os.getenv("AUTO_UPDATE_MINUTES", "0") or "0")  # se >0: cooldown in minuti
UPDATE_MIN_INTERVAL_HOURS = float(os.getenv("UPDATE_MIN_INTERVAL_HOURS", "6") or "6")

# Firma build (per vedere subito da /api/diag che Render gira QUESTO file)
BUILD_ID = os.getenv("BUILD_ID", "2026-02-24_autoupdate_forceupdate_v2").strip()


# ----------------------------
# In-memory cache per evitare update continui
# ----------------------------
_LAST_UPDATE_AT_UTC: Optional[datetime] = None
_LAST_UPDATE_RESULT: Dict[str, Any] = {}


# ----------------------------
# Helpers
# ----------------------------
def _now_utc() -> datetime:
    return datetime.utcnow()


def _now_iso() -> str:
    return _now_utc().strftime("%Y-%m-%d %H:%M:%S UTC")


def _safe_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _detect_sep(file_path: Path) -> str:
    """
    Supporta CSV con:
      - Date;Close
      - Date,Close
    """
    try:
        with file_path.open("r", encoding="utf-8", errors="ignore") as f:
            head = f.readline()
        if ";" in head and "," not in head:
            return ";"
        if "," in head and ";" not in head:
            return ","
        return ";" if head.count(";") >= head.count(",") else ","
    except Exception:
        return ";"


def _read_price_csv(file_path: Path) -> pd.DataFrame:
    """
    Ritorna DataFrame con colonne:
      - date (datetime naive)
      - close (float)
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
        raise ValueError(f"CSV {file_path.name}: troppo poche righe valide ({len(out)}).")

    try:
        out["date"] = out["date"].dt.tz_localize(None)
    except Exception:
        pass

    return out


def _write_price_csv(file_path: Path, df: pd.DataFrame) -> None:
    """
    Salva nel formato atteso (Date dd/mm/YYYY ; Close con 2 decimali, separatore ';')
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    save = pd.DataFrame(
        {
            "Date": pd.to_datetime(df["date"]).dt.strftime("%d/%m/%Y"),
            "Close": pd.to_numeric(df["close"]).map(lambda x: f"{float(x):.2f}"),
        }
    )
    save.to_csv(file_path, sep=";", index=False)


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
    Portafoglio 2 strumenti ribilanciato 1 volta l'anno (primo giorno di borsa dell'anno).
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
        values.append(v)

        if i != 0 and bool(rebalance_flags.iloc[i]):
            v_now = v
            ls80_shares = (v_now * w_ls80) / float(ls80_close.iloc[i])
            gold_shares = (v_now * w_gold) / float(gold_close.iloc[i])

    return pd.Series(values)


def _client_ip() -> str:
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
    """
    Su Render il filesystem può essere effimero: non bloccare se non si riesce.
    """
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        ASK_STORE_FILE.write_text(json.dumps(store, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


def _check_and_consume_quota(ip: str) -> Tuple[int, int]:
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


def _pkg_version(name: str) -> Optional[str]:
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        return None


def _routes_list() -> list[dict[str, str]]:
    routes = []
    for r in app.url_map.iter_rules():
        methods = ",".join(sorted(m for m in r.methods if m not in {"HEAD", "OPTIONS"}))
        routes.append({"rule": str(r), "methods": methods, "endpoint": r.endpoint})
    routes.sort(key=lambda x: (x["rule"], x["methods"]))
    return routes


def _json_error(message: str, status: int = 500, **extra: Any):
    payload = {"ok": False, "error": message, **extra}
    return jsonify(payload), status


def _yf_to_hist_df(hist: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Converte l'output di yfinance.download() in DataFrame standard:
      - date (datetime naive)
      - close (float)

    Gestisce anche MultiIndex.
    """
    if hist is None or hist.empty:
        return pd.DataFrame(columns=["date", "close"])

    close_ser = None

    if isinstance(hist.columns, pd.MultiIndex):
        lvl0 = set(hist.columns.get_level_values(0))
        if "Close" in lvl0:
            close_obj = hist["Close"]
            if hasattr(close_obj, "columns"):
                if ticker in close_obj.columns:
                    close_ser = close_obj[ticker]
                else:
                    close_ser = close_obj.iloc[:, 0]
            else:
                close_ser = close_obj
    else:
        if "Close" in hist.columns:
            close_ser = hist["Close"]

    if close_ser is None:
        return pd.DataFrame(columns=["date", "close"])

    tmp = close_ser.dropna().to_frame("close")
    tmp["date"] = tmp.index
    tmp["date"] = pd.to_datetime(tmp["date"], errors="coerce")

    try:
        tmp["date"] = tmp["date"].dt.tz_localize(None)
    except Exception:
        pass

    tmp["close"] = pd.to_numeric(tmp["close"], errors="coerce")
    tmp = tmp.dropna().sort_values("date")
    return tmp[["date", "close"]]


def _cooldown_seconds() -> int:
    """
    Cooldown per auto-update.
    - se AUTO_UPDATE_MINUTES > 0: usa minuti
    - altrimenti usa UPDATE_MIN_INTERVAL_HOURS
    """
    if AUTO_UPDATE_MINUTES and AUTO_UPDATE_MINUTES > 0:
        return int(AUTO_UPDATE_MINUTES * 60)
    return int(UPDATE_MIN_INTERVAL_HOURS * 3600)


def _should_run_update(force: bool = False) -> bool:
    global _LAST_UPDATE_AT_UTC
    if force:
        return True
    if not AUTO_UPDATE:
        return False
    if _LAST_UPDATE_AT_UTC is None:
        return True
    return (_now_utc() - _LAST_UPDATE_AT_UTC).total_seconds() >= _cooldown_seconds()


def _update_one_asset_in_memory(
    yf, ticker: str, file_path: Path
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Legge il CSV locale e prova ad aggiungere righe da Yahoo.

    Ritorna:
      - df_final (anche se non salvato su disco)
      - info (include persisted=True/False e reason)
    """
    info: Dict[str, Any] = {
        "asset": file_path.stem,
        "ticker_used": ticker,
        "file": str(file_path),
        "updated": False,
        "persisted": False,
    }

    df_local = _read_price_csv(file_path)
    last_local = df_local["date"].iloc[-1].date()
    info["last_local_date"] = str(last_local)

    # ✅ richiesta robusta: dal giorno successivo all'ultima data locale fino a oggi (+1 end)
    start = (last_local + timedelta(days=1)).strftime("%Y-%m-%d")
    end = (date.today() + timedelta(days=1)).strftime("%Y-%m-%d")
    info["requested_start"] = start
    info["requested_end"] = end

    hist_raw = yf.download(
        ticker,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=False,
        progress=False,
    )
    hist = _yf_to_hist_df(hist_raw, ticker)

    # fallback: a volte start/end su EU tickers dà vuoto; riproviamo con una finestra più ampia
    if hist.empty:
        hist_raw2 = yf.download(
            ticker,
            period="6mo",
            interval="1d",
            auto_adjust=False,
            progress=False,
        )
        hist2 = _yf_to_hist_df(hist_raw2, ticker)
        hist = hist2
        info["fallback"] = "period_6mo"

    if hist.empty:
        info["reason"] = "no_data_from_yahoo"
        return df_local, info

    new_rows = hist[hist["date"].dt.date > last_local]
    if new_rows.empty:
        info["reason"] = "already_up_to_date"
        info["last_yahoo_date"] = str(hist["date"].iloc[-1].date())
        return df_local, info

    out = pd.concat([df_local, new_rows], ignore_index=True)
    out = out.sort_values("date").drop_duplicates(subset=["date"], keep="last")

    info["updated"] = True
    info["added_rows"] = int(len(new_rows))
    info["last_date"] = str(out["date"].iloc[-1].date())
    info["last_value"] = float(out["close"].iloc[-1])

    # prova a persistere su disco (se fallisce, comunque useremo out in RAM)
    try:
        _write_price_csv(file_path, out)
        info["persisted"] = True
    except Exception as e:
        info["persisted"] = False
        info["persist_error"] = f"{type(e).__name__}: {e}"

    return out, info


def _maybe_update_data(force: bool = False) -> Dict[str, Any]:
    """
    Esegue aggiornamento (se non in cooldown).
    Salva risultato in cache e ritorna info.
    """
    global _LAST_UPDATE_AT_UTC, _LAST_UPDATE_RESULT

    if not _should_run_update(force=force):
        return {"skipped": True, "cooldown_seconds": _cooldown_seconds(), **_LAST_UPDATE_RESULT}

    try:
        import yfinance as yf
    except Exception as e:
        _LAST_UPDATE_AT_UTC = _now_utc()
        _LAST_UPDATE_RESULT = {"ok": False, "reason": f"yfinance_not_available: {type(e).__name__}: {e}"}
        return _LAST_UPDATE_RESULT

    try:
        df_ls, info_ls = _update_one_asset_in_memory(yf, LS80_TICKER, LS80_FILE)
        df_gd, info_gd = _update_one_asset_in_memory(yf, GOLD_TICKER, GOLD_FILE)

        _LAST_UPDATE_AT_UTC = _now_utc()
        _LAST_UPDATE_RESULT = {
            "ok": True,
            "time_utc": _now_iso(),
            "ls80": info_ls,
            "gold": info_gd,
        }
        return _LAST_UPDATE_RESULT
    except Exception as e:
        _LAST_UPDATE_AT_UTC = _now_utc()
        _LAST_UPDATE_RESULT = {"ok": False, "reason": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()}
        return _LAST_UPDATE_RESULT


def _get_latest_dataframes() -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Ritorna:
      - df_ls80 (aggiornato in RAM se update riuscito)
      - df_gold (aggiornato in RAM se update riuscito)
      - info_update (dettagli)
    """
    update_info = _maybe_update_data(force=False)

    # Se update eseguito con successo e ha aggiornato, potremmo voler rileggere i file,
    # ma per sicurezza (Render) usiamo sempre la lettura da file per coerenza,
    # e ci fidiamo dell'update in RAM solo per la compute (vedi sotto).
    # Qui facciamo semplice: leggiamo da file; se persisted=False ma updated=True,
    # allora leggiamo ancora da file (che resterà vecchio) -> quindi in compute useremo out in RAM.
    #
    # Per questo, in compute rifacciamo update_one_in_memory quando necessario.
    #
    # Qui ritorniamo i df da file come baseline:
    df_ls_file = _read_price_csv(LS80_FILE)
    df_gd_file = _read_price_csv(GOLD_FILE)

    return df_ls_file, df_gd_file, update_info


# ----------------------------
# Pages
# ----------------------------
@app.get("/")
def home():
    return render_template("index.html")


@app.get("/health")
def health():
    return jsonify({"ok": True, "time_utc": _now_iso()})


# ----------------------------
# Diagnostics
# ----------------------------
@app.get("/api/diag/routes")
def api_diag_routes():
    return jsonify({"ok": True, "routes": _routes_list(), "time_utc": _now_iso()})


@app.get("/api/diag")
def api_diag():
    diag: Dict[str, Any] = {
        "ok": True,
        "time_utc": _now_iso(),
        "build_id": BUILD_ID,
        "base_dir": str(BASE_DIR),
        "data_dir": str(DATA_DIR),
        "env": {
            "LS80_TICKER": LS80_TICKER,
            "GOLD_TICKER": GOLD_TICKER,
            "AUTO_UPDATE": AUTO_UPDATE,
            "AUTO_UPDATE_MINUTES": AUTO_UPDATE_MINUTES,
            "UPDATE_MIN_INTERVAL_HOURS": UPDATE_MIN_INTERVAL_HOURS,
            "OPENAI_API_KEY_present": bool(os.getenv("OPENAI_API_KEY", "").strip()),
            "OPENAI_MODEL": OPENAI_MODEL,
            "UPDATE_TOKEN_present": bool(UPDATE_TOKEN),
            "ASK_DAILY_LIMIT": ASK_DAILY_LIMIT,
        },
        "versions": {
            "python": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
            "flask": _pkg_version("flask"),
            "gunicorn": _pkg_version("gunicorn"),
            "pandas": _pkg_version("pandas"),
            "numpy": _pkg_version("numpy"),
            "openai": _pkg_version("openai"),
            "yfinance": _pkg_version("yfinance"),
            "requests": _pkg_version("requests"),
        },
        "files": {},
        "merge": {},
        "last_update_cache": {
            "last_update_at_utc": _LAST_UPDATE_AT_UTC.strftime("%Y-%m-%d %H:%M:%S UTC") if _LAST_UPDATE_AT_UTC else None,
            "last_update_result": _LAST_UPDATE_RESULT,
        },
        "routes_count": len(_routes_list()),
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
                    "sep_detected": _detect_sep(fp),
                    "first_date": str(df["date"].iloc[0].date()),
                    "last_date": str(df["date"].iloc[-1].date()),
                    "first_value": float(df["close"].iloc[0]),
                    "last_value": float(df["close"].iloc[-1]),
                }
            )
        except Exception as e:
            info["error"] = f"{type(e).__name__}: {e}"
        return info

    diag["files"]["ls80"] = file_info(LS80_FILE)
    diag["files"]["gold"] = file_info(GOLD_FILE)

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


@app.get("/api/force_update")
def api_force_update():
    """
    Forza update (bypassa cooldown) e restituisce dettagli + last_date dopo.
    """
    global _LAST_UPDATE_AT_UTC, _LAST_UPDATE_RESULT
    _LAST_UPDATE_AT_UTC = None
    _LAST_UPDATE_RESULT = {}

    res = _maybe_update_data(force=True)

    # rileggi da file per vedere se la persistenza ha funzionato
    after = {}
    try:
        after["ls80_last_date_file"] = str(_read_price_csv(LS80_FILE)["date"].iloc[-1].date())
    except Exception as e:
        after["ls80_last_date_file_error"] = f"{type(e).__name__}: {e}"
    try:
        after["gold_last_date_file"] = str(_read_price_csv(GOLD_FILE)["date"].iloc[-1].date())
    except Exception as e:
        after["gold_last_date_file_error"] = f"{type(e).__name__}: {e}"

    return jsonify({"ok": True, "result": res, "after_file": after, "time_utc": _now_iso()})


# ----------------------------
# API: compute (grafico + metriche)
# ----------------------------
@app.get("/api/compute")
def api_compute():
    try:
        w_gold = _safe_float(request.args.get("w_gold"))
        if w_gold is None:
            w_ls80 = _safe_float(request.args.get("w_ls80"))
            if w_ls80 is None:
                w_gold = 0.20
            else:
                w_gold = 1.0 - float(w_ls80)

        if w_gold > 1.0:
            w_gold = w_gold / 100.0

        w_gold = float(np.clip(w_gold, 0.0, 0.50))

        capital = _safe_float(request.args.get("capital")) or _safe_float(request.args.get("initial")) or 10000.0
        if capital <= 0:
            capital = 10000.0

        # baseline da file
        df_ls_file = _read_price_csv(LS80_FILE)
        df_gd_file = _read_price_csv(GOLD_FILE)

        update_info = _maybe_update_data(force=False)

        # Se update dice updated=True ma persisted=False, i file potrebbero essere rimasti vecchi.
        # Per garantire grafico aggiornato, in quel caso rifacciamo update in RAM per compute.
        df_ls = df_ls_file
        df_gd = df_gd_file
        ram_used = {"ls80": False, "gold": False}

        try:
            import yfinance as yf
        except Exception:
            yf = None

        if yf is not None and isinstance(update_info, dict) and update_info.get("ok"):
            ls_info = update_info.get("ls80") or {}
            gd_info = update_info.get("gold") or {}

            if ls_info.get("updated") and not ls_info.get("persisted"):
                df_ls, _ = _update_one_asset_in_memory(yf, LS80_TICKER, LS80_FILE)
                ram_used["ls80"] = True
            if gd_info.get("updated") and not gd_info.get("persisted"):
                df_gd, _ = _update_one_asset_in_memory(yf, GOLD_TICKER, GOLD_FILE)
                ram_used["gold"] = True

        df = df_ls.merge(df_gd, on="date", how="inner", suffixes=("_ls80", "_gold"))
        df = df.rename(columns={"close_ls80": "ls80", "close_gold": "gold"})

        if len(df) < 20:
            return _json_error(
                "Poche date in comune tra LS80 e Oro (merge troppo corto).",
                400,
                rows_inner=int(len(df)),
            )

        dates = df["date"]
        port = _annual_rebalance_portfolio(df["ls80"], df["gold"], dates, w_gold=w_gold, capital=float(capital))

        cagr = _compute_cagr(port, dates)
        max_dd = _compute_drawdown(port)
        dbl = _doubling_years(cagr)

        w_ls80 = 1.0 - w_gold
        az = 0.80 * w_ls80
        ob = 0.20 * w_ls80

        years_period = (dates.iloc[-1] - dates.iloc[0]).days / 365.25
        final_value = float(port.iloc[-1])

        last_data_date = str(dates.iloc[-1].date())

        metrics = {
            "cagr_portfolio": cagr,
            "max_dd_portfolio": max_dd,
            "doubling_years_portfolio": dbl,
            "final_portfolio": final_value,
            "final_years": years_period,
            "last_data_date": last_data_date,
            "weights": {
                "gold": w_gold,
                "ls80": w_ls80,
                "equity": az,
                "bond": ob,
            },
        }

        payload: Dict[str, Any] = {
            "ok": True,
            "dates": [d.strftime("%Y-%m-%d") for d in dates],
            "portfolio": [float(x) for x in port],
            "metrics": metrics,
            "composition": {"azionario": az, "obbligazionario": ob, "oro": w_gold},
            "data_info": {
                "updated_to": last_data_date,
                "auto_update": update_info,
                "ram_used": ram_used,
            },
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
        question = (data.get("question") or data.get("q") or "").strip()
        if not question:
            return _json_error("Scrivi una domanda.", 400)

        ip = _client_ip()
        remaining, limit = _check_and_consume_quota(ip)
        if remaining == 0 and limit > 0:
            return jsonify(
                {"ok": False, "error": "Limite giornaliero raggiunto.", "remaining": 0, "limit": limit}
            ), 429

        client = _openai_client()
        if client is None:
            return jsonify(
                {
                    "ok": True,
                    "answer": "Assistente non configurato: manca OPENAI_API_KEY su Render (Environment).",
                    "remaining": remaining,
                    "limit": limit,
                }
            )

        system_msg = (
            "Rispondi in italiano, in modo semplice e pratico.\n"
            "Contesto: sito 'Metodo Pigro' (ETF azion-obblig + oro). Informazione generale.\n"
            "Non è consulenza finanziaria personalizzata.\n"
            "Stile: breve, chiaro, con esempi pratici quando utile."
        )

        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_msg},
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
            answer = "Nessuna risposta (vuoto). Riprova tra poco."

        return jsonify({"ok": True, "answer": answer.strip(), "remaining": remaining, "limit": limit})

    except Exception as e:
        return _json_error(f"{type(e).__name__}: {e}", 500, traceback=traceback.format_exc())


# ----------------------------
# API: update data (manuale protetto da token)
# ----------------------------
@app.get("/api/update_data")
def api_update_data():
    token = (request.args.get("token") or "").strip()
    if not UPDATE_TOKEN or token != UPDATE_TOKEN:
        return _json_error("Token non valido.", 401)

    # forza update bypassando cooldown
    res = _maybe_update_data(force=True)

    after = {}
    try:
        after["ls80_last_date_file"] = str(_read_price_csv(LS80_FILE)["date"].iloc[-1].date())
    except Exception as e:
        after["ls80_last_date_file_error"] = f"{type(e).__name__}: {e}"
    try:
        after["gold_last_date_file"] = str(_read_price_csv(GOLD_FILE)["date"].iloc[-1].date())
    except Exception as e:
        after["gold_last_date_file_error"] = f"{type(e).__name__}: {e}"

    return jsonify({"ok": True, "result": res, "after_file": after, "time_utc": _now_iso()})


# ----------------------------
# Main (local)
# ----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "10000")), debug=True)
