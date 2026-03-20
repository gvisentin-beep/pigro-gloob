from __future__ import annotations

import math
import os
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import requests
from flask import Flask, jsonify, render_template, request, send_file

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

LS80_FILE = DATA_DIR / "ls80.csv"
GOLD_FILE = DATA_DIR / "gold.csv"
BTC_FILE = DATA_DIR / "btc.csv"
WORLD_FILE = DATA_DIR / "world.csv"

LS80_TICKER = os.getenv("LS80_TICKER", "VNGA80.MI").strip()
GOLD_TICKER = os.getenv("GOLD_TICKER", "GLD").strip()
BTC_TICKER = os.getenv("BTC_TICKER", "BTC/EUR").strip()
WORLD_TICKER = os.getenv("WORLD_TICKER", "URTH").strip()

UPDATE_TOKEN = os.getenv("UPDATE_TOKEN", "").strip()
TWELVE_DATA_API_KEY = os.getenv("TWELVE_DATA_API_KEY", "").strip()
TWELVE_DATA_BASE_URL = "https://api.twelvedata.com/time_series"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()

WEIGHT_LS80 = 0.80
WEIGHT_GOLD = 0.15
WEIGHT_BTC = 0.05

MIN_ROWS_REQUIRED = 20
STALE_WARNING_DAYS = 7

DEFAULT_LEVERAGE_CAPITAL = 100000.0
LEVERAGE_RATIO = 0.20
LOMBARD_RATE_ANNUAL = 0.025
DAY_COUNT = 365.25


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _now_iso() -> str:
    return _now_utc().strftime("%Y-%m-%d %H:%M:%S UTC")


def _json_error(msg: str, status: int = 400, **extra: Any):
    payload = {"ok": False, "error": msg}
    payload.update(extra)
    return jsonify(payload), status


def _require_token() -> Optional[Tuple[Any, int]]:
    if not UPDATE_TOKEN:
        return None
    token = (request.args.get("token") or "").strip()
    if token != UPDATE_TOKEN:
        return _json_error("Token non valido.", 401)
    return None


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def read_price_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(str(path))

    try:
        df = pd.read_csv(path, sep=";", dtype=str, encoding="utf-8-sig")
        df.columns = [str(c).strip().lower() for c in df.columns]
        if "date" in df.columns and "close" in df.columns:
            df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
            df["close"] = (
                df["close"].astype(str).str.strip().str.replace(",", ".", regex=False)
            )
            df["close"] = pd.to_numeric(df["close"], errors="coerce")
            df = (
                df.dropna(subset=["date", "close"])
                .sort_values("date")
                .drop_duplicates(subset=["date"], keep="last")
                .reset_index(drop=True)
            )
            if not df.empty:
                return df[["date", "close"]].copy()
    except Exception:
        pass

    df = pd.read_csv(
        path,
        sep=";",
        dtype=str,
        header=None,
        names=["date", "close"],
        encoding="utf-8-sig",
    )
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    df["close"] = df["close"].astype(str).str.strip().str.replace(",", ".", regex=False)
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = (
        df.dropna(subset=["date", "close"])
        .sort_values("date")
        .drop_duplicates(subset=["date"], keep="last")
        .reset_index(drop=True)
    )

    if df.empty:
        raise ValueError(f"CSV non valido: {path}")

    return df[["date", "close"]].copy()


def write_price_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy().sort_values("date").drop_duplicates(subset=["date"], keep="last")
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%d/%m/%Y")
    out.to_csv(path, sep=";", index=False)


def fetch_twelve_data(symbol: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    if not symbol:
        return None, "Ticker vuoto"
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
            return None, f"{data.get('code', '')} {data.get('message', 'errore sconosciuto')}".strip()

        values = data.get("values")
        if not values:
            return None, "Nessun valore restituito"

        df = pd.DataFrame(values)
        if "datetime" not in df.columns or "close" not in df.columns:
            return None, f"Colonne inattese: {list(df.columns)}"

        df["date"] = pd.to_datetime(df["datetime"], errors="coerce")
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df = (
            df.dropna(subset=["date", "close"])
            .sort_values("date")
            .drop_duplicates(subset=["date"], keep="last")
            .reset_index(drop=True)
        )
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

    old_df: Optional[pd.DataFrame] = None
    try:
        old_df = read_price_csv(path)
        info["previous_rows"] = int(len(old_df))
        info["previous_last_date"] = str(old_df["date"].iloc[-1].date())
    except Exception as e:
        info["read_error"] = f"{type(e).__name__}: {e}"

    new_df, err = fetch_twelve_data(symbol)
    if new_df is None or new_df.empty:
        info["reason"] = err or "no_data_from_twelve_data"
        if old_df is not None and not old_df.empty:
            info["usable"] = True
            info["fallback"] = True
            info["rows"] = int(len(old_df))
            info["first_date"] = str(old_df["date"].iloc[0].date())
            info["last_date"] = str(old_df["date"].iloc[-1].date())
            info["last_value"] = float(old_df["close"].iloc[-1])
        return info

    merged = pd.concat([old_df, new_df], ignore_index=True) if old_df is not None and not old_df.empty else new_df
    merged = (
        merged.drop_duplicates(subset=["date"], keep="last")
        .sort_values("date")
        .reset_index(drop=True)
    )
    write_price_csv(path, merged)

    info["updated"] = True
    info["usable"] = True
    info["fallback"] = False
    info["rows"] = int(len(merged))
    info["first_date"] = str(merged["date"].iloc[0].date())
    info["last_date"] = str(merged["date"].iloc[-1].date())
    info["last_value"] = float(merged["close"].iloc[-1])
    return info


def dataset_freshness_days(df: pd.DataFrame) -> Optional[int]:
    if df.empty:
        return None
    last_date = pd.Timestamp(df["date"].iloc[-1]).tz_localize(None)
    today = pd.Timestamp(_now_utc().date())
    return int((today - last_date.normalize()).days)


def build_merged_dataset() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    ls80 = read_price_csv(LS80_FILE).rename(columns={"close": "ls80"})
    gold = read_price_csv(GOLD_FILE).rename(columns={"close": "gold"})
    btc = read_price_csv(BTC_FILE).rename(columns={"close": "btc"})
    world = read_price_csv(WORLD_FILE).rename(columns={"close": "world"})

    freshness = {
        "ls80": dataset_freshness_days(ls80),
        "gold": dataset_freshness_days(gold),
        "btc": dataset_freshness_days(btc),
        "world": dataset_freshness_days(world),
    }

    df = ls80.merge(gold, on="date", how="outer")
    df = df.merge(btc, on="date", how="outer")
    df = df.merge(world, on="date", how="outer")
    df = df.sort_values("date").reset_index(drop=True)

    df[["ls80", "gold", "btc", "world"]] = df[["ls80", "gold", "btc", "world"]].ffill()
    df = df.dropna(subset=["ls80", "gold", "btc", "world"]).reset_index(drop=True)

    if df.empty:
        raise ValueError("Dataset vuoto dopo il merge dei quattro CSV")

    return df, freshness


def compute_cagr(series: pd.Series, dates: pd.Series) -> float:
    if len(series) < 2:
        return 0.0
    years = (dates.iloc[-1] - dates.iloc[0]).days / 365.25
    if years <= 0:
        return 0.0
    start = float(series.iloc[0])
    end = float(series.iloc[-1])
    if start <= 0 or end <= 0:
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

    events: list[dict] = []
    in_dd = False
    start_idx: Optional[int] = None

    for i in range(1, len(series)):
        if not in_dd and dd.iloc[i] < 0:
            start_idx = i - 1
            in_dd = True

        if in_dd and dd.iloc[i] == 0:
            sub = dd.iloc[start_idx:i]
            if len(sub) > 0 and start_idx is not None:
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


def compute_portfolio_fixed(ls80: pd.Series, gold: pd.Series, btc: pd.Series, capital: float) -> pd.Series:
    shares_ls80 = (capital * WEIGHT_LS80) / float(ls80.iloc[0])
    shares_gold = (capital * WEIGHT_GOLD) / float(gold.iloc[0])
    shares_btc = (capital * WEIGHT_BTC) / float(btc.iloc[0])
    return shares_ls80 * ls80 + shares_gold * gold + shares_btc * btc


def _rebalance_holdings(total_value: float, px_ls80: float, px_gold: float, px_btc: float) -> Dict[str, float]:
    return {
        "ls80": (total_value * WEIGHT_LS80) / px_ls80,
        "gold": (total_value * WEIGHT_GOLD) / px_gold,
        "btc": (total_value * WEIGHT_BTC) / px_btc,
    }


def compute_portfolios_annual_rebalance_with_leverage(
    df: pd.DataFrame,
    capital: float,
    leverage_ratio: float = LEVERAGE_RATIO,
    lombard_rate_annual: float = LOMBARD_RATE_ANNUAL,
) -> Dict[str, pd.Series]:
    dates = df["date"].reset_index(drop=True)
    px_ls80 = df["ls80"].reset_index(drop=True)
    px_gold = df["gold"].reset_index(drop=True)
    px_btc = df["btc"].reset_index(drop=True)

    n = len(df)
    if n == 0:
        raise ValueError("Dataset vuoto")

    pigro_hold = _rebalance_holdings(
        capital,
        float(px_ls80.iloc[0]),
        float(px_gold.iloc[0]),
        float(px_btc.iloc[0]),
    )

    equity0 = capital
    debt = equity0 * leverage_ratio
    gross0 = equity0 + debt
    leva_hold = _rebalance_holdings(
        gross0,
        float(px_ls80.iloc[0]),
        float(px_gold.iloc[0]),
        float(px_btc.iloc[0]),
    )

    pigro_values = []
    leva_equity_values = []
    debt_values = []
    gross_values = []

    for i in range(n):
        gross_pigro = (
            pigro_hold["ls80"] * float(px_ls80.iloc[i])
            + pigro_hold["gold"] * float(px_gold.iloc[i])
            + pigro_hold["btc"] * float(px_btc.iloc[i])
        )

        gross_leva = (
            leva_hold["ls80"] * float(px_ls80.iloc[i])
            + leva_hold["gold"] * float(px_gold.iloc[i])
            + leva_hold["btc"] * float(px_btc.iloc[i])
        )

        if i > 0:
            days = max((dates.iloc[i] - dates.iloc[i - 1]).days, 1)
            debt *= (1.0 + lombard_rate_annual * days / DAY_COUNT)

        equity_leva = gross_leva - debt

        pigro_values.append(gross_pigro)
        leva_equity_values.append(equity_leva)
        debt_values.append(debt)
        gross_values.append(gross_leva)

        if i < n - 1 and dates.iloc[i + 1].year != dates.iloc[i].year:
            pigro_hold = _rebalance_holdings(
                gross_pigro,
                float(px_ls80.iloc[i]),
                float(px_gold.iloc[i]),
                float(px_btc.iloc[i]),
            )

            new_equity = equity_leva
            new_debt = max(new_equity * leverage_ratio, 0.0)
            new_gross = max(new_equity + new_debt, 0.0)

            leva_hold = _rebalance_holdings(
                new_gross,
                float(px_ls80.iloc[i]),
                float(px_gold.iloc[i]),
                float(px_btc.iloc[i]),
            )
            debt = new_debt

    return {
        "pigro": pd.Series(pigro_values),
        "leva_equity": pd.Series(leva_equity_values),
        "leva_debt": pd.Series(debt_values),
        "leva_gross": pd.Series(gross_values),
    }


def _openai_client():
    if not OPENAI_API_KEY:
        return None
    try:
        from openai import OpenAI
        return OpenAI(api_key=OPENAI_API_KEY)
    except Exception:
        return None


@app.get("/")
def home():
    return render_template("index.html")


@app.get("/leva")
def leva_page():
    return render_template("leva.html")


@app.get("/health")
def health():
    return jsonify({"ok": True, "time_utc": _now_iso()})


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
                "stale_days": dataset_freshness_days(df),
            }
        except Exception as e:
            return {"exists": True, "file": str(path), "error": f"{type(e).__name__}: {e}"}

    payload: Dict[str, Any] = {
        "ok": True,
        "time_utc": _now_iso(),
        "files": {
            "ls80": info(LS80_FILE),
            "gold": info(GOLD_FILE),
            "btc": info(BTC_FILE),
            "world": info(WORLD_FILE),
        },
        "env": {
            "LS80_TICKER": LS80_TICKER,
            "GOLD_TICKER": GOLD_TICKER,
            "BTC_TICKER": BTC_TICKER,
            "WORLD_TICKER": WORLD_TICKER,
            "TWELVE_DATA_API_KEY_present": bool(TWELVE_DATA_API_KEY),
            "UPDATE_TOKEN_present": bool(UPDATE_TOKEN),
            "OPENAI_API_KEY_present": bool(OPENAI_API_KEY),
            "OPENAI_MODEL": OPENAI_MODEL,
        },
    }

    try:
        merged, freshness = build_merged_dataset()
        payload["merge"] = {
            "rows_after_outer_ffill": int(len(merged)),
            "first_date": str(merged["date"].iloc[0].date()),
            "last_date": str(merged["date"].iloc[-1].date()),
            "freshness_days": freshness,
        }
    except Exception as e:
        payload["merge_error"] = f"{type(e).__name__}: {e}"

    return jsonify(payload)


@app.get("/api/compute")
def api_compute():
    try:
        capital = _safe_float(request.args.get("capital", 10000), 10000.0)
        if capital <= 0:
            return _json_error("Capitale non valido.", 400)

        df, freshness = build_merged_dataset()
        if len(df) < MIN_ROWS_REQUIRED:
            return _json_error(
                "Serie troppo corta dopo il merge tra LS80, Oro, Bitcoin e MSCI World.",
                400,
                rows=int(len(df)),
            )

        dates = df["date"].reset_index(drop=True)
        portfolio = compute_portfolio_fixed(df["ls80"], df["gold"], df["btc"], capital).reset_index(drop=True)
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

        years_period = (dates.iloc[-1] - dates.iloc[0]).days / 365.25
        warnings: list[str] = []
        for name, stale_days in freshness.items():
            if stale_days is not None and stale_days > STALE_WARNING_DAYS:
                warnings.append(f"{name.upper()} fermo da {stale_days} giorni: usato ultimo valore disponibile")

        return jsonify(
            {
                "ok": True,
                "dates": dates.dt.strftime("%Y-%m-%d").tolist(),
                "portfolio": [round(float(x), 6) for x in portfolio.tolist()],
                "world": [round(float(x), 6) for x in world_scaled.tolist()],
                "drawdown_portfolio_pct": [round(float(x), 6) for x in dd_port.tolist()],
                "drawdown_world_pct": [round(float(x), 6) for x in dd_world.tolist()],
                "composition": {"ls80": WEIGHT_LS80, "gold": WEIGHT_GOLD, "btc": WEIGHT_BTC},
                "freshness_days": freshness,
                "warnings": warnings,
                "metrics": {
                    "final_portfolio": float(portfolio.iloc[-1]),
                    "final_world": float(world_scaled.iloc[-1]),
                    "final_years": float(years_period),
                    "cagr_portfolio": float(cagr_port),
                    "max_dd_portfolio": float(maxdd_port),
                    "doubling_years_portfolio": float(dbl_years) if dbl_years is not None else None,
                    "cagr_world": float(cagr_world),
                    "max_dd_world": float(maxdd_world),
                    "worst_episodes_portfolio": worst_port,
                    "worst_episodes_world": worst_world,
                },
            }
        )
    except Exception as e:
        return jsonify({"ok": False, "error": f"Errore compute: {type(e).__name__}: {e}"})


@app.get("/api/compute_leva")
def api_compute_leva():
    try:
        capital = _safe_float(request.args.get("capital", DEFAULT_LEVERAGE_CAPITAL), DEFAULT_LEVERAGE_CAPITAL)
        if capital <= 0:
            return _json_error("Capitale non valido.", 400)

        df, freshness = build_merged_dataset()
        if len(df) < MIN_ROWS_REQUIRED:
            return _json_error(
                "Serie troppo corta dopo il merge dei CSV.",
                400,
                rows=int(len(df)),
            )

        dates = df["date"].reset_index(drop=True)

        series = compute_portfolios_annual_rebalance_with_leverage(
            df=df,
            capital=capital,
            leverage_ratio=LEVERAGE_RATIO,
            lombard_rate_annual=LOMBARD_RATE_ANNUAL,
        )

        pigro = series["pigro"].reset_index(drop=True)
        leva_equity = series["leva_equity"].reset_index(drop=True)
        leva_debt = series["leva_debt"].reset_index(drop=True)
        leva_gross = series["leva_gross"].reset_index(drop=True)

        dd_pigro = compute_drawdown_series_pct(pigro)
        dd_leva = compute_drawdown_series_pct(leva_equity)

        cagr_pigro = compute_cagr(pigro, dates)
        cagr_leva = compute_cagr(leva_equity, dates)

        maxdd_pigro = compute_max_dd(pigro)
        maxdd_leva = compute_max_dd(leva_equity)

        dbl_pigro = doubling_years(cagr_pigro)
        dbl_leva = doubling_years(cagr_leva)

        worst_pigro = worst_drawdowns(pigro, dates, n=3)
        worst_leva = worst_drawdowns(leva_equity, dates, n=3)

        years_period = (dates.iloc[-1] - dates.iloc[0]).days / 365.25

        warnings: list[str] = []
        for name, stale_days in freshness.items():
            if stale_days is not None and stale_days > STALE_WARNING_DAYS:
                warnings.append(f"{name.upper()} fermo da {stale_days} giorni: usato ultimo valore disponibile")

        return jsonify(
            {
                "ok": True,
                "dates": dates.dt.strftime("%Y-%m-%d").tolist(),
                "pigro": [round(float(x), 6) for x in pigro.tolist()],
                "leva": [round(float(x), 6) for x in leva_equity.tolist()],
                "drawdown_pigro_pct": [round(float(x), 6) for x in dd_pigro.tolist()],
                "drawdown_leva_pct": [round(float(x), 6) for x in dd_leva.tolist()],
                "debt_series": [round(float(x), 6) for x in leva_debt.tolist()],
                "gross_series": [round(float(x), 6) for x in leva_gross.tolist()],
                "composition": {"ls80": WEIGHT_LS80, "gold": WEIGHT_GOLD, "btc": WEIGHT_BTC},
                "leverage": {
                    "ratio": LEVERAGE_RATIO,
                    "lombard_rate_annual": LOMBARD_RATE_ANNUAL,
                    "rebalance": "annuale",
                },
                "freshness_days": freshness,
                "warnings": warnings,
                "metrics": {
                    "initial_capital": float(capital),
                    "final_years": float(years_period),
                    "final_pigro": float(pigro.iloc[-1]),
                    "final_leva": float(leva_equity.iloc[-1]),
                    "cagr_pigro": float(cagr_pigro),
                    "cagr_leva": float(cagr_leva),
                    "max_dd_pigro": float(maxdd_pigro),
                    "max_dd_leva": float(maxdd_leva),
                    "doubling_years_pigro": float(dbl_pigro) if dbl_pigro is not None else None,
                    "doubling_years_leva": float(dbl_leva) if dbl_leva is not None else None,
                    "initial_debt": float(capital * LEVERAGE_RATIO),
                    "final_debt": float(leva_debt.iloc[-1]),
                    "final_gross_assets": float(leva_gross.iloc[-1]),
                    "worst_episodes_pigro": worst_pigro,
                    "worst_episodes_leva": worst_leva,
                },
            }
        )
    except Exception as e:
        return jsonify({"ok": False, "error": f"Errore compute_leva: {type(e).__name__}: {e}"})


@app.post("/api/ask")
def api_ask():
    try:
        data = request.get_json(silent=True) or {}
        question = (data.get("question") or "").strip()
        if not question:
            return _json_error("Scrivi una domanda.", 400)

        client = _openai_client()
        if client is None:
            return jsonify({"ok": True, "answer": "Assistente non configurato: manca OPENAI_API_KEY su Render."})

        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Rispondi in italiano, in modo semplice e pratico. "
                        "Contesto: sito Metodo Pigro variante 80% LS80, 15% Oro, 5% Bitcoin. "
                        "Informazione generale, non consulenza personalizzata."
                    ),
                },
                {"role": "user", "content": question},
            ],
            temperature=0.2,
        )

        answer = ""
        if resp and resp.choices and resp.choices[0].message and resp.choices[0].message.content:
            answer = resp.choices[0].message.content or ""
        return jsonify({"ok": True, "answer": answer.strip() or "Nessuna risposta disponibile."})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


@app.get("/api/update_data")
def api_update_data():
    err = _require_token()
    if err is not None:
        return err

    try:
        res_ls = update_one_asset(LS80_FILE, LS80_TICKER)
        res_gd = update_one_asset(GOLD_FILE, GOLD_TICKER)
        res_bt = update_one_asset(BTC_FILE, BTC_TICKER)
        res_wd = update_one_asset(WORLD_FILE, WORLD_TICKER)

        usable_all = all([
            bool(res_ls.get("usable")),
            bool(res_gd.get("usable")),
            bool(res_bt.get("usable")),
            bool(res_wd.get("usable")),
        ])

        return jsonify(
            {
                "ok": usable_all,
                "ls80": res_ls,
                "gold": res_gd,
                "btc": res_bt,
                "world": res_wd,
                "time_utc": _now_iso(),
            }
        )
    except Exception as e:
        return _json_error(f"{type(e).__name__}: {e}", 500)


@app.get("/api/force_update")
def api_force_update():
    return api_update_data()


@app.get("/faxsimile_execution_only.pdf")
def faxsimile_execution_only():
    static_pdf = BASE_DIR / "static" / "faxsimile_execution_only.pdf"
    if static_pdf.exists():
        return send_file(static_pdf, mimetype="application/pdf")
    return _json_error("PDF non trovato.", 404)


@app.get("/api/pdf")
def api_pdf():
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas

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
            download_name="gloob_pigro_8155.pdf",
        )
    except Exception as e:
        return _json_error(f"Errore PDF: {type(e).__name__}: {e}", 500)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=True)
