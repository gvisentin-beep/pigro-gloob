from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

ASSET_FILES: Dict[str, str] = {
    "ls80": "ls80.csv",
    "gold": "gold.csv",
}

DEFAULT_W_LS80 = 0.80
DEFAULT_W_GOLD = 0.20


def _smart_read_csv(fp: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(fp, sep=";")
    except Exception:
        df = pd.read_csv(fp)

    if len(df.columns) == 1 and (";" in str(df.columns[0])):
        df = pd.read_csv(fp, sep=";", header=0)

    df.columns = [str(c).strip() for c in df.columns]

    if "Date" in df.columns and "Close" in df.columns:
        df = df.rename(columns={"Date": "date", "Close": "price"})
    elif "date" in df.columns and "price" in df.columns:
        pass
    else:
        candidates_date = [c for c in df.columns if c.lower() in ("date", "data")]
        candidates_price = [c for c in df.columns if c.lower() in ("close", "price", "prezzo")]
        if candidates_date and candidates_price:
            df = df.rename(columns={candidates_date[0]: "date", candidates_price[0]: "price"})
        else:
            raise ValueError(
                f"{fp.name}: formato colonne non riconosciuto. "
                "Attesi: Date/Close (sep ';') oppure date/price."
            )

    df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=True)
    df["price"] = pd.to_numeric(df["price"], errors="coerce")

    df = df.dropna(subset=["date", "price"]).sort_values("date")
    df = df[~df["date"].duplicated(keep="last")]
    return df[["date", "price"]]


def _read_asset_series(asset_key: str) -> pd.Series:
    fp = DATA_DIR / ASSET_FILES[asset_key]
    if not fp.exists():
        raise FileNotFoundError(f"File mancante: {fp}")
    df = _smart_read_csv(fp)
    return pd.Series(df["price"].values, index=df["date"], name=asset_key).astype(float)


def _align_prices() -> pd.DataFrame:
    ls80 = _read_asset_series("ls80")
    gold = _read_asset_series("gold")
    df = pd.concat([ls80, gold], axis=1).dropna()
    if len(df) < 50:
        raise ValueError("Dati insufficienti dopo allineamento (troppe date mancanti).")
    return df


def _calc_drawdown(nav: pd.Series) -> float:
    peak = nav.cummax()
    dd = nav / peak - 1.0
    return float(dd.min())


def _calc_cagr(nav: pd.Series) -> float:
    nav = nav.dropna()
    if len(nav) < 2:
        return 0.0
    days = (nav.index[-1] - nav.index[0]).days
    years = days / 365.25 if days > 0 else 0.0
    if years <= 0:
        return 0.0
    return float((nav.iloc[-1] / nav.iloc[0]) ** (1 / years) - 1)


def _portfolio_nav(prices: pd.DataFrame, w_ls80: float, w_gold: float) -> pd.Series:
    norm = prices / prices.iloc[0] * 100.0
    rets = norm.pct_change().fillna(0.0)

    nav = pd.Series(index=rets.index, dtype=float)
    nav.iloc[0] = 100.0

    v_ls80 = nav.iloc[0] * w_ls80
    v_gold = nav.iloc[0] * w_gold

    for i in range(1, len(rets)):
        v_ls80 *= (1.0 + rets["ls80"].iloc[i])
        v_gold *= (1.0 + rets["gold"].iloc[i])
        nav.iloc[i] = v_ls80 + v_gold

        # rebalance a fine anno (ultimo giorno disponibile dell'anno)
        if i < len(rets) - 1 and rets.index[i + 1].year != rets.index[i].year:
            total = nav.iloc[i]
            v_ls80 = total * w_ls80
            v_gold = total * w_gold

    nav.name = "portfolio"
    return nav


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/lettera-execution-only")
def lettera_execution_only():
    return render_template("lettera_execution_only.html")


@app.route("/api/compute")
def api_compute():
    w_ls80 = float(request.args.get("w_ls80", DEFAULT_W_LS80))
    w_gold = float(request.args.get("w_gold", DEFAULT_W_GOLD))
    initial = float(request.args.get("initial", 10000))

    if w_ls80 < 0 or w_gold < 0:
        return jsonify({"error": "I pesi non possono essere negativi."}), 400

    s = w_ls80 + w_gold
    if s <= 0:
        return jsonify({"error": "Somma pesi deve essere > 0."}), 400

    w_ls80 /= s
    w_gold /= s

    try:
        prices = _align_prices()
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    nav_port = _portfolio_nav(prices, w_ls80, w_gold)
    nav_ls80 = (prices["ls80"] / prices["ls80"].iloc[0] * 100.0).copy()
    nav_ls80.name = "ls80_only"

    cagr_port = _calc_cagr(nav_port)
    mdd_port = _calc_drawdown(nav_port)
    cagr_ls80 = _calc_cagr(nav_ls80)
    mdd_ls80 = _calc_drawdown(nav_ls80)

    implied_equity = 0.80 * w_ls80
    implied_bonds = 0.20 * w_ls80
    implied_gold = w_gold

    eur_port = nav_port / 100.0 * initial
    eur_ls80 = nav_ls80 / 100.0 * initial

    out = pd.DataFrame(
        {
            "date": nav_port.index.strftime("%Y-%m-%d"),
            "eur_port": eur_port.values.round(2),
            "eur_ls80": eur_ls80.values.round(2),
        }
    )

    return jsonify(
        {
            "weights": {"ls80": w_ls80, "gold": w_gold},
            "initial": initial,
            "implied": {"equity": implied_equity, "bonds": implied_bonds, "gold": implied_gold},
            "stats": {
                "start": str(nav_port.index[0].date()),
                "end": str(nav_port.index[-1].date()),
                "cagr": cagr_port,
                "max_drawdown": mdd_port,
                "cagr_ls80": cagr_ls80,
                "max_drawdown_ls80": mdd_ls80,
            },
            "series": out.to_dict(orient="records"),
        }
    )


if __name__ == "__main__":
    app.run(debug=True)
