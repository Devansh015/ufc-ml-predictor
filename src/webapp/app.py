from flask import Flask, render_template, request, jsonify, send_from_directory
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "features" / "fight_features_alpha.csv"
DEFAULT_MODEL = ROOT / "models" / "lgbm_symmetric.pkl"

app = Flask(__name__)

# load once
df = pd.read_csv(DATA_PATH)
model_pkg = joblib.load(DEFAULT_MODEL)
clf = model_pkg.get("model") or model_pkg
imputer = model_pkg.get("imputer")
saved_features = model_pkg.get("features")


def normalize(name: str) -> str:
    return name.strip().lower()


def find_direct_matchup(df: pd.DataFrame, a: str, b: str):
    mask1 = (df["red_fighter"].str.strip().str.lower() == a) & (
        df["blue_fighter"].str.strip().str.lower() == b
    )
    mask2 = (df["red_fighter"].str.strip().str.lower() == b) & (
        df["blue_fighter"].str.strip().str.lower() == a
    )
    matches = df[mask1 | mask2]
    if matches.empty:
        return None, None
    row = matches.iloc[-1]
    swapped = False
    if (row["red_fighter"].strip().lower() == b) and (
        row["blue_fighter"].strip().lower() == a
    ):
        swapped = True
    return row, swapped


def build_synthetic_row(df: pd.DataFrame, a: str, b: str, feature_cols, use_recency: bool, decay: float, last_n: int):
    red_cols = [c for c in feature_cols if c.startswith("red_")]
    base = [c[len("red_") :] for c in red_cols]

    def fighter_stats(name: str):
        vals_by_feat = {bname: [] for bname in base}
        for idx, row in df.iterrows():
            rf = str(row.get("red_fighter", "")).strip().lower()
            bf = str(row.get("blue_fighter", "")).strip().lower()
            if rf == name:
                for bname in base:
                    col = "red_" + bname
                    if col in df.columns:
                        vals_by_feat[bname].append(row.get(col, np.nan))
            if bf == name:
                for bname in base:
                    col = "blue_" + bname
                    if col in df.columns:
                        vals_by_feat[bname].append(row.get(col, np.nan))

        stats = {}
        for bname, vals in vals_by_feat.items():
            clean = [v for v in vals if pd.notna(v)]
            if len(clean) == 0:
                stats[bname] = 0.0
                continue
            recent = clean[-last_n:][::-1]
            if use_recency:
                weights = np.array([decay ** k for k in range(len(recent))], dtype=float)
                if weights.sum() > 0:
                    wavg = float(np.sum(np.array(recent) * weights) / weights.sum())
                else:
                    wavg = float(np.mean(recent))
                stats[bname] = wavg
            else:
                stats[bname] = float(np.mean(recent))
        return stats

    stats_a = fighter_stats(a)
    stats_b = fighter_stats(b)

    row = {}
    for bname in base:
        row[f"red_{bname}"] = stats_a.get(bname, 0.0)
        row[f"blue_{bname}"] = stats_b.get(bname, 0.0)
        if f"delta_{bname}" in feature_cols:
            row[f"delta_{bname}"] = row[f"red_{bname}"] - row[f"blue_{bname}"]

    for c in feature_cols:
        if c not in row:
            row[c] = 0.0

    return pd.DataFrame([row])


@app.route("/", methods=["GET"])
def index():
    # Serve React single-file UI
    react_dir = ROOT / "src" / "webapp" / "react"
    idx = react_dir / "index.html"
    if idx.exists():
        return send_from_directory(str(react_dir), "index.html")
    return "React frontend not found", 404


@app.route('/react/<path:filename>')
def react_static(filename):
    react_dir = ROOT / "src" / "webapp" / "react"
    return send_from_directory(str(react_dir), filename)


@app.route("/predict", methods=["POST"])
def predict():
    body = request.json or {}
    a = normalize(body.get("fighter_a", ""))
    b = normalize(body.get("fighter_b", ""))
    recency = bool(body.get("recency", True))
    decay = float(body.get("decay", 0.9))
    last_n = int(body.get("last_n", 5))

    match_row, swapped = find_direct_matchup(df, a, b)
    if match_row is not None:
        X = match_row.to_frame().T
        X = X.reindex(columns=saved_features)
        X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        if imputer is not None:
            X = pd.DataFrame(imputer.transform(X), columns=X.columns)
        proba_red = clf.predict_proba(X)[:, 1][0]
        proba_a = 1.0 - proba_red if swapped else proba_red
        proba_b = 1.0 - proba_a
        winner = body.get("fighter_a") if proba_a > proba_b else body.get("fighter_b")
        return jsonify({"winner": winner})

    # build synthetic features using a canonical ordering so predictions are order-invariant
    a_can, b_can = (a, b) if a <= b else (b, a)
    Xsyn = build_synthetic_row(df, a_can, b_can, saved_features, use_recency=recency, decay=decay, last_n=last_n)
    Xsyn = Xsyn.reindex(columns=saved_features)
    Xsyn = Xsyn.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    if imputer is not None:
        Xsyn = pd.DataFrame(imputer.transform(Xsyn), columns=Xsyn.columns)
    proba_red = clf.predict_proba(Xsyn)[:, 1][0]
    # map probability back to the caller's fighter order
    if a == a_can:
        proba_a = proba_red
        proba_b = 1.0 - proba_red
    else:
        proba_a = 1.0 - proba_red
        proba_b = proba_red
    winner = body.get("fighter_a") if proba_a > proba_b else body.get("fighter_b")
    return jsonify({"winner": winner})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="127.0.0.1", port=port)
