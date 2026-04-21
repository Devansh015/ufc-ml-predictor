from flask import Flask, request, jsonify, send_from_directory, redirect
import joblib
import pandas as pd
import numpy as np
import re
from pathlib import Path
from collections import defaultdict
import os

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "features" / "fight_features_alpha.csv"
DETAILS_PATH = ROOT / "data" / "raw" / "fight_details.csv"
DEFAULT_MODEL = ROOT / "models" / "lgbm_symmetric.pkl"
PUBLIC_DIR = ROOT / "public"

app = Flask(__name__)

# load once
df = pd.read_csv(DATA_PATH)
model_pkg = joblib.load(DEFAULT_MODEL)
clf = model_pkg.get("model") or model_pkg
imputer = model_pkg.get("imputer")
saved_features = model_pkg.get("features")

# ── Weight-class mapping ──────────────────────────────────────────
# Ordered longest-first so "Light Heavyweight" matches before "Heavyweight", etc.
STANDARD_CLASSES = [
    "Women's Featherweight",
    "Women's Bantamweight",
    "Women's Flyweight",
    "Women's Strawweight",
    "Light Heavyweight",
    "Heavyweight",
    "Middleweight",
    "Welterweight",
    "Lightweight",
    "Featherweight",
    "Bantamweight",
    "Flyweight",
    "Strawweight",
]

def _extract_weight_class(subtitle: str) -> str | None:
    """Map a fight_subtitle like 'UFC Interim Lightweight Title Bout' → 'Lightweight'."""
    if not isinstance(subtitle, str):
        return None
    for wc in STANDARD_CLASSES:
        if wc.lower() in subtitle.lower():
            return wc
    return None

details_df = pd.read_csv(DETAILS_PATH)
details_df["weight_class"] = details_df["fight_subtitle"].apply(_extract_weight_class)
# merge weight class onto the features dataframe via fight_url
_wc_map = details_df.dropna(subset=["weight_class"]).drop_duplicates("fight_url")[["fight_url", "weight_class"]]
df = df.merge(_wc_map, on="fight_url", how="left")

# Build fighter → set of weight classes
_fighter_classes: dict[str, set[str]] = defaultdict(set)
for _, row in df.dropna(subset=["weight_class"]).iterrows():
    wc = row["weight_class"]
    rf = str(row["red_fighter"]).strip()
    bf = str(row["blue_fighter"]).strip()
    if rf:
        _fighter_classes[rf].add(wc)
    if bf:
        _fighter_classes[bf].add(wc)


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


# Build unique fighters list at startup
_all_fighters = sorted(
    set(
        df["red_fighter"].dropna().str.strip().tolist()
        + df["blue_fighter"].dropna().str.strip().tolist()
    ),
    key=str.lower,
)

# Weight classes that actually have fighters (sorted by weight, heaviest first for men, then women)
_WEIGHT_ORDER = [
    "Heavyweight", "Light Heavyweight", "Middleweight", "Welterweight",
    "Lightweight", "Featherweight", "Bantamweight", "Flyweight", "Strawweight",
    "Women's Featherweight", "Women's Bantamweight", "Women's Flyweight", "Women's Strawweight",
]
_active_classes = [wc for wc in _WEIGHT_ORDER if any(wc in classes for classes in _fighter_classes.values())]

# Pre-compute fighters per weight class
_fighters_by_class: dict[str, list[str]] = {}
for wc in _active_classes:
    names = sorted(
        [name for name, classes in _fighter_classes.items() if wc in classes],
        key=str.lower,
    )
    _fighters_by_class[wc] = names


@app.route("/weightclasses", methods=["GET"])
def weightclasses():
    """Return ordered list of weight classes."""
    return jsonify(_active_classes)


@app.route("/fighters", methods=["GET"])
def fighters():
    """Return list of fighter names, optionally filtered by weight class."""
    wc = request.args.get("weight_class", "").strip()
    if wc and wc in _fighters_by_class:
        return jsonify(_fighters_by_class[wc])
    return jsonify(_all_fighters)


@app.route("/", methods=["GET"])
def index():
    public_index = PUBLIC_DIR / "index.html"
    if public_index.exists():
        return redirect("/index.html", code=307)

    # Fallback for local development before the public copy exists.
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

        # prepare stat breakdown
        stat_map = [
            ("striking_accuracy", "sig_acc"),
            ("strikes_per_min", "sig_landed_per_min"),
            ("takedown_acc", "td_acc"),
            ("takedowns_per_15", "td_landed_per_15"),
            ("striking_def", "sig_acc"),
            ("takedown_def", "td_acc"),
            ("reach", "reach"),
        ]

        def map_red_to_inputs(red_name_is_a: bool):
            return (body.get("fighter_a"), body.get("fighter_b")) if red_name_is_a else (body.get("fighter_b"), body.get("fighter_a"))

        # if swapped==False then match_row red==fighter_a
        red_is_a = not swapped
        fighter_a_name, fighter_b_name = map_red_to_inputs(red_is_a)

        stats = {}
        row = X.iloc[0]
        for label, feat in stat_map:
            red_col = f"red_{feat}"
            blue_col = f"blue_{feat}"
            red_val = float(row[red_col]) if red_col in row.index and pd.notna(row[red_col]) else None
            blue_val = float(row[blue_col]) if blue_col in row.index and pd.notna(row[blue_col]) else None

            # for defensive metrics, invert opponent accuracy when appropriate
            if label == "striking_def" and red_val is None and blue_val is None:
                # try to infer from opponent's accuracy
                pass

            # map to caller ordering
            if red_is_a:
                a_val, b_val = red_val, blue_val
            else:
                a_val, b_val = blue_val, red_val

            better = None
            try:
                if a_val is not None and b_val is not None:
                    if a_val > b_val:
                        better = fighter_a_name
                    elif b_val > a_val:
                        better = fighter_b_name
            except Exception:
                better = None

            stats[label] = {"fighter_a": a_val, "fighter_b": b_val, "better": better}

        winner = body.get("fighter_a") if proba_a > proba_b else body.get("fighter_b")
        return jsonify({
            "winner": winner,
            "p_a": float(proba_a),
            "p_b": float(proba_b),
            "p_a_pct": round(float(proba_a) * 100.0, 1),
            "p_b_pct": round(float(proba_b) * 100.0, 1),
            "stats": stats,
        })

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
    # build same stat breakdown for synthetic row
    stat_map = [
        ("striking_accuracy", "sig_acc"),
        ("strikes_per_min", "sig_landed_per_min"),
        ("takedown_acc", "td_acc"),
        ("takedowns_per_15", "td_landed_per_15"),
        ("striking_def", "sig_acc"),
        ("takedown_def", "td_acc"),
        ("reach", "reach"),
    ]

    # red corresponds to a_can
    red_is_a = (a == a_can)
    fighter_a_name = body.get("fighter_a")
    fighter_b_name = body.get("fighter_b")

    stats = {}
    row = Xsyn.iloc[0]
    for label, feat in stat_map:
        red_col = f"red_{feat}"
        blue_col = f"blue_{feat}"
        red_val = float(row[red_col]) if red_col in row.index and pd.notna(row[red_col]) else None
        blue_val = float(row[blue_col]) if blue_col in row.index and pd.notna(row[blue_col]) else None

        if red_is_a:
            a_val, b_val = red_val, blue_val
        else:
            a_val, b_val = blue_val, red_val

        better = None
        if a_val is not None and b_val is not None:
            if a_val > b_val:
                better = fighter_a_name
            elif b_val > a_val:
                better = fighter_b_name

        stats[label] = {"fighter_a": a_val, "fighter_b": b_val, "better": better}

    winner = body.get("fighter_a") if proba_a > proba_b else body.get("fighter_b")
    return jsonify({
        "winner": winner,
        "p_a": float(proba_a),
        "p_b": float(proba_b),
        "p_a_pct": round(float(proba_a) * 100.0, 1),
        "p_b_pct": round(float(proba_b) * 100.0, 1),
        "stats": stats,
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="127.0.0.1", port=port)
