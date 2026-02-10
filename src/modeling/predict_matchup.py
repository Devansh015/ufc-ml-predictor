#!/usr/bin/env python3
"""Compare two fighters and predict which has higher win probability.

Behavior:
- If an exact historical matchup row exists in `data/features/fight_features_alpha.csv`, use it.
- Otherwise build a synthetic matchup by aggregating recent per-fighter stats and computing deltas.

Output: prints probabilities for fighter A and B and recommended winner.
"""
import argparse
import json
from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd


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
    # choose most recent by row order (last occurrence)
    row = matches.iloc[-1]
    swapped = False
    if (row["red_fighter"].strip().lower() == b) and (
        row["blue_fighter"].strip().lower() == a
    ):
        swapped = True
    return row, swapped


def build_synthetic_row(df: pd.DataFrame, a: str, b: str, feature_cols: List[str], use_recency: bool = False, decay: float = 0.8, last_n: int = 10):
    # Determine base features (strip red_/blue_)
    red_cols = [c for c in feature_cols if c.startswith("red_")]
    base = [c[len("red_") :] for c in red_cols]

    def fighter_stats(name: str):
        # collect historical values in chronological order (assumes df is from past->present)
        vals_by_feat = {bname: [] for bname in base}
        for idx, row in df.iterrows():
            # check red appearance
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
            # drop NaN and keep in chronological order
            clean = [v for v in vals if pd.notna(v)]
            if len(clean) == 0:
                stats[bname] = 0.0
                continue
            # take last_n most recent
            recent = clean[-last_n:][::-1]  # most recent first
            if use_recency:
                # weights: decay**k with k=0 for most recent
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

    # construct row with red_* for A and blue_* for B
    row = {}
    for bname in base:
        row[f"red_{bname}"] = stats_a.get(bname, 0.0)
        row[f"blue_{bname}"] = stats_b.get(bname, 0.0)
        # delta features often exist as 'delta_<name>' or explicit columns
        if f"delta_{bname}" in feature_cols:
            row[f"delta_{bname}"] = row[f"red_{bname}"] - row[f"blue_{bname}"]

    # ensure all requested feature_cols present
    for c in feature_cols:
        if c not in row:
            row[c] = 0.0

    return pd.DataFrame([row])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fighter-a", required=True)
    p.add_argument("--fighter-b", required=True)
    p.add_argument("--model", default="models/lgbm_tuned_final.pkl")
    p.add_argument("--data", default="data/features/fight_features_alpha.csv")
    p.add_argument("--as-red", action="store_true", help="Treat fighter-a as red corner")
    p.add_argument("--recency", action="store_true", help="Use recency-weighted aggregates (decay weighting)")
    p.add_argument("--decay", type=float, default=0.8, help="Recency decay factor (0<decay<=1), higher keeps recent fights more important")
    p.add_argument("--last-n", type=int, default=10, help="Use up to this many most recent fights when aggregating per-fighter stats")
    p.add_argument("--force-synthetic", action="store_true", help="Ignore direct historical matchup and force synthetic aggregates")
    args = p.parse_args()

    df = pd.read_csv(args.data)
    # keep original columns

    a = normalize(args.fighter_a)
    b = normalize(args.fighter_b)

    model_pkg = joblib.load(args.model)
    clf = model_pkg.get("model") or model_pkg
    imputer = model_pkg.get("imputer")
    saved_features = model_pkg.get("features")

    # try direct matchup (unless user requests synthetic)
    match_row, swapped = find_direct_matchup(df, a, b)
    if match_row is not None and not args.force_synthetic:
        # create X from saved_features using the match_row
        X = match_row.to_frame().T
        # ensure all numeric feature columns present
        X = X.reindex(columns=saved_features)
        # coerce to numeric (non-numeric -> NaN) so imputer can handle missing values
        X = X.apply(pd.to_numeric, errors='coerce')
        X = X.fillna(0.0)
        if imputer is not None:
            X = pd.DataFrame(imputer.transform(X), columns=X.columns)
        proba_red = clf.predict_proba(X)[:, 1][0]
        if swapped:
            # fighter A was blue in this historical row
            proba_a = 1.0 - proba_red
        else:
            proba_a = proba_red
        proba_b = 1.0 - proba_a
        winner = args.fighter_a if proba_a >= proba_b else args.fighter_b
        out = {"fighter_a": args.fighter_a, "fighter_b": args.fighter_b, "p_a": proba_a, "p_b": proba_b, "winner": winner, "method": "historical_row"}
        print(json.dumps(out, indent=2))
        return

    # build synthetic matchup
    print("Building synthetic matchup from per-fighter aggregates.")
    Xsyn = build_synthetic_row(df, a, b, saved_features, use_recency=args.recency, decay=args.decay, last_n=args.last_n)
    # keep only numeric and saved_features order
    Xsyn = Xsyn.reindex(columns=saved_features)
    Xsyn = Xsyn.apply(pd.to_numeric, errors='coerce').fillna(0.0)
    if imputer is not None:
        Xsyn = pd.DataFrame(imputer.transform(Xsyn), columns=Xsyn.columns)

    proba_red = clf.predict_proba(Xsyn)[:, 1][0]
    # map to fighter-a/fighter-b depending on --as-red
    if args.as_red:
        proba_a = proba_red
    else:
        # default: assume fighter-a -> red
        proba_a = proba_red
    proba_b = 1.0 - proba_a
    winner = args.fighter_a if proba_a >= proba_b else args.fighter_b
    out = {"fighter_a": args.fighter_a, "fighter_b": args.fighter_b, "p_a": proba_a, "p_b": proba_b, "winner": winner, "method": "synthetic_aggregates", "recency_used": args.recency, "decay": args.decay, "last_n": args.last_n}
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
