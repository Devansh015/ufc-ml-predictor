import argparse
import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             f1_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import train_test_split


# ---------- Leakage-safe column blacklist ----------
_LEAK_COLS = {"winner", "blue_win"}


def load_and_prep(path, target_col="red_win"):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in {path}.")
    return df


def _feature_target_split(df, target_col="red_win"):
    """Return (X_numeric, y) after dropping non-feature and leaked columns."""
    y = df[target_col].astype(int)
    drop_cols = [c for c in
                 {"fight_url", "red_fighter", "blue_fighter"} | _LEAK_COLS
                 if c in df.columns]
    X = df.drop(columns=drop_cols + [target_col], errors="ignore")
    X = X.select_dtypes(include=[np.number])
    # drop all-NaN columns
    X = X.loc[:, ~X.isna().all()]
    return X, y


def symmetry_augment(df):
    """Create mirrored rows with red/blue swapped and correct labels."""
    swapped = []
    for _, r in df.iterrows():
        nr = {}
        for col, val in r.items():
            if col == "red_fighter":
                nr["red_fighter"] = r.get("blue_fighter")
            elif col == "blue_fighter":
                nr["blue_fighter"] = r.get("red_fighter")
            elif col.startswith("red_") and col != "red_win":
                nr["blue_" + col[4:]] = val
            elif col.startswith("blue_"):
                nr["red_" + col[5:]] = val
            elif col.startswith("delta_"):
                nr[col] = -val if pd.notna(val) else val
            elif col == "red_win":
                nr["red_win"] = 1 - int(val)
            else:
                nr[col] = val
        swapped.append(nr)
    return pd.concat([df, pd.DataFrame(swapped)], ignore_index=True, sort=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/features/fight_features_alpha.csv")
    p.add_argument("--params-file", default=None, help="JSON file with model params (optional)")
    p.add_argument("--model-out", default="models/lgbm_symmetric.pkl")
    p.add_argument("--metrics-out", default="models/metrics_symmetric.json")
    p.add_argument("--feature-imp-out", default="models/feature_importances_symmetric.csv")
    p.add_argument("--test-size", type=float, default=0.15)
    p.add_argument("--val-size", type=float, default=0.15)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--n-estimators", type=int, default=2000)
    p.add_argument("--early-stopping", type=int, default=200)
    args = p.parse_args()

    df = load_and_prep(args.input)

    # ---- Fight-URL-grouped 3-way split (train/val/test) ----
    # Prevents augmented-pair leakage and uses *non-augmented* val for early stopping
    fight_urls = df["fight_url"].unique()
    rng = np.random.RandomState(args.random_state)
    rng.shuffle(fight_urls)
    n = len(fight_urls)
    train_end = int(n * (1 - args.test_size - args.val_size))
    val_end = int(n * (1 - args.test_size))

    train_urls = set(fight_urls[:train_end])
    val_urls = set(fight_urls[train_end:val_end])
    test_urls = set(fight_urls[val_end:])

    raw_train = df[df["fight_url"].isin(train_urls)].copy()
    raw_val = df[df["fight_url"].isin(val_urls)].copy()
    raw_test = df[df["fight_url"].isin(test_urls)].copy()

    # Symmetry-augment ONLY the training set
    train_df = symmetry_augment(raw_train)

    X_train, y_train = _feature_target_split(train_df)
    X_val, y_val = _feature_target_split(raw_val)      # NOT augmented
    X_test, y_test = _feature_target_split(raw_test)    # NOT augmented

    # Align columns across all three sets
    common_cols = sorted(set(X_train.columns) & set(X_val.columns) & set(X_test.columns))
    X_train = X_train[common_cols]
    X_val = X_val[common_cols]
    X_test = X_test[common_cols]

    imputer = SimpleImputer(strategy="median")
    X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=common_cols)
    X_val = pd.DataFrame(imputer.transform(X_val), columns=common_cols)
    X_test = pd.DataFrame(imputer.transform(X_test), columns=common_cols)

    print(f"Train: {len(X_train)} (augmented)  |  Val: {len(X_val)}  |  Test: {len(X_test)}")
    print(f"Features: {len(common_cols)}")

    # allow loading tuned params
    model_params = {}
    if args.params_file:
        import json as _json

        with open(args.params_file, "r") as _fh:
            loaded = _json.load(_fh)
        # support study file format {"best_params": {...}}
        if "best_params" in loaded:
            model_params = loaded["best_params"]
        else:
            model_params = loaded

    # ensure n_estimators and random_state
    model_params.setdefault("n_estimators", args.n_estimators)
    model_params.setdefault("learning_rate", 0.01)
    model_params.setdefault("num_leaves", 16)
    model_params.setdefault("min_child_samples", 20)
    model_params.setdefault("subsample", 0.8)
    model_params.setdefault("colsample_bytree", 0.8)
    model_params.setdefault("reg_lambda", 1.0)
    model_params.setdefault("reg_alpha", 0.1)
    model_params.setdefault("random_state", args.random_state)
    model_params.setdefault("n_jobs", -1)

    clf = lgb.LGBMClassifier(**model_params, verbose=-1)

    # fit with early stopping on non-augmented validation set
    callbacks = []
    if args.early_stopping and int(args.early_stopping) > 0:
        callbacks.append(lgb.early_stopping(stopping_rounds=int(args.early_stopping)))
    callbacks.append(lgb.log_evaluation(period=0))

    clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric="auc", callbacks=callbacks)

    # predictions
    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    y_pred = clf.predict(X_test)

    metrics = {
        "roc_auc": float(roc_auc_score(y_test, y_pred_proba)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "best_iteration": int(getattr(clf, "best_iteration_") or -1),
    }

    # outputs
    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": clf, "imputer": imputer, "features": list(common_cols)}, args.model_out)

    Path(args.metrics_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.metrics_out, "w") as fh:
        json.dump(metrics, fh, indent=2)

    # feature importances
    fi = pd.DataFrame({"feature": common_cols, "importance": clf.feature_importances_})
    fi = fi.sort_values("importance", ascending=False)
    fi.to_csv(args.feature_imp_out, index=False)

    print("Training complete.")
    print(f"Model saved to: {args.model_out}")
    print(f"Metrics saved to: {args.metrics_out}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
