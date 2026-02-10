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


def load_and_prep(path, target_col="red_win"):
    df = pd.read_csv(path)
    # normalize column names
    df.columns = df.columns.str.strip()
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in {path}.")
    y = df[target_col].astype(int)
    # drop obvious non-feature columns
    drop_cols = [c for c in ["fight_url", "red_fighter", "blue_fighter"] if c in df.columns]
    X = df.drop(columns=drop_cols + [target_col], errors="ignore")
    # keep numeric columns
    X = X.select_dtypes(include=[np.number])
    return X, y


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/features/fight_features_alpha.csv")
    p.add_argument("--params-file", default=None, help="JSON file with model params (optional)")
    p.add_argument("--model-out", default="models/lgbm_model.pkl")
    p.add_argument("--metrics-out", default="models/metrics.json")
    p.add_argument("--feature-imp-out", default="models/feature_importances.csv")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--n-estimators", type=int, default=500)
    p.add_argument("--early-stopping", type=int, default=50)
    args = p.parse_args()

    X, y = load_and_prep(args.input)
    # drop columns that are entirely missing (these can't be imputed)
    dropped_allna = [c for c in X.columns if X[c].isna().all()]
    if dropped_allna:
        print(f"Dropping all-NaN columns: {dropped_allna}")
        X = X.drop(columns=dropped_allna)

    imputer = SimpleImputer(strategy="median")
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X_imputed, y, test_size=args.test_size, stratify=y, random_state=args.random_state
    )

    # further split train -> train/val for early stopping
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.15, stratify=y_train, random_state=args.random_state
    )

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
    model_params.setdefault("random_state", args.random_state)
    model_params.setdefault("n_jobs", -1)

    clf = lgb.LGBMClassifier(**model_params)

    # fit with early stopping via callbacks
    callbacks = []
    if args.early_stopping and int(args.early_stopping) > 0:
        callbacks.append(lgb.early_stopping(stopping_rounds=int(args.early_stopping)))
    # silence training logs
    callbacks.append(lgb.log_evaluation(period=0))

    clf.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], eval_metric="auc", callbacks=callbacks)

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
    joblib.dump({"model": clf, "imputer": imputer, "features": list(X.columns)}, args.model_out)

    Path(args.metrics_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.metrics_out, "w") as fh:
        json.dump(metrics, fh, indent=2)

    # feature importances
    fi = pd.DataFrame({"feature": X.columns, "importance": clf.feature_importances_})
    fi = fi.sort_values("importance", ascending=False)
    fi.to_csv(args.feature_imp_out, index=False)

    print("Training complete.")
    print(f"Model saved to: {args.model_out}")
    print(f"Metrics saved to: {args.metrics_out}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
