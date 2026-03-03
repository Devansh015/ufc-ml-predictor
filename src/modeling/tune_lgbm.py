import argparse
import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_val_score


# Columns that must never be used as features (target leakage)
_LEAK_COLS = {"winner", "blue_win"}


def load_and_prep(path, target_col="red_win"):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in {path}.")
    y = df[target_col].astype(int)
    drop_cols = [c for c in
                 {"fight_url", "red_fighter", "blue_fighter"} | _LEAK_COLS
                 if c in df.columns]
    X = df.drop(columns=drop_cols + [target_col], errors="ignore")
    X = X.select_dtypes(include=[np.number])
    # drop all-NaN cols
    X = X.loc[:, ~X.isna().all()]
    imputer = SimpleImputer(strategy="median")
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    return X_imputed, y, imputer


def objective(trial, X, y, cv_splits):
    params = {
        # narrower ranges and single-threaded to avoid nested parallelism
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "num_leaves": trial.suggest_int("num_leaves", 8, 64),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 80),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 0, 5),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 1.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 1.0, log=True),
        "random_state": 42,
        "n_jobs": 1,
    }

    clf = lgb.LGBMClassifier(**params)
    # use cross_val_score with roc_auc; run single-threaded to avoid nested jobs
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
    try:
        scores = cross_val_score(clf, X, y, cv=cv, scoring="roc_auc", n_jobs=1)
        return float(scores.mean())
    except Exception:
        return 0.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/features/fight_features_alpha.csv")
    p.add_argument("--n-trials", type=int, default=20)
    p.add_argument("--cv-splits", type=int, default=5)
    p.add_argument("--timeout", type=int, default=0, help="seconds, 0 means no timeout")
    p.add_argument("--model-out", default="models/lgbm_tuned.pkl")
    p.add_argument("--params-out", default="models/best_params.json")
    args = p.parse_args()

    X, y, imputer = load_and_prep(args.input)

    study = optuna.create_study(direction="maximize")
    func = lambda trial: objective(trial, X, y, args.cv_splits)
    if args.timeout and args.timeout > 0:
        study.optimize(func, timeout=args.timeout)
    else:
        study.optimize(func, n_trials=args.n_trials)

    best = study.best_params
    best["n_estimators"] = int(best["n_estimators"]) if "n_estimators" in best else 100

    # Train final model on 80% data (keep 20% truly unseen for honest eval)
    from sklearn.model_selection import train_test_split as _tts
    X_fit, _X_hold, y_fit, _y_hold = _tts(X, y, test_size=0.2, stratify=y, random_state=42)
    final_clf = lgb.LGBMClassifier(**best, n_jobs=-1, random_state=42)
    final_clf.fit(X_fit, y_fit)

    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": final_clf, "imputer": imputer, "features": list(X.columns)}, args.model_out)

    Path(args.params_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.params_out, "w") as fh:
        json.dump({"best_params": best, "best_value": study.best_value}, fh, indent=2)

    print("Tuning complete.")
    print(f"Best ROC AUC: {study.best_value:.4f}")
    print(f"Best params written to: {args.params_out}")
    print(f"Tuned model written to: {args.model_out}")


if __name__ == "__main__":
    main()
