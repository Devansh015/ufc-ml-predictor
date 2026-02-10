#!/usr/bin/env python3
"""Generate ROC, calibration plots and SHAP explanations for a trained model."""
import argparse
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (auc, accuracy_score, confusion_matrix,
                             f1_score, precision_score, recall_score,
                             roc_curve, roc_auc_score)
from sklearn.model_selection import train_test_split


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/lgbm_tuned_final.pkl")
    p.add_argument("--data", default="data/features/fight_features_alpha.csv")
    p.add_argument("--target", default="red_win")
    p.add_argument("--out-dir", default="models/analysis")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--random-state", type=int, default=42)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_pkg = joblib.load(args.model)
    clf = model_pkg.get("model") or model_pkg
    imputer = model_pkg.get("imputer")
    features = model_pkg.get("features")

    df = pd.read_csv(args.data)
    df.columns = df.columns.str.strip()
    if args.target not in df.columns:
        raise KeyError(f"Target column '{args.target}' not found in {args.data}")

    if features is None:
        # fallback: use all numeric except target
        X = df.select_dtypes(include=[np.number]).drop(columns=[args.target], errors="ignore")
        features = list(X.columns)
    else:
        # ensure all features exist in df
        missing = [f for f in features if f not in df.columns]
        if missing:
            raise KeyError(f"Saved features missing from data: {missing}")
        X = df[features]

    y = df[args.target].astype(int)

    # impute with saved imputer if present
    if imputer is not None:
        X_imp = pd.DataFrame(imputer.transform(X), columns=features)
    else:
        X_imp = X.fillna(X.median())

    X_train, X_test, y_train, y_test = train_test_split(
        X_imp, y, test_size=args.test_size, stratify=y, random_state=args.random_state
    )

    # predictions
    y_proba = clf.predict_proba(X_test)[:, 1]
    y_pred = clf.predict(X_test)

    metrics = {
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

    # save metrics
    with open(out_dir / "analysis_metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2)

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.savefig(out_dir / "roc_curve.png", bbox_inches="tight")
    plt.close()

    # Calibration curve
    prob_true, prob_pred = calibration_curve(y_test, y_proba, n_bins=10)
    plt.figure(figsize=(6, 6))
    plt.plot(prob_pred, prob_true, marker="o", label="Calibration")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title("Calibration Curve")
    plt.grid(alpha=0.3)
    plt.savefig(out_dir / "calibration_curve.png", bbox_inches="tight")
    plt.close()

    # SHAP explanations (try import)
    try:
        import shap

        expl = shap.TreeExplainer(clf)
        shap_values = expl.shap_values(X_test)
        # for binary classifiers shap_values may be a list
        if isinstance(shap_values, list):
            sv = shap_values[1]
        else:
            sv = shap_values

        # summary (beeswarm)
        plt.figure(figsize=(8, 6))
        shap.summary_plot(sv, X_test, show=False)
        plt.savefig(out_dir / "shap_summary_beeswarm.png", bbox_inches="tight")
        plt.close()

        # bar plot
        plt.figure(figsize=(8, 6))
        shap.summary_plot(sv, X_test, plot_type="bar", show=False)
        plt.savefig(out_dir / "shap_summary_bar.png", bbox_inches="tight")
        plt.close()
        shap_status = "ok"
    except Exception as e:
        shap_status = f"failed: {e}"

    # save a short report
    report = {
        "plots": {
            "roc": str(out_dir / "roc_curve.png"),
            "calibration": str(out_dir / "calibration_curve.png"),
        },
        "shap_status": shap_status,
        "metrics": metrics,
    }
    with open(out_dir / "analysis_report.json", "w") as fh:
        json.dump(report, fh, indent=2)

    print("Analysis complete.")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
