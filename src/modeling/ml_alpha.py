import pandas as pd
import numpy as np
from pathlib import Path
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss
from sklearn.model_selection import train_test_split
import joblib

DATA = "data/features/fight_features_alpha.csv"
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def main():
    df = pd.read_csv(DATA)

    # Time-safe split: use event chronology proxy via file order (already built chronologically)
    # We'll do an 80/20 split by row index.
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].copy()
    test_df  = df.iloc[split_idx:].copy()

    y_train = train_df["red_win"].astype(int)
    y_test  = test_df["red_win"].astype(int)

    # Drop non-features
    drop_cols = ["fight_url", "red_fighter", "blue_fighter", "red_win"]
    X_train = train_df.drop(columns=drop_cols)
    X_test  = test_df.drop(columns=drop_cols)

    # Fill any remaining NaNs
    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)

    model = LGBMClassifier(
        n_estimators=2000,
        learning_rate=0.02,
        num_leaves=63,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=42,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric="binary_logloss",
    )

    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)

    acc = accuracy_score(y_test, preds)
    ll = log_loss(y_test, probs)
    brier = brier_score_loss(y_test, probs)

    print(f"Test Accuracy: {acc:.4f}")
    print(f"Test LogLoss:  {ll:.4f}")
    print(f"Test Brier:    {brier:.4f}")

    # Save model + feature columns
    joblib.dump(
        {"model": model, "feature_cols": X_train.columns.tolist()},
        MODELS_DIR / "lgbm_alpha.pkl"
    )

    # Feature importance
    imp = pd.DataFrame({
        "feature": X_train.columns,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)

    imp.to_csv(MODELS_DIR / "feature_importance_alpha.csv", index=False)
    print("Saved models/lgbm_alpha.pkl and models/feature_importance_alpha.csv")

if __name__ == "__main__":
    main()
