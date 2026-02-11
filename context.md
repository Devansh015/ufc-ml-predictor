# UFC ML Predictor Project Context

## Completed Features

- Scraped all UFC event and fight-details pages
- Built a reliable fight outcome dataset (red/blue corners, W/L/D/NC) from fight-details pages only
- Implemented a feature engineering script for recency-weighted, leak-free fighter stats
- Patched feature engineering to use fights_clean.csv for correct, leak-free labels and filtering
- Generated the final model-ready dataset (one row per fight, balanced target, no data leakage)

## Next Steps

- Train and evaluate a LightGBM model
- Document modeling results and pipeline
 
## Model Analysis Artifacts (generated)

- ROC curve: models/analysis/roc_curve.png
- Calibration curve: models/analysis/calibration_curve.png
- SHAP summary (beeswarm): models/analysis/shap_summary_beeswarm.png
- SHAP summary (bar): models/analysis/shap_summary_bar.png
- Analysis metrics & report: models/analysis/analysis_metrics.json, models/analysis/analysis_report.json

Notes: SHAP was computed using `shap.TreeExplainer` on the held-out test split created with `random_state=42` and `test_size=0.2`.
pip3 install -r requirements.txt
# or
pip3 install flask pandas numpy joblib