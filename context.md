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
