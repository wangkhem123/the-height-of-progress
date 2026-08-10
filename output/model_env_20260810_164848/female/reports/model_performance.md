# Model Performance Report
**Generated:** 2026-08-10 16:49

## Ensemble Metrics

| Metric | Value |
|--------|-------|
| R2 | 0.9579 |
| RMSE | 0.8752 cm |
| MAE | 0.6027 cm |

**Target:** R2 >= 0.5  
**Status:** PASS

## Per-Fold Metrics

| Fold | R2 | RMSE | MAE |
|------|-----|------|-----|
| 1 | 0.9515 | 0.9511 | 0.6199 |
| 2 | 0.9185 | 1.1924 | 0.6767 |
| 3 | 0.9450 | 0.9234 | 0.5757 |
| 4 | 0.9166 | 1.3094 | 0.7639 |
| 5 | 0.9366 | 1.0861 | 0.6637 |

**Mean fold R2:** 0.9337 +/- 0.0140

## Top Features

| Rank | Feature | Importance % | Stability |
|------|---------|-------------|-----------|
| 1 | physician_density | 33.27 | 0.900 |
| 2 | protein_g_per_day | 12.62 | 0.897 |
| 3 | breastfeeding_exclusive_pct | 11.05 | 0.900 |
| 4 | healthcare_pct_gdp | 7.31 | 0.800 |
| 5 | gini_index | 7.08 | 0.848 |
| 6 | pm25_ug_m3 | 6.40 | 0.774 |
| 7 | animal_protein_pct | 5.13 | 0.862 |
| 8 | caloric_intake_kcal | 4.93 | 0.779 |
| 9 | gdp_per_capita_ppp | 4.70 | 0.858 |
| 10 | sanitation_pct | 3.87 | 0.857 |
| 11 | urbanization_pct | 3.64 | 0.869 |

## Config

```json
{
  "objective": "reg:squarederror",
  "tree_method": "hist",
  "max_depth": 6,
  "learning_rate": 0.05,
  "n_estimators": 300,
  "subsample": 0.8,
  "colsample_bytree": 0.8,
  "min_child_weight": 3,
  "gamma": 0.1,
  "reg_alpha": 0.1,
  "reg_lambda": 1.0,
  "random_state": 42,
  "n_jobs": -1
}
```