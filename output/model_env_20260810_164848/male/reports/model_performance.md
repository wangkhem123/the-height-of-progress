# Model Performance Report
**Generated:** 2026-08-10 16:48

## Ensemble Metrics

| Metric | Value |
|--------|-------|
| R2 | 0.9599 |
| RMSE | 0.9939 cm |
| MAE | 0.6812 cm |

**Target:** R2 >= 0.5  
**Status:** PASS

## Per-Fold Metrics

| Fold | R2 | RMSE | MAE |
|------|-----|------|-----|
| 1 | 0.9521 | 1.0724 | 0.7167 |
| 2 | 0.9224 | 1.3822 | 0.8132 |
| 3 | 0.9430 | 1.1177 | 0.6828 |
| 4 | 0.9280 | 1.4030 | 0.8731 |
| 5 | 0.9478 | 1.1492 | 0.7307 |

**Mean fold R2:** 0.9386 +/- 0.0115

## Top Features

| Rank | Feature | Importance % | Stability |
|------|---------|-------------|-----------|
| 1 | physician_density | 41.13 | 0.935 |
| 2 | protein_g_per_day | 12.48 | 0.787 |
| 3 | breastfeeding_exclusive_pct | 8.95 | 0.837 |
| 4 | healthcare_pct_gdp | 7.64 | 0.833 |
| 5 | gini_index | 5.90 | 0.884 |
| 6 | urbanization_pct | 4.48 | 0.730 |
| 7 | gdp_per_capita_ppp | 4.33 | 0.833 |
| 8 | pm25_ug_m3 | 4.29 | 0.766 |
| 9 | caloric_intake_kcal | 3.82 | 0.892 |
| 10 | sanitation_pct | 3.65 | 0.886 |
| 11 | animal_protein_pct | 3.33 | 0.843 |

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