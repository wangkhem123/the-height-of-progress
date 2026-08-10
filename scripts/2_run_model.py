"""
====================================================================================
STAGE 2 - MODEL TRAINING  (file: 2_run_model.py)
====================================================================================

Trains a disjoint-fold bagged XGBoost ensemble on the prepared height dataset.
Reports R², RMSE, MAE per fold and ensemble. Exports feature importance + SHAP.

WHAT THIS FILE DOES:
  1. Loads prepared dataset from Stage 1 (data/processed/unified_height_features.csv)
  2. Filters to birth cohorts >= MIN_BIRTH_YEAR (where features align)
  3. Trains N_FOLDS disjoint-fold XGBoost models
  4. Evaluates bagged ensemble on union of per-fold holdouts
  5. Computes gain-based and SHAP feature importance
  6. Saves all outputs: metrics JSON, importance CSV, plots, model pickles

OUTPUT (per run):
  output/model_<timestamp>/
    reports/ensemble_metrics.json       <- R², RMSE, MAE
    reports/feature_importance.csv      <- gain-based ranking with stability
    reports/shap_importance.csv         <- SHAP-based ranking
    reports/model_performance.md        <- human-readable summary
    model/ensemble_metadata.json        <- model config
    model/member_N/height_model.pkl     <- trained models
    plots/feature_importance.png        <- bar chart
    plots/shap_summary.png             <- SHAP beeswarm

Pipeline order: config.py -> 1_run_data_prep.py -> [2_run_model.py]

Usage:
    python scripts/2_run_model.py
    python scripts/2_run_model.py --max-depth 8 --n-estimators 500
"""
import sys, os, json, pickle, argparse, warnings
from pathlib import Path
from datetime import datetime
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    TARGET_VARIABLE, MODEL_PARAMS, N_FOLDS, TEST_SIZE, RANDOM_STATE,
    MIN_BIRTH_YEAR, OUTPUT_DIR, EXCLUDE_COLUMNS, MIN_R2_THRESHOLD, PLOT_DPI,
    DATA_PROCESSED_DIR, PREPARED_DATA_FILE,
)

PROJECT_ROOT = Path(__file__).parent.parent


# =============================================================================
# MODEL CLASS
# =============================================================================
class HeightModel:
    """Single XGBoost regressor wrapper."""
    def __init__(self, params=None):
        self.params = params or MODEL_PARAMS
        self.model = xgb.XGBRegressor(**self.params)
        self.feature_columns = None

    def train(self, X, y):
        self.feature_columns = X.columns.tolist()
        self.model.fit(X, y, verbose=False)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X, y):
        preds = self.predict(X)
        return {'r2_score': float(r2_score(y, preds)),
                'rmse': float(np.sqrt(mean_squared_error(y, preds))),
                'mae': float(mean_absolute_error(y, preds))}

    def get_importance(self):
        imp = self.model.feature_importances_
        df = pd.DataFrame({'feature': self.feature_columns, 'importance': imp})
        df['importance_pct'] = (df['importance'] / df['importance'].sum() * 100).round(3)
        return df.sort_values('importance', ascending=False).reset_index(drop=True)


# =============================================================================
# ENSEMBLE TRAINING
# =============================================================================
def train_ensemble(X, y, params, n_folds=N_FOLDS):
    """Train disjoint-fold bagged ensemble. Returns (members, results)."""
    n = len(X)
    print(f"\n{'='*60}")
    print(f"DISJOINT-FOLD ENSEMBLE ({n_folds} folds, {n} rows, {X.shape[1]} features)")
    print(f"{'='*60}")

    rng = np.random.RandomState(RANDOM_STATE)
    folds = np.array_split(rng.permutation(n), n_folds)

    members, fold_metrics, eval_X, eval_y = [], [], [], []
    for i, idx in enumerate(folds, 1):
        Xf, yf = X.iloc[idx], y.iloc[idx]
        Xtr, Xte, ytr, yte = train_test_split(Xf, yf, test_size=TEST_SIZE, random_state=RANDOM_STATE)
        print(f"\n  Fold {i}/{n_folds}: train {len(Xtr)} / test {len(Xte)}")

        m = HeightModel(params)
        m.train(Xtr, ytr)
        metrics = m.evaluate(Xte, yte)
        print(f"    R²={metrics['r2_score']:.4f}  RMSE={metrics['rmse']:.2f}  MAE={metrics['mae']:.2f}")

        members.append(m)
        fold_metrics.append(metrics)
        eval_X.append(Xte)
        eval_y.append(yte)

    # Ensemble evaluation
    X_eval = pd.concat(eval_X, ignore_index=True)
    y_eval = pd.concat(eval_y, ignore_index=True)
    ens_preds = np.mean([m.predict(X_eval) for m in members], axis=0)
    ens_metrics = {'r2_score': float(r2_score(y_eval, ens_preds)),
                   'rmse': float(np.sqrt(mean_squared_error(y_eval, ens_preds))),
                   'mae': float(mean_absolute_error(y_eval, ens_preds))}

    print(f"\n  ENSEMBLE: R²={ens_metrics['r2_score']:.4f}  "
          f"RMSE={ens_metrics['rmse']:.2f}  MAE={ens_metrics['mae']:.2f}")

    # Aggregate importance (mean ± std across members)
    imps = [m.get_importance() for m in members]
    merged = imps[0][['feature']].copy()
    for i, imp in enumerate(imps):
        merged = merged.merge(imp[['feature','importance_pct']].rename(
            columns={'importance_pct': f'p{i}'}), on='feature', how='outer')
    pcols = [c for c in merged.columns if c.startswith('p')]
    merged['importance_pct_mean'] = merged[pcols].mean(axis=1).round(3)
    merged['importance_pct_std'] = merged[pcols].std(axis=1).round(3)
    merged['stability'] = (1 - merged['importance_pct_std'] / merged['importance_pct_mean'].clip(0.01)).round(3)
    merged = merged.sort_values('importance_pct_mean', ascending=False).reset_index(drop=True)
    merged['rank'] = range(1, len(merged) + 1)
    importance = merged[['rank','feature','importance_pct_mean','importance_pct_std','stability']]

    print(f"\n  Top features:\n{importance.head(10).to_string(index=False)}")

    return members, {'ensemble_metrics': ens_metrics, 'fold_metrics': fold_metrics,
                     'importance': importance, 'X_eval': X_eval, 'y_eval': y_eval}


# =============================================================================
# SAVE OUTPUTS
# =============================================================================
def save_outputs(run_dir, members, results, feature_cols, params):
    """Persist all model artefacts."""
    rdir = os.path.join(run_dir, "reports")
    pdir = os.path.join(run_dir, "plots")
    mdir = os.path.join(run_dir, "model")
    for d in [rdir, pdir, mdir]:
        os.makedirs(d, exist_ok=True)

    ens = results['ensemble_metrics']
    folds = results['fold_metrics']
    importance = results['importance']

    # 1. Metrics JSON
    with open(f"{rdir}/ensemble_metrics.json", 'w') as f:
        json.dump({'ensemble': ens, 'folds': folds}, f, indent=2)

    # 2. Feature importance CSV
    importance.to_csv(f"{rdir}/feature_importance.csv", index=False)

    # 3. SHAP
    try:
        import shap
        explainer = shap.TreeExplainer(members[0].model)
        sv = explainer.shap_values(results['X_eval'])
        shap_imp = pd.DataFrame({'feature': feature_cols,
                                 'mean_abs_shap': np.abs(sv).mean(axis=0)})
        shap_imp = shap_imp.sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)
        shap_imp['rank'] = range(1, len(shap_imp) + 1)
        shap_imp.to_csv(f"{rdir}/shap_importance.csv", index=False)

        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 8))
        shap.summary_plot(sv, results['X_eval'], feature_names=feature_cols, show=False)
        plt.savefig(f"{pdir}/shap_summary.png", dpi=PLOT_DPI, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"  SHAP skipped: {e}")

    # 4. Feature importance plot
    try:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        top = importance.head(min(14, len(importance)))
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.barh(range(len(top)), top['importance_pct_mean'], color='steelblue')
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels(top['feature'])
        ax.invert_yaxis()
        ax.set_xlabel('Importance (%)')
        ax.set_title('Feature Importance (XGBoost Gain)')
        plt.tight_layout()
        plt.savefig(f"{pdir}/feature_importance.png", dpi=PLOT_DPI, bbox_inches='tight')
        plt.close()
    except Exception:
        pass

    # 5. Model pickles
    for i, m in enumerate(members, 1):
        member_dir = f"{mdir}/member_{i}"
        os.makedirs(member_dir, exist_ok=True)
        with open(f"{member_dir}/height_model.pkl", 'wb') as f:
            pickle.dump(m.model, f)
        with open(f"{member_dir}/metadata.json", 'w') as f:
            json.dump({'member': i, 'metrics': folds[i-1], 'params': params,
                       'features': feature_cols, 'trained': datetime.now().isoformat()}, f, indent=2)
    with open(f"{mdir}/ensemble_metadata.json", 'w') as f:
        json.dump({'type': 'bagged_disjoint_fold', 'n_members': len(members),
                   'features': feature_cols, 'created': datetime.now().isoformat()}, f, indent=2)

    # 6. Markdown report
    r2s = [fm['r2_score'] for fm in folds]
    lines = [
        "# Model Performance Report",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "", "## Ensemble Metrics", "",
        "| Metric | Value |", "|--------|-------|",
        f"| R2 | {ens['r2_score']:.4f} |",
        f"| RMSE | {ens['rmse']:.4f} cm |",
        f"| MAE | {ens['mae']:.4f} cm |",
        f"", f"**Target:** R2 >= {MIN_R2_THRESHOLD}  ",
        f"**Status:** {'PASS' if ens['r2_score'] >= MIN_R2_THRESHOLD else 'BELOW TARGET'}",
        "", "## Per-Fold Metrics", "",
        "| Fold | R2 | RMSE | MAE |", "|------|-----|------|-----|",
    ]
    for i, fm in enumerate(folds, 1):
        lines.append(f"| {i} | {fm['r2_score']:.4f} | {fm['rmse']:.4f} | {fm['mae']:.4f} |")
    lines += ["", f"**Mean fold R2:** {np.mean(r2s):.4f} +/- {np.std(r2s):.4f}",
              "", "## Top Features", "",
              "| Rank | Feature | Importance % | Stability |", "|------|---------|-------------|-----------|"]
    for _, row in importance.head(14).iterrows():
        lines.append(f"| {int(row['rank'])} | {row['feature']} | {row['importance_pct_mean']:.2f} | {row['stability']:.3f} |")
    lines += ["", "## Config", "", "```json", json.dumps(params, indent=2), "```"]
    with open(f"{rdir}/model_performance.md", 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print(f"\n  Outputs saved to: {run_dir}/")


# =============================================================================
# GENETIC PAIR COMPARISON (--compare mode)
# =============================================================================
def run_genetic_pair_comparison(df):
    """Compare genetically close country pairs to isolate environment effects."""
    df = df[df['sex'] == 'male']
    PAIRS = [
        ("KOR", "PRK", "S.KOREA vs N.KOREA", "Genetically IDENTICAL, split 1945"),
        ("NLD", "DEU", "NETHERLANDS vs GERMANY", "Very close Germanic ancestry"),
        ("SWE", "DNK", "SWEDEN vs DENMARK", "Near-identical Scandinavian"),
        ("GTM", "NGA", "GUATEMALA vs NIGERIA", "Different genetics, similar poverty"),
    ]
    YEARS = [1900, 1940, 1970, 1980, 1990, 1996]

    print("\n" + "=" * 70)
    print("GENETIC PAIR COMPARISONS")
    print("=" * 70)

    for code_a, code_b, title, note in PAIRS:
        a = df[df['country_code'] == code_a].sort_values('birth_year')
        b = df[df['country_code'] == code_b].sort_values('birth_year')
        print(f"\n{'─'*70}\n  {title}\n  ({note})\n{'─'*70}")
        print(f"  {'Year':<6} {code_a:>6} {code_b:>6} {'Gap':>6} | {'meals_A':>7} {'meals_B':>7} | {'prot_A':>6} {'prot_B':>6} | {'cal_A':>6} {'cal_B':>6}")

        for yr in YEARS:
            ra = a[a['birth_year'] == yr]
            rb = b[b['birth_year'] == yr]
            if ra.empty or rb.empty:
                continue
            ra, rb = ra.iloc[0], rb.iloc[0]
            gap = ra.height_cm - rb.height_cm
            print(f"  {yr:<6} {ra.height_cm:>6.1f} {rb.height_cm:>6.1f} {gap:>+5.1f} | "
                  f"{ra.get('school_meal_pct',0):>5.0f}% {rb.get('school_meal_pct',0):>5.0f}% | "
                  f"{ra.get('protein_g_per_day',0):>5.0f}g {rb.get('protein_g_per_day',0):>5.0f}g | "
                  f"{ra.get('caloric_intake_kcal',0):>5.0f} {rb.get('caloric_intake_kcal',0):>5.0f}")

        a96 = a[a.birth_year == 1996].iloc[0]
        b96 = b[b.birth_year == 1996].iloc[0]
        a00 = a[a.birth_year == 1900].iloc[0]
        b00 = b[b.birth_year == 1900].iloc[0]
        print(f"\n  Century: {code_a} +{a96.height_cm-a00.height_cm:.1f} cm | {code_b} +{b96.height_cm-b00.height_cm:.1f} cm")
        if abs(a00.height_cm - b00.height_cm) < 1.5 and abs(a96.height_cm - b96.height_cm) > 5:
            print(f"  >> DIVERGENCE: Same genes, environment split them apart!")


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description='Stage 2: XGBoost model training')
    parser.add_argument('--max-depth', type=int, default=None)
    parser.add_argument('--n-estimators', type=int, default=None)
    parser.add_argument('--learning-rate', type=float, default=None)
    parser.add_argument('--compare', action='store_true', help='Run genetic pair comparison (no training)')
    parser.add_argument('--combined', action='store_true',
                        help='Train a single combined model (with is_male). Default is sex-separate.')
    args = parser.parse_args()

    # Load data (needed for both modes)
    data_path = PROJECT_ROOT / DATA_PROCESSED_DIR / PREPARED_DATA_FILE
    df = pd.read_csv(data_path)

    # --- Compare mode ---
    if args.compare:
        run_genetic_pair_comparison(df)
        return

    # Build params with overrides
    params = dict(MODEL_PARAMS)
    if args.max_depth: params['max_depth'] = args.max_depth
    if args.n_estimators: params['n_estimators'] = args.n_estimators
    if args.learning_rate: params['learning_rate'] = args.learning_rate

    print("=" * 60)
    print("STAGE 2: MODEL TRAINING")
    print("=" * 60)

    # Filter and prepare
    print(f"\n  Loading: {data_path}")
    print(f"  Full dataset: {len(df)} rows x {len(df.columns)} cols")

    # Filter to aligned cohorts
    df = df[df['birth_year'] >= MIN_BIRTH_YEAR].copy()
    print(f"  After filter (>= {MIN_BIRTH_YEAR}): {len(df)} rows")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.combined:
        # --- Legacy: single combined model with is_male ---
        feature_cols = [c for c in df.columns if c not in EXCLUDE_COLUMNS]
        X = df[feature_cols].copy()
        y = df[TARGET_VARIABLE].copy()
        for col in X.columns[X.isnull().any()]:
            X[col] = X[col].fillna(X[col].median())
        print(f"  Mode: COMBINED (with is_male)")
        print(f"  Features ({len(feature_cols)}): {feature_cols}")

        run_dir = str(PROJECT_ROOT / OUTPUT_DIR / f"model_combined_{ts}")
        members, results = train_ensemble(X, y, params)
        save_outputs(run_dir, members, results, feature_cols, params)
        ens = results['ensemble_metrics']
        print(f"\n{'='*60}")
        print(f"MODEL COMPLETE -> {run_dir}")
        print(f"  R2={ens['r2_score']:.4f}  RMSE={ens['rmse']:.2f} cm  MAE={ens['mae']:.2f} cm")
        print(f"{'='*60}")
    else:
        # --- Default: sex-separate models (environment-only, no is_male) ---
        # Environment features = everything except IDs, sex, target
        env_exclude = list(EXCLUDE_COLUMNS) + ['is_male']
        feature_cols = [c for c in df.columns if c not in env_exclude]

        print(f"  Mode: SEX-SEPARATE (environment-only, no is_male)")
        print(f"  Environment features ({len(feature_cols)}): {feature_cols}")

        run_dir = str(PROJECT_ROOT / OUTPUT_DIR / f"model_env_{ts}")
        os.makedirs(run_dir, exist_ok=True)

        all_results = {}
        for sex_label in ['male', 'female']:
            print(f"\n{'='*60}")
            print(f"  TRAINING: {sex_label.upper()}-ONLY MODEL")
            print(f"{'='*60}")

            subset = df[df['sex'] == sex_label].copy().reset_index(drop=True)
            X = subset[feature_cols].copy()
            y = subset[TARGET_VARIABLE].copy()

            # Impute NaN with median
            for col in X.columns[X.isnull().any()]:
                med = X[col].median()
                n_miss = X[col].isnull().sum()
                X[col] = X[col].fillna(med)
                print(f"    Imputed {col}: {n_miss} NaN -> median ({med:.1f})")

            print(f"  Rows: {len(X)}, Features: {len(feature_cols)}")

            sex_dir = f"{run_dir}/{sex_label}"
            members, results = train_ensemble(X, y, params)
            save_outputs(sex_dir, members, results, feature_cols, params)
            all_results[sex_label] = results

        # Combined summary report
        lines = [
            "# Environment-Only Model Performance",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "",
            "## Approach",
            "Separate models trained for males and females using ONLY childhood",
            "environment features (no sex/gender variable). This isolates how much",
            "of between-country height variation is explained by environment alone.",
            "",
            "## Results Summary",
            "",
            "| Model | N | Features | R² | RMSE (cm) | MAE (cm) |",
            "|-------|---|----------|-----|-----------|----------|",
        ]
        for sex_label in ['male', 'female']:
            ens = all_results[sex_label]['ensemble_metrics']
            n = len(df[df['sex'] == sex_label])
            lines.append(f"| {sex_label.capitalize()}-only | {n} | {len(feature_cols)} | "
                         f"{ens['r2_score']:.4f} | {ens['rmse']:.2f} | {ens['mae']:.2f} |")

        avg_r2 = np.mean([all_results[s]['ensemble_metrics']['r2_score'] for s in ['male','female']])
        lines += [
            "",
            f"**Average environment-only R²: {avg_r2:.4f}**",
            "",
            "## Interpretation",
            "",
            f"Childhood environment explains **~{avg_r2*100:.0f}%** of between-country height",
            "differences within each sex.",
            "The remaining variance likely reflects:",
            "- Genetic population differences",
            "- Unmeasured environmental factors (e.g., disease burden, food quality)",
            "- Measurement noise in the source data",
            "",
            "## Environment Features Used",
            "",
        ]
        for f in feature_cols:
            lines.append(f"- {f}")

        lines += [
            "", "## Per-Sex Feature Importance (Top 10)", "",
        ]
        for sex_label in ['male', 'female']:
            imp = all_results[sex_label]['importance']
            lines += [f"### {sex_label.capitalize()}", "",
                      "| Rank | Feature | Importance % | Stability |",
                      "|------|---------|-------------|-----------|"]
            for _, row in imp.head(10).iterrows():
                lines.append(f"| {int(row['rank'])} | {row['feature']} | "
                             f"{row['importance_pct_mean']:.2f} | {row['stability']:.3f} |")
            lines.append("")

        lines += ["", "## Config", "", "```json", json.dumps(params, indent=2), "```"]

        rdir = f"{run_dir}/reports"
        os.makedirs(rdir, exist_ok=True)
        with open(f"{rdir}/model_performance.md", 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

        # Save combined metrics JSON
        combined_metrics = {
            'male': all_results['male']['ensemble_metrics'],
            'female': all_results['female']['ensemble_metrics'],
            'average_r2': avg_r2,
            'mode': 'sex_separate_environment_only',
            'features': feature_cols,
            'n_features': len(feature_cols),
        }
        with open(f"{rdir}/ensemble_metrics.json", 'w') as f:
            json.dump(combined_metrics, f, indent=2)

        # Print final summary
        print(f"\n{'='*60}")
        print(f"MODEL COMPLETE -> {run_dir}")
        print(f"{'='*60}")
        for sex_label in ['male', 'female']:
            ens = all_results[sex_label]['ensemble_metrics']
            print(f"  {sex_label.upper():>8}: R²={ens['r2_score']:.4f}  "
                  f"RMSE={ens['rmse']:.2f} cm  MAE={ens['mae']:.2f} cm")
        print(f"  {'AVG':>8}: R²={avg_r2:.4f}")
        print(f"\n  → Environment alone explains ~{avg_r2*100:.0f}% of height variance")
        print(f"  Report: {rdir}/model_performance.md")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
