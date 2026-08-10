"""Quick deep-dive analysis of the all-countries height dataset."""
import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
df = pd.read_csv(PROJECT_ROOT / "data/processed/unified_height_features.csv")

print("=" * 70)
print("DATASET DEEP DIVE")
print("=" * 70)

print(f"\n--- Shape ---")
print(f"  Total rows: {len(df)}")
print(f"  Countries: {df.country_code.nunique()}")
print(f"  Birth years: {df.birth_year.min()} to {df.birth_year.max()}")
male = df[df.sex == 'male']
female = df[df.sex == 'female']
print(f"  Male: {len(male)}, Female: {len(female)}")

print(f"\n--- Target Variable (height_cm) ---")
print(f"  Overall:  mean={df.height_cm.mean():.1f}, std={df.height_cm.std():.1f}, range=[{df.height_cm.min():.1f}, {df.height_cm.max():.1f}]")
print(f"  Male:     mean={male.height_cm.mean():.1f}, std={male.height_cm.std():.1f}, range=[{male.height_cm.min():.1f}, {male.height_cm.max():.1f}]")
print(f"  Female:   mean={female.height_cm.mean():.1f}, std={female.height_cm.std():.1f}, range=[{female.height_cm.min():.1f}, {female.height_cm.max():.1f}]")
print(f"  M-F gap:  {male.height_cm.mean() - female.height_cm.mean():.1f} cm")

print(f"\n--- Missing Data ---")
features = [c for c in df.columns if c not in ['country_code', 'country_name', 'birth_year', 'sex', 'height_cm']]
for col in features:
    n_miss = df[col].isna().sum()
    pct = n_miss / len(df) * 100
    if n_miss > 0:
        print(f"  {col:35s} {n_miss:>5} missing ({pct:.0f}%)")

print(f"\n--- Feature Correlations with Height (male only) ---")
male_clean = male[features + ['height_cm']].dropna()
print(f"  (Using {len(male_clean)} complete rows)")
for col in sorted(features, key=lambda c: abs(male_clean[c].corr(male_clean['height_cm'])), reverse=True):
    corr = male_clean[col].corr(male_clean['height_cm'])
    print(f"  {col:35s} r = {corr:+.3f}")

print(f"\n--- Variance Explained by Sex Alone ---")
total_var = df.height_cm.var()
within_var = df.groupby('sex')['height_cm'].var().mean()
r2_sex = 1 - within_var / total_var
print(f"  R-squared from sex alone: {r2_sex:.3f}")
print(f"  Remaining variance for environment: {1 - r2_sex:.3f}")

print(f"\n--- Height by Country (male, birth_year=2000) ---")
latest = male[male.birth_year == 2000].sort_values('height_cm', ascending=False)
print(f"  Top 10 tallest:")
for _, r in latest.head(10).iterrows():
    print(f"    {r.country_name:30s} {r.height_cm:.1f} cm")
print(f"  Bottom 10 shortest:")
for _, r in latest.tail(10).iterrows():
    print(f"    {r.country_name:30s} {r.height_cm:.1f} cm")

print(f"\n--- Male-Only Model Potential ---")
print(f"  Male std: {male.height_cm.std():.2f} cm")
print(f"  If model explains 50% of male variance: RMSE ~ {male.height_cm.std() * np.sqrt(0.5):.2f} cm")
print(f"  If model explains 70% of male variance: RMSE ~ {male.height_cm.std() * np.sqrt(0.3):.2f} cm")

print(f"\n--- Birth Year Trend (males, global mean) ---")
yearly = male.groupby('birth_year')['height_cm'].mean()
print(f"  1966: {yearly.iloc[0]:.1f} cm")
print(f"  1980: {yearly.loc[1980]:.1f} cm")
print(f"  1990: {yearly.loc[1990]:.1f} cm")
print(f"  2000: {yearly.iloc[-1]:.1f} cm")
print(f"  Total gain 1966-2000: +{yearly.iloc[-1] - yearly.iloc[0]:.1f} cm")
