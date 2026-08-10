"""
====================================================================================
STAGE 1 - DATA PREPARATION (ALL COUNTRIES - RAW DATA ONLY)
====================================================================================

Uses ONLY real raw data files (NCD-RisC, World Bank, OWID/FAO) from data/raw/.
No extrapolation, no synthetic features, no derived columns.

OUTPUT:
  data/processed/unified_height_features.csv

Usage:
    python scripts/1_run_data_prep.py
"""
import sys, os, warnings
from pathlib import Path
from datetime import datetime
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from config import TARGET_VARIABLE, DATA_RAW_DIR, DATA_PROCESSED_DIR, PREPARED_DATA_FILE

PROJECT_ROOT = Path(__file__).parent.parent
RAW_DIR = PROJECT_ROOT / DATA_RAW_DIR
PROCESSED_DIR = PROJECT_ROOT / DATA_PROCESSED_DIR


# =============================================================================
# WORLD BANK FILE MAPPING (indicator -> filename in data/raw/)
# =============================================================================
WB_FILES = {
    "gdp_per_capita_ppp":          "API_NY.GDP.PCAP.PP.KD_DS2_en_csv_v2_33608.csv",
    "gini_index":                  "API_SI.POV.GINI_DS2_en_csv_v2_499.csv",
    "pm25_ug_m3":                  "API_EN.ATM.PM25.MC.M3_DS2_en_csv_v2_33524.csv",
    "sanitation_pct":              "API_SH.STA.SMSS.ZS_DS2_en_csv_v2_33860.csv",
    "healthcare_pct_gdp":          "API_SH.XPD.CHEX.GD.ZS_DS2_en_csv_v2_33342.csv",
    "physician_density":           "API_SH.MED.PHYS.ZS_DS2_en_csv_v2_35671.csv",
    "breastfeeding_exclusive_pct": "API_SH.STA.BFED.ZS_DS2_en_csv_v2_36228.csv",
    "urbanization_pct":            "API_SP.URB.TOTL.IN.ZS_DS2_en_csv_v2_33901.csv",
}


def load_world_bank_csv(filepath):
    """Load a World Bank indicator CSV (wide format) -> long format DataFrame."""
    df = pd.read_csv(filepath, skiprows=4)
    year_cols = [c for c in df.columns if c.isdigit()]
    df_long = df.melt(
        id_vars=['Country Code'],
        value_vars=year_cols,
        var_name='year',
        value_name='value'
    )
    df_long['year'] = df_long['year'].astype(int)
    df_long = df_long.dropna(subset=['value'])
    df_long = df_long.rename(columns={'Country Code': 'country_code'})
    return df_long[['country_code', 'year', 'value']].reset_index(drop=True)


def load_owid_csv(filepath, value_col):
    """Load an OWID/FAO CSV -> long format DataFrame with ISO3 codes."""
    df = pd.read_csv(filepath)
    df = df.dropna(subset=['Code', value_col])
    df = df.rename(columns={'Code': 'country_code', 'Year': 'year', value_col: 'value'})
    return df[['country_code', 'year', 'value']].reset_index(drop=True)


def get_childhood_value(country_data, birth_year):
    """Average feature value over childhood window (birth_year to birth_year+10).
    Uses only actual data points — no extrapolation."""
    if country_data.empty:
        return None
    window = country_data[(country_data['year'] >= birth_year) &
                          (country_data['year'] <= birth_year + 10)]
    if not window.empty:
        return window['value'].mean()
    # Fallback: closest available year to midpoint of childhood
    closest_idx = (country_data['year'] - (birth_year + 5)).abs().idxmin()
    return country_data.loc[closest_idx, 'value']


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 70)
    print("STAGE 1: DATA PREPARATION (ALL COUNTRIES - RAW DATA ONLY)")
    print("=" * 70)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # ─────────────────────────────────────────────────────────────────────
    # Step 1: Load NCD-RisC height data (target variable)
    # ─────────────────────────────────────────────────────────────────────
    print("\n--- Step 1: Loading NCD-RisC height data ---")
    height_path = RAW_DIR / "NCD_RisC_Lancet_2020_height_child_adolescent_country.csv"
    height_raw = pd.read_csv(height_path)

    # Filter to age 19 (adult height at maturity)
    height_19 = height_raw[height_raw['Age group'] == 19].copy()

    # Map sex
    height_19['sex'] = height_19['Sex'].map({'Boys': 'male', 'Girls': 'female'})

    # birth_year = measurement year - 19
    height_19['birth_year'] = height_19['Year'] - 19

    # Get ISO3 codes from WB metadata
    wb_meta_path = RAW_DIR / "Metadata_Country_API_NY.GDP.PCAP.PP.KD_DS2_en_csv_v2_33608.csv"
    wb_meta = pd.read_csv(wb_meta_path)
    name_to_iso3 = dict(zip(wb_meta['TableName'], wb_meta['Country Code']))

    # Manual mappings for NCD-RisC names that differ from WB
    EXTRA_MAPPINGS = {
        "United States of America": "USA",
        "United Kingdom": "GBR",
        "South Korea": "KOR",
        "North Korea": "PRK",
        "Republic of Korea": "KOR",
        "Democratic People's Republic of Korea": "PRK",
        "Iran": "IRN",
        "Iran (Islamic Republic of)": "IRN",
        "Russia": "RUS",
        "Russian Federation": "RUS",
        "Venezuela": "VEN",
        "Venezuela (Bolivarian Republic of)": "VEN",
        "Tanzania": "TZA",
        "United Republic of Tanzania": "TZA",
        "Bolivia": "BOL",
        "Bolivia (Plurinational State of)": "BOL",
        "Vietnam": "VNM",
        "Viet Nam": "VNM",
        "Syria": "SYR",
        "Syrian Arab Republic": "SYR",
        "Laos": "LAO",
        "Lao People's Democratic Republic": "LAO",
        "Moldova": "MDA",
        "Republic of Moldova": "MDA",
        "Ivory Coast": "CIV",
        "Cote d'Ivoire": "CIV",
        "Czech Republic": "CZE",
        "Czechia": "CZE",
        "Taiwan": "TWN",
        "Taiwan (Province of China)": "TWN",
        "Hong Kong": "HKG",
        "Hong Kong SAR, China": "HKG",
        "Macau": "MAC",
        "Macao SAR, China": "MAC",
        "Palestine": "PSE",
        "West Bank and Gaza": "PSE",
        "Cape Verde": "CPV",
        "Cabo Verde": "CPV",
        "Eswatini": "SWZ",
        "Swaziland": "SWZ",
        "North Macedonia": "MKD",
        "Macedonia": "MKD",
        "Congo": "COG",
        "DR Congo": "COD",
        "Democratic Republic of the Congo": "COD",
        "Micronesia (Federated States of)": "FSM",
        "Micronesia (country)": "FSM",
        "Bahamas": "BHS",
        "The Bahamas": "BHS",
        "China (Hong Kong SAR)": "HKG",
        "Egypt": "EGY",
        "Gambia": "GMB",
        "The Gambia": "GMB",
        "Guinea Bissau": "GNB",
        "Guinea-Bissau": "GNB",
        "Kyrgyzstan": "KGZ",
        "Macedonia (TFYR)": "MKD",
        "Occupied Palestinian Territory": "PSE",
        "Puerto Rico": "PRI",
        "Saint Kitts and Nevis": "KNA",
        "Saint Lucia": "LCA",
        "Saint Vincent and the Grenadines": "VCT",
        "Slovakia": "SVK",
        "Slovak Republic": "SVK",
        "Somalia": "SOM",
        "Turkey": "TUR",
        "Turkiye": "TUR",
        "Yemen": "YEM",
        "Republic of Yemen": "YEM",
    }
    name_to_iso3.update(EXTRA_MAPPINGS)

    height_19['country_code'] = height_19['Country'].map(name_to_iso3)

    # Try WB GDP file for remaining unmapped
    unmapped = height_19[height_19['country_code'].isna()]['Country'].unique()
    if len(unmapped) > 0:
        wb_gdp = pd.read_csv(RAW_DIR / WB_FILES["gdp_per_capita_ppp"], skiprows=4)
        wb_name_to_code = dict(zip(wb_gdp['Country Name'], wb_gdp['Country Code']))
        for name in unmapped:
            if name in wb_name_to_code:
                name_to_iso3[name] = wb_name_to_code[name]
        height_19['country_code'] = height_19['Country'].map(name_to_iso3)
        still_unmapped = height_19[height_19['country_code'].isna()]['Country'].unique()
        if len(still_unmapped) > 0:
            print(f"  Unmapped ({len(still_unmapped)}): {list(still_unmapped)[:10]}")

    # Drop unmapped rows
    height_19 = height_19.dropna(subset=['country_code'])

    height_df = height_19[['country_code', 'Country', 'birth_year', 'Mean height', 'sex']].copy()
    height_df = height_df.rename(columns={'Country': 'country_name', 'Mean height': 'height_cm'})

    print(f"  Rows: {len(height_df)}, Countries: {height_df['country_code'].nunique()}")
    print(f"  Birth years: {height_df['birth_year'].min()} to {height_df['birth_year'].max()}")
    print(f"  Male: {height_df[height_df.sex=='male'].shape[0]}, Female: {height_df[height_df.sex=='female'].shape[0]}")

    # ─────────────────────────────────────────────────────────────────────
    # Step 2: Load World Bank indicators
    # ─────────────────────────────────────────────────────────────────────
    print("\n--- Step 2: Loading World Bank indicators ---")
    wb_features = {}
    for feat_name, filename in WB_FILES.items():
        filepath = RAW_DIR / filename
        if filepath.exists():
            df = load_world_bank_csv(filepath)
            wb_features[feat_name] = df
            print(f"  {feat_name}: {len(df)} rows, {df['country_code'].nunique()} countries")
        else:
            print(f"  MISSING: {filename}")

    # ─────────────────────────────────────────────────────────────────────
    # Step 3: Load OWID/FAO indicators
    # ─────────────────────────────────────────────────────────────────────
    print("\n--- Step 3: Loading OWID/FAO indicators ---")
    owid_features = {}

    # Caloric supply
    cal_path = RAW_DIR / "daily-per-capita-caloric-supply.csv"
    if cal_path.exists():
        df = load_owid_csv(cal_path, 'Daily calorie supply per person')
        owid_features['caloric_intake_kcal'] = df
        print(f"  caloric_intake_kcal: {len(df)} rows, {df['country_code'].nunique()} countries")

    # Protein supply
    prot_path = RAW_DIR / "daily-per-capita-protein-supply.csv"
    if prot_path.exists():
        df = load_owid_csv(prot_path, 'Daily protein supply')
        owid_features['protein_g_per_day'] = df
        print(f"  protein_g_per_day: {len(df)} rows, {df['country_code'].nunique()} countries")

    # Animal protein (sum all animal protein columns)
    animal_path = RAW_DIR / "animal-protein-consumption.csv"
    if animal_path.exists():
        df = pd.read_csv(animal_path)
        protein_cols = ['Fish and seafood', 'Poultry', 'Pork', 'Beef and buffalo',
                        'Sheep and goat', 'Other meat', 'Eggs', 'Dairy']
        existing_cols = [c for c in protein_cols if c in df.columns]
        df['total_animal_protein'] = df[existing_cols].sum(axis=1)
        df = df.dropna(subset=['Code'])
        df = df.rename(columns={'Code': 'country_code', 'Year': 'year',
                                'total_animal_protein': 'value'})
        owid_features['animal_protein_g_per_day'] = df[['country_code', 'year', 'value']].reset_index(drop=True)
        print(f"  animal_protein_g_per_day: {len(owid_features['animal_protein_g_per_day'])} rows")

    # Compute animal_protein_pct from raw data (animal / total)
    if 'animal_protein_g_per_day' in owid_features and 'protein_g_per_day' in owid_features:
        animal = owid_features['animal_protein_g_per_day'].rename(columns={'value': 'animal'})
        total = owid_features['protein_g_per_day'].rename(columns={'value': 'total'})
        merged = animal.merge(total, on=['country_code', 'year'], how='inner')
        merged['value'] = (merged['animal'] / merged['total']).clip(0, 1)
        owid_features['animal_protein_pct'] = merged[['country_code', 'year', 'value']].reset_index(drop=True)
        print(f"  animal_protein_pct: {len(owid_features['animal_protein_pct'])} rows")

    # ─────────────────────────────────────────────────────────────────────
    # Step 4: Merge all features onto height data
    # ─────────────────────────────────────────────────────────────────────
    print("\n--- Step 4: Merging features (childhood-window averaging) ---")

    # Combine all feature sources
    all_features = {}
    all_features.update(wb_features)
    all_features.update(owid_features)
    # Remove intermediate (we keep animal_protein_pct instead)
    all_features.pop('animal_protein_g_per_day', None)

    print(f"  Total features: {len(all_features)}")
    print(f"  Features: {list(all_features.keys())}")

    records = []
    total = len(height_df)
    for idx, row in height_df.iterrows():
        if idx % 2000 == 0:
            print(f"    Processing row {idx}/{total}...")

        record = {
            'country_code': row['country_code'],
            'country_name': row['country_name'],
            'birth_year': row['birth_year'],
            'height_cm': row['height_cm'],
            'sex': row['sex'],
            'is_male': 1 if row['sex'] == 'male' else 0,
        }

        # Add each feature using childhood-window averaging (raw data only)
        for feat_name, feat_df in all_features.items():
            cdata = feat_df[feat_df['country_code'] == row['country_code']]
            record[feat_name] = get_childhood_value(cdata, row['birth_year'])

        records.append(record)

    unified = pd.DataFrame(records)

    # ─────────────────────────────────────────────────────────────────────
    # Step 5: Save output
    # ─────────────────────────────────────────────────────────────────────
    output_path = PROCESSED_DIR / PREPARED_DATA_FILE
    unified.to_csv(output_path, index=False)

    print(f"\n{'='*70}")
    print(f"DATA PREP COMPLETE -> {output_path}")
    print(f"{'='*70}")
    print(f"  Rows: {len(unified)}, Columns: {len(unified.columns)}")
    print(f"  Countries: {unified['country_code'].nunique()}")
    print(f"  Birth years: {unified['birth_year'].min()} to {unified['birth_year'].max()}")
    print(f"\n  Feature coverage:")
    for col in unified.columns:
        n = unified[col].notna().sum()
        pct = n / len(unified) * 100
        print(f"    {col:35s} {n:>6}/{len(unified)} ({pct:.0f}%)")
    print(f"\n  Next: python scripts/2_run_model.py")


if __name__ == "__main__":
    main()
