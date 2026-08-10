"""
====================================================================================
CONFIGURATION — VizCon 2026 Height Story (All Countries)
====================================================================================
"""

# =============================================================================
# TARGET
# =============================================================================
TARGET_VARIABLE = "height_cm"

# =============================================================================
# MODEL HYPERPARAMETERS (XGBoost)
# =============================================================================
MODEL_PARAMS = {
    'objective': 'reg:squarederror',
    'tree_method': 'hist',
    'max_depth': 6,
    'learning_rate': 0.05,
    'n_estimators': 300,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 3,
    'gamma': 0.1,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'random_state': 42,
    'n_jobs': -1,
}

# =============================================================================
# ENSEMBLE / TRAINING SETTINGS
# =============================================================================
N_FOLDS = 5
TEST_SIZE = 0.2
RANDOM_STATE = 42
MIN_BIRTH_YEAR = 1970  # Only train on cohorts with good data alignment

# =============================================================================
# PATHS
# =============================================================================
OUTPUT_DIR = "output"
DATA_RAW_DIR = "data/raw"
DATA_PROCESSED_DIR = "data/processed"
PREPARED_DATA_FILE = "unified_height_features.csv"

# =============================================================================
# FEATURES — columns excluded from modeling
# =============================================================================
EXCLUDE_COLUMNS = ['country_code', 'country_name', 'birth_year', 'sex', TARGET_VARIABLE]

# =============================================================================
# VALIDATION
# =============================================================================
MIN_R2_THRESHOLD = 0.50
PLOT_DPI = 150
