from pathlib import Path

# Paths
DATA_PATH = Path("data/WA_Fn-UseC_-HR-Employee-Attrition.csv")
ARTIFACTS = Path("artifacts")
FIG_DIR = ARTIFACTS / "figures"
TAB_DIR = ARTIFACTS / "tables"
MODEL_DIR = ARTIFACTS / "models"

# Modeling
TARGET_COL = "Attrition"  # Yes/No
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Categorical handling
NUMERIC_IMPUTE_STRATEGY = "median"
