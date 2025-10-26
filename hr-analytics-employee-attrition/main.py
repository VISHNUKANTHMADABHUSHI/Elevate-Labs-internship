from src.utils import ensure_dirs
from src.config import ARTIFACTS, FIG_DIR, TAB_DIR, MODEL_DIR
from src.eda import run_eda
from src.train import train_models
from src.evaluate import evaluate
from src.explain import explain

if __name__ == "__main__":
    ensure_dirs(ARTIFACTS, FIG_DIR, TAB_DIR, MODEL_DIR)
    print("Step 1/4: EDA & clean data → artifacts/...")
    run_eda()
    print("Step 2/4: Train models → artifacts/models/...")
    train_models()
    print("Step 3/4: Evaluate models → artifacts/tables/metrics.csv ...")
    evaluate()
    print("Step 4/4: Explainability (SHAP) → artifacts/figures/shap_summary.png ...")
    explain()
    print("✅ Done. Import artifacts/tables/clean_hr_data.csv into Power BI for dashboard.")
