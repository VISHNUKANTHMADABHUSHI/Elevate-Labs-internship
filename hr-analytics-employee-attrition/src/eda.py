import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from .config import FIG_DIR, TAB_DIR, TARGET_COL
from .load_data import load_raw
from .utils import ensure_dirs

def run_eda():
    ensure_dirs(FIG_DIR, TAB_DIR)
    df = load_raw()

    # Basic overview
    desc = df.describe(include="all")
    desc.to_csv(TAB_DIR / "describe.csv", index=True)

    # Class balance
    ax = df[TARGET_COL].value_counts().plot(kind="bar", rot=0, title="Attrition Class Balance")
    plt.tight_layout(); plt.savefig(FIG_DIR / "class_balance.png"); plt.close()

    # Attrition by Department
    if "Department" in df.columns:
        plt.figure()
        sns.barplot(x="Department", y=None, hue=TARGET_COL,
                    data=df, estimator=lambda x: len(x)/len(df), errorbar=None)
        plt.title("Attrition Rate by Department")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout(); plt.savefig(FIG_DIR / "attrition_by_department.png"); plt.close()

    # Attrition vs OverTime
    if "OverTime" in df.columns:
        plt.figure()
        sns.countplot(x="OverTime", hue=TARGET_COL, data=df)
        plt.title("Attrition by OverTime")
        plt.tight_layout(); plt.savefig(FIG_DIR / "attrition_by_overtime.png"); plt.close()

    # Numeric correlation
    plt.figure()
    num_df = df.select_dtypes(include="number")
    if not num_df.empty:
        corr = num_df.corr(numeric_only=True)
        sns.heatmap(corr, cmap="coolwarm", center=0)
        plt.title("Numeric Correlation Heatmap")
        plt.tight_layout(); plt.savefig(FIG_DIR / "correlation_heatmap.png"); plt.close()

    # Save a lightly cleaned CSV for BI (drop obvious ID-like cols)
    drop_cols = [c for c in df.columns if c.lower() in {"employee_number", "employeecount", "standardhours", "over18"}]
    bi_df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    bi_df.to_csv(TAB_DIR / "clean_hr_data.csv", index=False)

if __name__ == "__main__":
    run_eda()
