import shap
import pandas as pd
import matplotlib.pyplot as plt
from joblib import load
from .config import MODEL_DIR, TAB_DIR, FIG_DIR
from .utils import ensure_dirs

def explain():
    ensure_dirs(FIG_DIR)
    X_test = pd.read_csv("artifacts/tables/X_test_preview.csv")
    y_test = pd.read_csv("artifacts/tables/y_test.csv").squeeze("columns")

    # use Decision Tree (tree explainer) if available; fallback to KernelExplainer
    model = load(MODEL_DIR / "dtree.pkl")
    # The model is a Pipeline; get the final estimator & transformed data
    pre = model.named_steps["pre"]
    clf = model.named_steps["clf"]
    X_test_trans = pre.transform(X_test)

    try:
        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(X_test_trans)
        # For binary classification, shap_values is list [class0, class1]
        sv = shap_values[1] if isinstance(shap_values, list) else shap_values
        plt.figure()
        shap.summary_plot(sv, X_test_trans, show=False)
        plt.tight_layout(); plt.savefig(FIG_DIR / "shap_summary.png"); plt.close()
    except Exception:
        # fallback (slow) – comment out if too heavy
        explainer = shap.KernelExplainer(model.predict_proba, shap.kmeans(X_test, 30))
        sv = explainer.shap_values(X_test.sample(200, random_state=42))
        plt.figure()
        shap.summary_plot(sv[1], X_test.sample(200, random_state=42), show=False)
        plt.tight_layout(); plt.savefig(FIG_DIR / "shap_summary.png"); plt.close()

if __name__ == "__main__":
    explain()
