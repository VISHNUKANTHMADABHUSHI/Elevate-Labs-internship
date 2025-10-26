import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from joblib import load
from .config import MODEL_DIR, TAB_DIR

def evaluate():
    X_test = pd.read_csv("artifacts/tables/X_test_preview.csv")
    y_test = pd.read_csv("artifacts/tables/y_test.csv").squeeze("columns")

    results = []
    cms = {}

    for name in ["logreg", "dtree"]:
        model = load(MODEL_DIR / f"{name}.pkl")
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary", zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        results.append({"model": name, "accuracy": acc, "precision": prec, "recall": rec, "f1": f1})
        cms[name] = cm

    pd.DataFrame(results).to_csv(TAB_DIR / "metrics.csv", index=False)

    # Flatten confusion matrices
    cm_rows = []
    for name, cm in cms.items():
        tn, fp, fn, tp = cm.ravel()
        cm_rows.append({"model": name, "TN": tn, "FP": fp, "FN": fn, "TP": tp})
    pd.DataFrame(cm_rows).to_csv(TAB_DIR / "confusion_matrices.csv", index=False)

if __name__ == "__main__":
    evaluate()
