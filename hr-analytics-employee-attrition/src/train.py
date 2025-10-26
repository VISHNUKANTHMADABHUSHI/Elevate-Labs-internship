import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from joblib import dump

from .config import ARTIFACTS, MODEL_DIR, TAB_DIR, TEST_SIZE, RANDOM_STATE, TARGET_COL
from .utils import ensure_dirs
from .preprocess import split_xy, build_preprocessor

def train_models():
    ensure_dirs(ARTIFACTS, MODEL_DIR, TAB_DIR)
    df = pd.read_csv("artifacts/tables/clean_hr_data.csv")  # from EDA step
    X, y = split_xy(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    pre = build_preprocessor(X_train)

    # Logistic Regression
    logreg = Pipeline(steps=[
        ("pre", pre),
        ("clf", LogisticRegression(max_iter=200, n_jobs=None, class_weight="balanced"))
    ])
    logreg.fit(X_train, y_train)
    dump(logreg, MODEL_DIR / "logreg.pkl")

    # Decision Tree
    dtree = Pipeline(steps=[
        ("pre", pre),
        ("clf", DecisionTreeClassifier(random_state=RANDOM_STATE, class_weight="balanced"))
    ])
    dtree.fit(X_train, y_train)
    dump(dtree, MODEL_DIR / "dtree.pkl")

    # Keep test split for evaluation
    X_train.to_csv(TAB_DIR / "X_train_preview.csv", index=False)
    X_test.to_csv(TAB_DIR / "X_test_preview.csv", index=False)
    y_train.to_csv(TAB_DIR / "y_train.csv", index=False)
    y_test.to_csv(TAB_DIR / "y_test.csv", index=False)

if __name__ == "__main__":
    train_models()
