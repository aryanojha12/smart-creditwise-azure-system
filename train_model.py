"""
Smart CreditWise - train_model.py

Loads the loan dataset, cleans it, builds a preprocessing + model pipeline,
trains and compares 3 classifiers, selects the best by ROC-AUC, and saves
the COMPLETE pipeline (preprocessing + model together) to:

    models/loan_risk_pipeline.pkl

Saving the full pipeline (not just the model) guarantees the exact same
preprocessing is applied at training and prediction time - no leakage,
no mismatch.

Run:
    python train_model.py
"""
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "dataset.csv"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

TARGET = "Loan_Status"

CATEGORICAL_FEATURES = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "Property_Area",
]
NUMERIC_FEATURES = [
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History",
]
ALL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def load_and_clean_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df[TARGET] = df[TARGET].map({"Y": 1, "N": 0})
    # "3+" -> "3" so it's a clean category (still treated as categorical, not numeric)
    df["Dependents"] = df["Dependents"].astype(str).str.replace("+", "", regex=False)
    return df


def build_preprocessor() -> ColumnTransformer:
    categorical_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    numeric_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
            ("num", numeric_pipeline, NUMERIC_FEATURES),
        ]
    )


def main():
    df = load_and_clean_data()
    X = df[ALL_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    candidates = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
        "random_forest": RandomForestClassifier(
            n_estimators=200, max_depth=6, random_state=42
        ),
        "gradient_boosting": GradientBoostingClassifier(random_state=42),
    }

    results = {}
    fitted = {}

    for name, clf in candidates.items():
        pipe = Pipeline(steps=[("preprocess", build_preprocessor()), ("model", clf)])
        pipe.fit(X_train, y_train)

        preds = pipe.predict(X_test)
        probs = pipe.predict_proba(X_test)[:, 1]
        cm = confusion_matrix(y_test, preds).tolist()

        results[name] = {
            "accuracy": round(accuracy_score(y_test, preds), 4),
            "precision": round(precision_score(y_test, preds), 4),
            "recall": round(recall_score(y_test, preds), 4),
            "f1": round(f1_score(y_test, preds), 4),
            "roc_auc": round(roc_auc_score(y_test, probs), 4),
            "confusion_matrix": cm,  # [[TN, FP], [FN, TP]]
        }
        fitted[name] = pipe

    best_name = max(results, key=lambda n: results[n]["roc_auc"])
    best_pipeline = fitted[best_name]

    # Save the COMPLETE pipeline (preprocessing + model) as one object
    joblib.dump(best_pipeline, MODEL_DIR / "loan_risk_pipeline.pkl")

    # Feature importance / coefficients for the selected model, for the
    # Model Insights + explanation pages (real values, not fabricated)
    feature_names = list(
        best_pipeline.named_steps["preprocess"].get_feature_names_out()
    )
    model_step = best_pipeline.named_steps["model"]
    if hasattr(model_step, "feature_importances_"):
        importances = model_step.feature_importances_.tolist()
    elif hasattr(model_step, "coef_"):
        importances = model_step.coef_[0].tolist()
    else:
        importances = None

    report = {
        "selected_model": best_name,
        "selection_criterion": "highest ROC-AUC on held-out test set",
        "features_used": ALL_FEATURES,
        "n_features_raw": len(ALL_FEATURES),
        "n_features_after_encoding": len(feature_names),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "all_models": results,
        "feature_names_encoded": feature_names,
        "feature_importance_or_coef": importances,
    }
    with open(MODEL_DIR / "metrics.json", "w") as f:
        json.dump(report, f, indent=2)

    print("=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    for name, m in results.items():
        marker = "  <-- SELECTED" if name == best_name else ""
        print(
            f"{name:20s} acc={m['accuracy']:.4f} prec={m['precision']:.4f} "
            f"rec={m['recall']:.4f} f1={m['f1']:.4f} roc_auc={m['roc_auc']:.4f}{marker}"
        )
    print(f"\nSaved: models/loan_risk_pipeline.pkl")
    print(f"Saved: models/metrics.json")


if __name__ == "__main__":
    main()
