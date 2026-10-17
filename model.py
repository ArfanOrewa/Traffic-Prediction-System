"""
model.py — Train and evaluate the congestion-prediction models.

- Primary model: RandomForestClassifier (the resume's headline model).
- Comparison (for the analytics page): XGBoost vs Logistic Regression,
  trained on the identical split.
- Saves: the fitted RF pipeline, the raw feature column list, the
  model-comparison table, and the typical-conditions lookup used by the router.

Run:
    python model.py
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

try:
    from xgboost import XGBClassifier
    _XGBOOST_AVAILABLE = True
except Exception:  # e.g. missing OpenMP runtime (libomp) on macOS
    XGBClassifier = None
    _XGBOOST_AVAILABLE = False

from preprocess import (CATEGORICAL_FEATURES, NUMERIC_FEATURES,
                        build_preprocessor, build_typical_conditions,
                        load_dataframe, prepare)

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

RF_PARAMS = dict(n_estimators=200, max_depth=12,
                 random_state=42, class_weight="balanced")


def feature_names(pipe: Pipeline) -> list:
    """Human-readable feature names after the ColumnTransformer."""
    pre = pipe.named_steps["preprocess"]
    names = []
    for name, transformer, columns in pre.transformers_:
        if name == "num":
            names.extend([f"num__{c}" for c in columns])
        elif name == "cat":
            ohe = transformer.named_steps["onehot"]
            names.extend([f"cat__{c}" for c in ohe.get_feature_names_out(columns)])
    return names


def evaluate(name: str, y_true, y_pred) -> dict:
    """Metric dict for one model."""
    return {
        "model": name,
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision_macro": round(float(precision_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "recall_macro": round(float(recall_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "f1_macro": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
    }


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading and preparing data...")
    df = load_dataframe()
    X, y = prepare(df)
    print(f"  rows: {len(X):,} | features: {X.shape[1]} | classes: {sorted(y.unique())}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y)
    print(f"  train: {len(X_train):,} | test: {len(X_test):,}")

    pre = build_preprocessor()

    # ------------------------- primary model: Random Forest ----------------
    rf = Pipeline([("preprocess", pre),
                   ("clf", RandomForestClassifier(**RF_PARAMS))])
    print("\nTraining Random Forest...")
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"\n=== Random Forest ===\nAccuracy: {acc:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, digits=4))
    print("Confusion matrix [rows=true, cols=pred]:")
    print(confusion_matrix(y_test, y_pred, labels=["Low", "Medium", "High"]))

    importances = rf.named_steps["clf"].feature_importances_
    names = feature_names(rf)
    top = sorted(zip(names, importances), key=lambda t: t[1], reverse=True)[:10]
    print("\nTop 10 feature importances:")
    for name, imp in top:
        print(f"  {name:45s} {imp:.4f}")

    assert acc >= 0.85, f"Acceptance failed: accuracy {acc:.4f} < 0.85"

    # ------------------------------- comparison ----------------------------
    from sklearn.preprocessing import LabelEncoder
    label_enc = LabelEncoder()
    y_train_enc = label_enc.fit_transform(y_train)
    y_test_enc = label_enc.transform(y_test)

    contenders = {
        "Random Forest": (RandomForestClassifier(**RF_PARAMS), y_train, y_test, False),
    }
    if _XGBOOST_AVAILABLE:
        # XGBoost needs integer labels.
        contenders["XGBoost"] = (
            XGBClassifier(n_estimators=200, max_depth=6,
                          learning_rate=0.08, subsample=0.9,
                          colsample_bytree=0.9, random_state=42,
                          n_jobs=-1, eval_metric="mlogloss"),
            y_train_enc, y_test_enc, True,
        )
    else:
        print("  (XGBoost skipped: library could not be loaded "
              "— on macOS run `brew install libomp` to enable it)")
    contenders["Logistic Regression"] = (
        LogisticRegression(max_iter=2000, n_jobs=-1),
        y_train, y_test, False,
    )
    results = []
    print("\n=== Model comparison (test set) ===")
    for name, (clf, yt_tr, yt_te, encoded) in contenders.items():
        pipe = Pipeline([("preprocess", build_preprocessor()), ("clf", clf)])
        pipe.fit(X_train, yt_tr)
        y_hat = pipe.predict(X_test)
        if encoded:  # map integer predictions back to label strings
            y_hat = label_enc.inverse_transform(y_hat)
            yt_te = label_enc.inverse_transform(yt_te)
        metrics = evaluate(name, yt_te, y_hat)
        results.append(metrics)
        print(f"  {name:20s} acc={metrics['accuracy']:.4f} "
              f"f1={metrics['f1_macro']:.4f}")

    # --------------------------------- saving ------------------------------
    joblib.dump(rf, MODELS_DIR / "traffic_rf_model.pkl")
    with open(MODELS_DIR / "feature_columns.json", "w") as f:
        json.dump(NUMERIC_FEATURES + CATEGORICAL_FEATURES, f, indent=2)
    with open(MODELS_DIR / "model_comparison.json", "w") as f:
        json.dump(results, f, indent=2)
    typical = build_typical_conditions(df)
    with open(MODELS_DIR / "typical_conditions.json", "w") as f:
        json.dump(typical, f)

    print(f"\nSaved: traffic_rf_model.pkl, feature_columns.json, "
          f"model_comparison.json, typical_conditions.json -> {MODELS_DIR}")
    print("\nDone.")


if __name__ == "__main__":
    main()
