"""
Task 4: Severity Classifier (Model Selection over Multiple Algorithms)

Goal:
    - Load sentence embeddings + labeled data.
    - Train several classifiers:
        * Logistic Regression
        * RandomForest
        * MLP
        * XGBoost (if available)
    - Evaluate each on the same test set.
    - Select the best model based on macro-F1 (tie-breaker: accuracy).
    - Save:
        * Best model (joblib)
        * LabelEncoder
        * Per-model reports
        * JSON summary of metrics

Inputs (from Task 3, inside data_embeddings/):
    - data_embeddings/drug_food_interactions_embedded.csv
    - data_embeddings/embeddings_all_mpnet_base_v2.npy

Outputs (also into data_embeddings/):
    - data_embeddings/best_severity_classifier.pkl
    - data_embeddings/label_encoder_severity.pkl
    - data_embeddings/severity_models_metrics.json
    - data_embeddings/severity_report_<model_name>.txt
"""

import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# Try to import XGBoost (optional)
try:
    from xgboost import XGBClassifier

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("[INFO] xgboost not installed. XGBoost model will be skipped.")


# ==========================
# 1. Paths / Config
# ==========================

DATA_DIR = "data_embeddings"

CSV_PATH = os.path.join(DATA_DIR, "drug_food_interactions_embedded.csv")
NPY_PATH = os.path.join(DATA_DIR, "embeddings_all_mpnet_base_v2.npy")

BEST_MODEL_PATH = os.path.join(DATA_DIR, "best_severity_classifier.pkl")
LABEL_ENCODER_PATH = os.path.join(DATA_DIR, "label_encoder_severity.pkl")
METRICS_JSON_PATH = os.path.join(DATA_DIR, "severity_models_metrics.json")
REPORT_TEMPLATE = os.path.join(DATA_DIR, "severity_report_{model_name}.txt")


# ==========================
# 2. Load data
# ==========================

print(f"Loading DataFrame from: {CSV_PATH}")
df = pd.read_csv(CSV_PATH)

print(f"Loading embeddings from: {NPY_PATH}")
embeddings = np.load(NPY_PATH)

print(f"DataFrame shape: {df.shape}")
print(f"Embeddings shape: {embeddings.shape}")

assert (
    df.shape[0] == embeddings.shape[0]
), "Mismatch between df rows and embedding rows!"

X = embeddings  # (N, 768)
y_str = df["severity"].astype(str).values


# ==========================
# 3. Encode labels
# ==========================

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_str)

print("\nSeverity classes mapping:")
for idx, cls in enumerate(label_encoder.classes_):
    print(f"  {idx} -> {cls}")


# ==========================
# 4. Split train / test
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain size: {X_train.shape[0]}  |  Test size: {X_test.shape[0]}")


# ==========================
# 5. Define models
# ==========================

models = {}

# Logistic Regression
models["logreg"] = LogisticRegression(
    max_iter=5000, class_weight="balanced", n_jobs=-1, multi_class="auto"
)

# RandomForest
models["random_forest"] = RandomForestClassifier(
    n_estimators=300, random_state=42, class_weight="balanced_subsample", n_jobs=-1
)

# MLP
models["mlp"] = MLPClassifier(
    hidden_layer_sizes=(256, 128),
    activation="relu",
    solver="adam",
    max_iter=300,
    random_state=42,
)

# XGBoost (optional)
if XGBOOST_AVAILABLE:
    models["xgboost"] = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="multi:softprob",
        tree_method="hist",
        eval_metric="mlogloss",
        random_state=42,
    )


# ==========================
# 6. Train & evaluate each model
# ==========================

all_metrics = {}
best_model_name = None
best_macro_f1 = -1.0
best_accuracy = -1.0
best_model = None

for name, model in models.items():
    print(f"\n==============================")
    print(f"Training model: {name}")
    print(f"==============================")

    # Train
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
    cm = confusion_matrix(y_test, y_pred)

    print(f"[{name}] Accuracy: {acc:.4f}")
    print(f"[{name}] Macro F1: {macro_f1:.4f}")
    print(f"[{name}] Confusion matrix:\n{cm}")

    # Save metrics
    all_metrics[name] = {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }

    # Save a text report for this model
    report_path = REPORT_TEMPLATE.format(model_name=name)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Model: {name}\n\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"Macro F1: {macro_f1:.4f}\n\n")
        f.write("Classification report:\n")
        f.write(report)
        f.write("\nConfusion matrix (rows=true, cols=pred):\n")
        f.write(np.array2string(cm))
    print(f"[{name}] Saved detailed report to: {report_path}")

    # Track best model (primary metric: macro-F1, tie-breaker: accuracy)
    if macro_f1 > best_macro_f1 or (macro_f1 == best_macro_f1 and acc > best_accuracy):
        best_macro_f1 = macro_f1
        best_accuracy = acc
        best_model_name = name
        best_model = model


# ==========================
# 7. Save best model & label encoder
# ==========================

if best_model is None:
    raise RuntimeError("No model was trained successfully.")

print(f"\nBest model: {best_model_name}")
print(f"  Macro F1: {best_macro_f1:.4f}")
print(f"  Accuracy: {best_accuracy:.4f}")

joblib.dump(best_model, BEST_MODEL_PATH)
joblib.dump(label_encoder, LABEL_ENCODER_PATH)

print(f"Saved best model to: {BEST_MODEL_PATH}")
print(f"Saved label encoder to: {LABEL_ENCODER_PATH}")


# ==========================
# 8. Save metrics summary
# ==========================

summary = {
    "best_model_name": best_model_name,
    "best_macro_f1": best_macro_f1,
    "best_accuracy": best_accuracy,
    "models": all_metrics,
}

with open(METRICS_JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print(f"\nSaved metrics summary to: {METRICS_JSON_PATH}")
print("\nDone.")
print("\nLater, you can load the best model with:")
print(f"  clf = joblib.load('{BEST_MODEL_PATH}')")
print(f"  le = joblib.load('{LABEL_ENCODER_PATH}')")
print("  # embs = np.load('data_embeddings/embeddings_all_mpnet_base_v2.npy')")
print("  # pred = le.inverse_transform(clf.predict(embs[i:i+1]))")
