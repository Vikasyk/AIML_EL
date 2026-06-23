"""
train_rf.py
===========
Train a **Random Forest classifier** to predict a road edge's ``risk_level``
(Low / Medium / High) from weather / accident features.

Deliverables (per the project spec)
-----------------------------------
* Hyperparameters: n_estimators=200, class_weight='balanced', min_samples_split=5
* Evaluation: accuracy, precision/recall/F1 per class, confusion matrix, ROC-AUC
              (target accuracy > 80%)
* Baseline comparison: vs Logistic Regression
* Overfitting check: train accuracy vs test accuracy
* Confusion-matrix heatmap -> outputs/rf_confusion_matrix.png
* Saves the trained model -> models/rf_risk.pkl

Run:  python src/train_rf.py
"""

from __future__ import annotations

import os
import pickle

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

from preprocess import FEATURES, RISK_NAMES, load_processed

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
MODEL_PATH = os.path.join(MODELS_DIR, "rf_risk.pkl")


def train_rf(data: dict | None = None) -> RandomForestClassifier:
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    if data is None:
        data = load_processed()

    X_train, X_test = data["X_train"], data["X_test"]
    y_train, y_test = data["y_clf_train"], data["y_clf_test"]

    if data.get("is_synthetic"):
        print("[rf] NOTE: training on SYNTHETIC data (real CSV not found).")

    # --- Train the Random Forest ---------------------------------------- #
    rf = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        min_samples_split=5,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    # --- Evaluate -------------------------------------------------------- #
    train_acc = accuracy_score(y_train, rf.predict(X_train))
    test_pred = rf.predict(X_test)
    test_proba = rf.predict_proba(X_test)
    test_acc = accuracy_score(y_test, test_pred)
    weighted_f1 = f1_score(y_test, test_pred, average="weighted")

    # ROC-AUC (one-vs-rest, macro). Guard against a class missing in y_test.
    try:
        roc_auc = roc_auc_score(y_test, test_proba, multi_class="ovr", average="macro")
    except ValueError:
        roc_auc = float("nan")

    print("\n========== RANDOM FOREST (risk_level classification) ==========")
    print(f"Train accuracy : {train_acc:.4f}")
    print(f"Test  accuracy : {test_acc:.4f}   (target > 0.80)")
    print(f"Weighted F1    : {weighted_f1:.4f}")
    print(f"ROC-AUC (macro): {roc_auc:.4f}")
    print(f"Overfit gap    : {train_acc - test_acc:+.4f}  (train - test acc)")
    print("\nPer-class report:")
    print(classification_report(y_test, test_pred, target_names=RISK_NAMES, zero_division=0))

    # --- Baseline: Logistic Regression ---------------------------------- #
    base = LogisticRegression(max_iter=1000, class_weight="balanced", multi_class="auto")
    base.fit(X_train, y_train)
    base_acc = accuracy_score(y_test, base.predict(X_test))
    print("--- Baseline: Logistic Regression ---")
    print(f"Baseline Test accuracy : {base_acc:.4f}")
    print(f"=> Random Forest improves accuracy by {test_acc - base_acc:+.4f}")
    print("===============================================================")

    # --- Confusion matrix heatmap --------------------------------------- #
    _plot_confusion_matrix(y_test, test_pred)

    # --- Feature importance (printed) ----------------------------------- #
    importances = rf.feature_importances_
    print("\nFeature importance (risk):")
    for feat, imp in sorted(zip(FEATURES, importances), key=lambda t: -t[1]):
        print(f"    {feat:<20} {imp:.4f}")

    # --- Persist model --------------------------------------------------- #
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(rf, f)
    print(f"[rf] Saved model -> {MODEL_PATH}")

    return rf


def _plot_confusion_matrix(y_test, y_pred) -> None:
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5.5, 4.5))

    try:
        import seaborn as sns

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=RISK_NAMES,
            yticklabels=RISK_NAMES,
        )
    except ImportError:
        # Fallback to pure matplotlib if seaborn is not installed.
        ax = plt.gca()
        im = ax.imshow(cm, cmap="Blues")
        plt.colorbar(im)
        ax.set_xticks(range(len(RISK_NAMES)), labels=RISK_NAMES)
        ax.set_yticks(range(len(RISK_NAMES)), labels=RISK_NAMES)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Random Forest — risk_level confusion matrix")
    plt.tight_layout()
    path = os.path.join(OUTPUTS_DIR, "rf_confusion_matrix.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"[rf] Saved confusion matrix -> {path}")


if __name__ == "__main__":
    train_rf()
