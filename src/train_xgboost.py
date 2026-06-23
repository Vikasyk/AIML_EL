"""
train_xgboost.py
================
Train an **XGBoost regressor** to predict a road edge's ``delay_score`` (0.0-1.0)
from weather / accident features.

Deliverables (per the project spec)
-----------------------------------
* Hyperparameters: n_estimators=200, max_depth=5, learning_rate=0.1, subsample=0.8
* Evaluation: RMSE and R^2  (target R^2 > 0.75)
* Baseline comparison: vs a plain Linear Regression
* Explainability: SHAP summary plot  -> outputs/shap_xgboost.png
                  feature-importance bar chart -> outputs/xgb_importance.png
* Overfitting check: train RMSE vs test RMSE
* Saves the trained model -> models/xgboost_delay.pkl

Run:  python src/train_xgboost.py
"""

from __future__ import annotations

import os
import pickle

import matplotlib

matplotlib.use("Agg")  # headless backend (no display needed)
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor

from preprocess import FEATURES, load_processed

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
MODEL_PATH = os.path.join(MODELS_DIR, "xgboost_delay.pkl")


def rmse(y_true, y_pred) -> float:
    """Root mean squared error (version-independent)."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def train_xgboost(data: dict | None = None) -> XGBRegressor:
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    if data is None:
        data = load_processed()

    X_train, X_test = data["X_train"], data["X_test"]
    y_train, y_test = data["y_reg_train"], data["y_reg_test"]

    if data.get("is_synthetic"):
        print("[xgboost] NOTE: training on SYNTHETIC data (real CSV not found).")

    # --- Train the XGBoost regressor ------------------------------------- #
    model = XGBRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # --- Evaluate -------------------------------------------------------- #
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_rmse = rmse(y_train, train_pred)
    test_rmse = rmse(y_test, test_pred)
    test_r2 = r2_score(y_test, test_pred)

    print("\n========== XGBOOST (delay_score regression) ==========")
    print(f"Train RMSE : {train_rmse:.4f}")
    print(f"Test  RMSE : {test_rmse:.4f}   (target 0.10 - 0.18)")
    print(f"Test  R^2  : {test_r2:.4f}   (target > 0.75)")
    print(f"Overfit gap: {test_rmse - train_rmse:+.4f}  (test - train RMSE)")

    # --- Baseline: Linear Regression ------------------------------------ #
    baseline = LinearRegression().fit(X_train, y_train)
    base_pred = baseline.predict(X_test)
    print("\n--- Baseline: Linear Regression ---")
    print(f"Baseline Test RMSE : {rmse(y_test, base_pred):.4f}")
    print(f"Baseline Test R^2  : {r2_score(y_test, base_pred):.4f}")
    print(f"=> XGBoost improves R^2 by {test_r2 - r2_score(y_test, base_pred):+.4f}")
    print("======================================================")

    # --- Feature importance bar chart ----------------------------------- #
    _plot_feature_importance(model)

    # --- SHAP explainability -------------------------------------------- #
    _plot_shap(model, X_test)

    # --- Persist model --------------------------------------------------- #
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"[xgboost] Saved model -> {MODEL_PATH}")

    return model


def _plot_feature_importance(model: XGBRegressor) -> None:
    importances = model.feature_importances_
    order = np.argsort(importances)
    plt.figure(figsize=(7, 4))
    plt.barh(np.array(FEATURES)[order], importances[order], color="#3949ab")
    plt.xlabel("Importance (gain)")
    plt.title("XGBoost feature importance — delay_score")
    plt.tight_layout()
    path = os.path.join(OUTPUTS_DIR, "xgb_importance.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"[xgboost] Saved feature-importance chart -> {path}")


def _plot_shap(model: XGBRegressor, X_test) -> None:
    """SHAP summary plot; skipped gracefully if shap is not installed."""
    try:
        import shap
    except ImportError:
        print("[xgboost] shap not installed -> skipping SHAP plot "
              "(pip install shap to enable).")
        return

    try:
        sample = X_test.iloc[:500]
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(sample)
        shap.summary_plot(shap_values, sample, show=False)
        path = os.path.join(OUTPUTS_DIR, "shap_xgboost.png")
        plt.savefig(path, bbox_inches="tight", dpi=120)
        plt.close()
        print(f"[xgboost] Saved SHAP summary plot -> {path}")
    except Exception as exc:  # noqa: BLE001  (SHAP can be finicky across versions)
        print(f"[xgboost] SHAP plot failed ({exc}) -> skipping.")


if __name__ == "__main__":
    train_xgboost()
