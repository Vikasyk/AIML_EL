"""
preprocess.py
=============
Data cleaning and feature engineering for the Smart Delivery & Traffic
Management System.

Responsibilities
----------------
1. Load the Kaggle "US Accidents (2016-2023)" dataset from ``data/US_Accidents.csv``.
   * If the CSV is not present, fall back to a **clearly-flagged synthetic
     sample** so the rest of the project is runnable end-to-end immediately.
2. Drop nulls in key columns and encode categorical features with LabelEncoder.
3. Engineer the two ML targets:
       delay_score = (Severity - 1) / 3.0      (regression target, 0.0 .. 1.0)
       risk_level  = Low(0) / Medium(1) / High(2)   (classification target)
4. Scale numerical features with StandardScaler (mainly so the Linear/Logistic
   regression baselines are fair; the tree models are scale-invariant).
5. Produce an 80/20 stratified train/test split and persist everything to
   ``data/processed.pkl`` for the training scripts.

This module is independently runnable:  ``python src/preprocess.py``
It is also imported by ``train_xgboost.py`` and ``train_rf.py``.
"""

from __future__ import annotations

import os
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# --------------------------------------------------------------------------- #
# Paths and feature definitions
# --------------------------------------------------------------------------- #
HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_CSV = os.path.join(DATA_DIR, "US_Accidents.csv")
PROCESSED_PKL = os.path.join(DATA_DIR, "processed.pkl")

# Numerical features (will be scaled with StandardScaler).
NUMERIC_FEATURES = [
    "Temperature(F)",
    "Visibility(mi)",
    "Wind_Speed(mph)",
    "Precipitation(in)",
]
# Categorical features (LabelEncoded to integers).
CATEGORICAL_FEATURES = [
    "Weather_Condition",
    "Sunrise_Sunset",
]
# Full ordered feature list used by BOTH ML models (and by the pipeline at
# inference time — keep this order stable).
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Columns we need from the raw CSV.
RAW_COLUMNS = ["Severity"] + NUMERIC_FEATURES + CATEGORICAL_FEATURES

RISK_NAMES = ["Low", "Medium", "High"]


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_us_accidents(path: str = RAW_CSV, nrows: int | None = 200_000) -> pd.DataFrame | None:
    """Load the real US Accidents CSV if it exists, else return ``None``.

    Only the columns we need are read, and ``nrows`` caps the row count so the
    ~1 GB file is manageable on a laptop (set ``nrows=None`` for the full file).
    """
    if not os.path.exists(path):
        return None
    print(f"[preprocess] Loading REAL dataset from {path} (nrows={nrows}) ...")
    # usecols keeps memory low; some columns may be missing in older dumps.
    df = pd.read_csv(path, usecols=lambda c: c in RAW_COLUMNS, nrows=nrows)
    return df


def make_synthetic(n: int = 20_000, seed: int = 42) -> pd.DataFrame:
    """Generate a **synthetic** stand-in for the US Accidents dataset.

    The synthetic data deliberately embeds a learnable relationship between the
    weather features and accident severity so the ML models reach realistic
    accuracy:  low visibility, heavy precipitation, fog/snow and night-time all
    push severity up.  This is NOT real data — it exists only so the project is
    runnable before the Kaggle CSV is downloaded.
    """
    rng = np.random.default_rng(seed)

    temperature = rng.normal(62, 18, n)
    visibility = np.clip(rng.normal(8.0, 3.0, n), 0.1, 10.0)
    wind = np.clip(rng.normal(8.0, 5.0, n), 0.0, 45.0)
    # Precipitation is zero most of the time, with an occasional spike.
    precipitation = np.where(rng.random(n) < 0.25, rng.exponential(0.08, n), 0.0)

    weather_choices = ["Clear", "Cloudy", "Rain", "Snow", "Fog"]
    weather_probs = [0.42, 0.26, 0.18, 0.07, 0.07]
    weather = rng.choice(weather_choices, size=n, p=weather_probs)

    daynight = rng.choice(["Day", "Night"], size=n, p=[0.62, 0.38])

    # Latent severity driver (continuous), then bucket into 1..4.
    # Hazardous weather, poor visibility, precipitation and darkness all raise
    # severity.  Noise is kept moderate so the weather features remain a strong
    # predictor (yielding the report's expected R^2 ~0.75-0.88 / accuracy ~80%+).
    latent = (
        1.4 * (weather == "Rain").astype(float)
        + 2.2 * (weather == "Snow").astype(float)
        + 1.8 * (weather == "Fog").astype(float)
        + 1.5 * (precipitation > 0.05).astype(float)
        + 1.2 * (visibility < 4.0).astype(float)
        + 0.6 * (daynight == "Night").astype(float)
        + 0.5 * (wind > 15).astype(float)
        + rng.normal(0.0, 0.22, n)  # moderate noise -> realistic, not perfect
    )
    # Map latent -> severity 1..4 by quantiles, giving a balanced class spread:
    #   Low(1) ~45%, Medium(2) ~30%, High(3..4) ~25%.
    q = np.quantile(latent, [0.45, 0.75, 0.92])
    severity = np.ones(n, dtype=int)
    severity[latent > q[0]] = 2
    severity[latent > q[1]] = 3
    severity[latent > q[2]] = 4

    return pd.DataFrame(
        {
            "Severity": severity,
            "Temperature(F)": temperature,
            "Visibility(mi)": visibility,
            "Wind_Speed(mph)": wind,
            "Precipitation(in)": precipitation,
            "Weather_Condition": weather,
            "Sunrise_Sunset": daynight,
        }
    )


# --------------------------------------------------------------------------- #
# Feature engineering
# --------------------------------------------------------------------------- #
def build_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``delay_score`` (regression) and ``risk_level`` (classification)."""
    df = df.copy()
    df["delay_score"] = (df["Severity"] - 1) / 3.0
    df["risk_level"] = df["Severity"].apply(
        lambda s: 0 if s == 1 else (1 if s == 2 else 2)
    )
    return df


def encode_categoricals(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
    """LabelEncode each categorical column; return encoders for reuse."""
    df = df.copy()
    encoders: dict[str, LabelEncoder] = {}
    for col in CATEGORICAL_FEATURES:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    return df, encoders


def get_processed_data(
    nrows: int | None = 200_000,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """Full preprocessing pipeline -> dictionary of train/test arrays + artefacts.

    Returns a dict with keys:
        X, y_reg, y_clf            : full scaled feature frame + both targets
        X_train, X_test            : scaled feature splits
        y_reg_train, y_reg_test    : regression target splits
        y_clf_train, y_clf_test    : classification target splits
        scaler                     : fitted StandardScaler (numeric cols)
        encoders                   : dict of fitted LabelEncoders
        features                   : ordered feature-name list
        is_synthetic               : True if synthetic fallback was used
    """
    df = load_us_accidents(nrows=nrows)
    is_synthetic = df is None
    if is_synthetic:
        print(
            "[preprocess] *** REAL DATASET NOT FOUND -> USING SYNTHETIC SAMPLE ***\n"
            "[preprocess] Download data/US_Accidents.csv from Kaggle for real results.\n"
            "[preprocess] (kaggle.com/datasets/sobhanmoosavi/us-accidents)"
        )
        df = make_synthetic()

    df = df[RAW_COLUMNS].copy()

    # In the real US Accidents data, Precipitation(in) and Wind_Speed(mph) are
    # NaN in the vast majority of rows. Dropping those would discard ~90% of the
    # dataset, so instead treat a missing precipitation as "no recorded rainfall"
    # (0.0) and a missing wind speed as the column median. Then drop nulls only
    # in the genuinely essential columns.
    if "Precipitation(in)" in df:
        df["Precipitation(in)"] = df["Precipitation(in)"].fillna(0.0)
    if "Wind_Speed(mph)" in df:
        df["Wind_Speed(mph)"] = df["Wind_Speed(mph)"].fillna(
            df["Wind_Speed(mph)"].median()
        )
    essential = ["Severity", "Temperature(F)", "Visibility(mi)",
                 "Weather_Condition", "Sunrise_Sunset"]
    df = df.dropna(subset=essential)

    # Targets + categorical encoding.
    df = build_targets(df)
    df, encoders = encode_categoricals(df)

    # Assemble feature matrix and scale numeric columns.
    X = df[FEATURES].copy().astype(float)
    scaler = StandardScaler()
    X[NUMERIC_FEATURES] = scaler.fit_transform(X[NUMERIC_FEATURES])

    y_reg = df["delay_score"].values
    y_clf = df["risk_level"].values

    # Stratified 80/20 split (stratify on the risk class so all 3 classes
    # appear in both splits).  We split indices once so the regression and
    # classification splits stay aligned.
    idx = np.arange(len(X))
    idx_train, idx_test = train_test_split(
        idx, test_size=test_size, random_state=random_state, stratify=y_clf
    )

    out = {
        "X": X,
        "y_reg": y_reg,
        "y_clf": y_clf,
        "X_train": X.iloc[idx_train].reset_index(drop=True),
        "X_test": X.iloc[idx_test].reset_index(drop=True),
        "y_reg_train": y_reg[idx_train],
        "y_reg_test": y_reg[idx_test],
        "y_clf_train": y_clf[idx_train],
        "y_clf_test": y_clf[idx_test],
        "scaler": scaler,
        "encoders": encoders,
        "features": FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "is_synthetic": is_synthetic,
    }
    return out


def save_processed(data: dict, path: str = PROCESSED_PKL) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f)
    print(f"[preprocess] Saved processed data -> {path}")


def load_processed(path: str = PROCESSED_PKL) -> dict:
    """Load previously-saved processed data, or compute + save it if missing."""
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    data = get_processed_data()
    save_processed(data, path)
    return data


# --------------------------------------------------------------------------- #
# CLI / smoke test
# --------------------------------------------------------------------------- #
def main() -> None:
    data = get_processed_data()
    save_processed(data)

    print("\n========== PREPROCESS SUMMARY ==========")
    print(f"Source            : {'SYNTHETIC' if data['is_synthetic'] else 'REAL US Accidents'}")
    print(f"Total rows        : {len(data['X']):,}")
    print(f"Features          : {data['features']}")
    print(f"Train / Test rows : {len(data['X_train']):,} / {len(data['X_test']):,}")
    print(f"delay_score range : [{data['y_reg'].min():.3f}, {data['y_reg'].max():.3f}]")

    # Class balance for the classification target.
    unique, counts = np.unique(data["y_clf"], return_counts=True)
    print("Risk class balance:")
    for cls, cnt in zip(unique, counts):
        print(f"    {RISK_NAMES[cls]:<7} : {cnt:>7,}  ({cnt / len(data['y_clf']):.1%})")
    print("========================================")


if __name__ == "__main__":
    main()
