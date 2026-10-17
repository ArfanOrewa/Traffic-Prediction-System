"""
preprocess.py — Feature engineering for the traffic dataset.

Turns the raw CSV into a model-ready feature matrix:
  - engineered flags: is_peak_hour, is_weekend
  - one-hot encoding: day_of_week, weather, road_id
  - median / most-frequent imputation for any missing values
  - duplicate removal

The ColumnTransformer built here is saved inside the model pipeline, so
training and inference always apply identical transformations.
"""

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

DATA_PATH = Path(__file__).resolve().parent / "data" / "traffic_data.csv"

PEAK_HOURS = {8, 9, 10, 17, 18, 19, 20}
WEEKEND = {"Saturday", "Sunday"}

# Raw columns fed into the preprocessor (post feature-engineering).
NUMERIC_FEATURES = ["hour", "is_peak_hour", "is_weekend",
                    "is_holiday", "vehicle_count", "avg_speed_kmph"]
CATEGORICAL_FEATURES = ["day_of_week", "weather", "road_id"]
TARGET = "congestion_level"


def load_dataframe(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the traffic CSV with parsed timestamps."""
    df = pd.read_csv(path, parse_dates=["timestamp"])
    return df


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add is_peak_hour / is_weekend flags. Returns a copy."""
    df = df.copy()
    df["is_peak_hour"] = df["hour"].isin(PEAK_HOURS).astype(int)
    df["is_weekend"] = df["day_of_week"].isin(WEEKEND).astype(int)
    return df


def build_preprocessor() -> ColumnTransformer:
    """ColumnTransformer: impute + one-hot categoricals, impute numerics."""
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, NUMERIC_FEATURES),
        ("cat", categorical_pipe, CATEGORICAL_FEATURES),
    ])


def prepare(df: pd.DataFrame):
    """
    Full preparation: engineer features, drop duplicates,
    return (X_raw, y) where X_raw uses RAW feature columns.
    """
    df = add_engineered_features(df)
    df = df.drop_duplicates().reset_index(drop=True)
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]
    return X, y


def build_typical_conditions(df: pd.DataFrame) -> dict:
    """
    Typical traffic per (road_id, hour, is_weekend): median vehicle_count
    and avg_speed_kmph. Used by the router to predict congestion for a
    future time slot where no live sensor reading exists.
    Returns a nested dict: road_id -> hour -> is_weekend -> {vehicle_count, avg_speed_kmph}.
    """
    df = add_engineered_features(df)
    grouped = (df.groupby(["road_id", "hour", "is_weekend"])
                 .agg(vehicle_count=("vehicle_count", "median"),
                      avg_speed_kmph=("avg_speed_kmph", "median")))
    typical = {}
    for (road_id, hour, is_weekend), row in grouped.iterrows():
        typical.setdefault(road_id, {}).setdefault(int(hour), {})[int(is_weekend)] = {
            "vehicle_count": float(row["vehicle_count"]),
            "avg_speed_kmph": float(row["avg_speed_kmph"]),
        }
    return typical
