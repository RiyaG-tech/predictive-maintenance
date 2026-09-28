"""
Preprocessing module for Predictive Maintenance System.
Handles data loading, validation, feature engineering, and pipeline creation.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Base raw sensor columns
RAW_NUMERIC_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

RAW_CATEGORICAL_FEATURES = ["Type"]

# Domain engineered features
ENGINEERED_FEATURES = [
    "Temp difference [K]",
    "Power [kW]",
    "Overstrain factor [min x Nm]",
    "Temp ratio",
]

ALL_NUMERIC_FEATURES = RAW_NUMERIC_FEATURES + ENGINEERED_FEATURES
TARGET_COLUMN = "Machine failure"
FAILURE_MODES = ["TWF", "HDF", "PWF", "OSF", "RNF"]
EXCLUDE_COLUMNS = ["UDI", "Product ID"] + FAILURE_MODES


def load_data(filepath: str) -> pd.DataFrame:
    """Load predictive maintenance dataset and clean basic formatting."""
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        raise FileNotFoundError(f"Could not load data file at {filepath}: {e}")

    # Strip column names of leading/trailing whitespaces
    df.columns = df.columns.str.strip()

    # Drop duplicate records if any
    df = df.drop_duplicates()

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute domain-informed engineering features:
    1. Temp difference [K] = Process temperature - Air temperature (critical for Heat Dissipation Failure HDF)
    2. Power [kW] = Rotational speed * Torque * (2 * pi / 60) / 1000 (critical for Power Failure PWF)
    3. Overstrain factor [min x Nm] = Tool wear * Torque (critical for Overstrain Failure OSF)
    4. Temp ratio = Process temperature / Air temperature
    """
    df_feat = df.copy()

    # Physical heat dissipation relationship
    df_feat["Temp difference [K]"] = (
        df_feat["Process temperature [K]"] - df_feat["Air temperature [K]"]
    )

    # Mechanical power (P = omega * tau = (2*pi*N/60) * Torque)
    df_feat["Power [kW]"] = (
        df_feat["Rotational speed [rpm]"] * df_feat["Torque [Nm]"] * (2 * np.pi / 60.0) / 1000.0
    )

    # Overstrain load product
    df_feat["Overstrain factor [min x Nm]"] = (
        df_feat["Tool wear [min]"] * df_feat["Torque [Nm]"]
    )

    # Thermal expansion ratio
    df_feat["Temp ratio"] = (
        df_feat["Process temperature [K]"] / df_feat["Air temperature [K]"]
    )

    return df_feat


def get_preprocessor():
    """
    Create a ColumnTransformer pipeline that scales numerical features
    and one-hot encodes categorical machine types.
    """
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,
        drop="first"  # Avoid dummy variable trap
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, ALL_NUMERIC_FEATURES),
            ("cat", categorical_transformer, RAW_CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    return preprocessor


def prepare_train_test_data(
    df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
):
    """
    Prepare feature matrix X and target y, and return stratified train/test split.
    Prevents data leakage by separating before fitting any transformations.
    """
    df_feat = engineer_features(df)

    # Drop target, IDs and failure mode detail columns from features
    feature_cols = [
        col for col in df_feat.columns
        if col not in EXCLUDE_COLUMNS and col != TARGET_COLUMN
    ]

    X = df_feat[feature_cols]
    y = df_feat[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test, feature_cols
