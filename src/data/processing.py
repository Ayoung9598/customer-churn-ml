from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer


@dataclass
class DatasetSplits:
    X_train: pd.DataFrame
    X_valid: pd.DataFrame
    y_train: pd.Series
    y_valid: pd.Series


def _normalize_binary_target(y: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(y):
        # Already numeric; coerce to 0/1 if needed
        return y.astype(int)

    mapping_true = {"yes", "true", "1", "y", "t"}
    mapping_false = {"no", "false", "0", "n", "f"}

    def _map(val):
        if pd.isna(val):
            return np.nan
        s = str(val).strip().lower()
        if s in mapping_true:
            return 1
        if s in mapping_false:
            return 0
        return val

    mapped = y.map(_map)
    # If still non-numeric values exist, try factorization with positive class heuristics
    if not pd.api.types.is_numeric_dtype(mapped):
        uniques = pd.Series(mapped.dropna().unique()).astype(str).str.lower()
        # Heuristic: labels like 'yes'/'churn' treated as 1
        pos_like = {"yes", "true", "churn", "positive", "1"}
        if len(uniques) == 2 and any(u in pos_like for u in uniques):
            positive_label = next((u for u in uniques if u in pos_like), uniques.iloc[0])
            mapped = mapped.astype(str).str.lower().eq(positive_label).astype(int)
        else:
            # Fallback: first label -> 0, second -> 1
            labels = list(uniques)
            if len(labels) == 2:
                mapped = mapped.astype(str).str.lower().map({labels[0]: 0, labels[1]: 1}).astype(int)
    return mapped


def load_csv_dataset(path: str, target_column: str) -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset")
    X = df.drop(columns=[target_column])
    y = df[target_column]
    y = _normalize_binary_target(y)
    return X, y


def coerce_numeric_frame(X):
    # Convert problematic strings (e.g., ' ') to NaN so imputers can handle them
    if isinstance(X, pd.DataFrame):
        return X.apply(pd.to_numeric, errors="coerce")
    # Fallback for array-like
    return pd.to_numeric(X, errors="coerce")


def create_preprocessor(
    numeric_features: List[str],
    categorical_features: List[str],
) -> ColumnTransformer:
    numeric_transformer = Pipeline(
        steps=[
            ("coerce", FunctionTransformer(coerce_numeric_frame, feature_names_out="one-to-one")),
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=False)),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    return preprocessor


def train_valid_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
) -> DatasetSplits:
    stratify_labels = y if stratify else None
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify_labels
    )
    return DatasetSplits(X_train=X_train, X_valid=X_valid, y_train=y_train, y_valid=y_valid)
