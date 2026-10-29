"""
Feature Engineering Module
===========================
Provides reusable functions for building the sklearn preprocessing
pipeline (``ColumnTransformer``) that is shared by the training,
evaluation, and serving stages.

This module is *imported* by other scripts — it is not executed
directly.
"""

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────
# Config
# ────────────────────────────────────────────────────────────────────

def load_params(path: str = "params.yaml") -> dict:
    """Load pipeline configuration."""
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# ────────────────────────────────────────────────────────────────────
# Feature lists
# ────────────────────────────────────────────────────────────────────

def get_feature_lists(params: dict) -> tuple[list[str], list[str]]:
    """Return ``(numeric_cols, categorical_cols)`` for the ColumnTransformer.

    * **numeric_cols** includes the raw numeric columns, ordinal-encoded
      columns, and comorbidity binary flags.
    * **categorical_cols** is the list of string-valued features that need
      one-hot encoding.
    """
    feat = params["features"]

    numeric: list[str] = list(feat["numeric"])

    # Ordinal-encoded columns have already been turned into integers
    # during preprocessing — treat them as numeric.
    if "ordinal" in feat:
        numeric += list(feat["ordinal"].keys())

    # Comorbidity flags are binary 0/1 — treat as numeric.
    if "comorbidity_flags" in feat:
        numeric += list(feat["comorbidity_flags"].keys())

    categorical: list[str] = list(feat["categorical"])

    return numeric, categorical


# ────────────────────────────────────────────────────────────────────
# Preprocessor construction
# ────────────────────────────────────────────────────────────────────

def build_preprocessor(params: dict) -> ColumnTransformer:
    """Build a ``ColumnTransformer`` that handles missing values,
    scaling, and encoding in a single object.

    Numeric pipeline
        ``SimpleImputer(strategy='median')`` → ``StandardScaler``

    Categorical pipeline
        ``SimpleImputer(strategy='most_frequent')`` → ``OneHotEncoder``
    """
    numeric_cols, categorical_cols = get_feature_lists(params)

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )
    return preprocessor


# ────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────

def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Extract human-readable feature names from a *fitted*
    ``ColumnTransformer``."""
    return list(preprocessor.get_feature_names_out())


def prepare_data(
    df: pd.DataFrame,
    target_col: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """Split a DataFrame into ``(X, y)``."""
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


def encode_target(y: pd.Series, classes: list[str]) -> np.ndarray:
    """Map target string labels to integer codes using the order
    defined in *params.yaml*."""
    label_to_int = {label: idx for idx, label in enumerate(classes)}
    return y.map(label_to_int).values


def decode_target(y_enc: np.ndarray, classes: list[str]) -> np.ndarray:
    """Map integer codes back to string labels."""
    return np.array([classes[int(i)] for i in y_enc])
