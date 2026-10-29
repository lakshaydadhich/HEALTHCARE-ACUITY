"""
Unit tests for preprocessing, feature engineering, and inference pipelines.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add src to python path
src_dir = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(src_dir))

from feature_engineering import (
    build_preprocessor,
    decode_target,
    encode_target,
    get_feature_lists,
)
from preprocess import (
    apply_valid_ranges,
    clean_categoricals,
    clean_comorbidities,
    clean_department,
    clean_gender,
    clean_insurance,
    create_comorbidity_flags,
    encode_ordinal_features,
    time_based_split,
)


def test_clean_gender():
    raw = pd.Series(["M", "Male", "m", "f", "Female", "F", "Other"])
    cleaned = clean_gender(raw)
    assert cleaned.iloc[0] == "M"
    assert cleaned.iloc[1] == "M"
    assert cleaned.iloc[2] == "M"
    assert cleaned.iloc[3] == "F"
    assert cleaned.iloc[4] == "F"
    assert cleaned.iloc[5] == "F"


def test_clean_department():
    raw = pd.Series(["ward", "WARD", "Ward", "icu", "ICU", "er", "Emergency", "ER"])
    cleaned = clean_department(raw)
    assert set(cleaned.unique()) == {"Ward", "ICU", "ER"}


def test_clean_insurance():
    raw = pd.Series(["Private", "priv", "Government", "Govt", "Self-Pay", "self pay", "selfpay"])
    cleaned = clean_insurance(raw)
    assert set(cleaned.unique()) == {"Private", "Government", "Self-Pay"}


def test_clean_comorbidities():
    raw = pd.Series(["hypertension", "Diabetes", "COPD", "ckd", "Both", "no comorbidity", "No_Comorbidity"])
    cleaned = clean_comorbidities(raw)
    assert "Hypertension" in cleaned.values
    assert "Diabetes" in cleaned.values
    assert "COPD" in cleaned.values
    assert "CKD" in cleaned.values
    assert "Both" in cleaned.values
    assert "No_Comorbidity" in cleaned.values


def test_apply_valid_ranges():
    df = pd.DataFrame({
        "Age": [-5, 25, 65, 999],
        "Mean_HR": [-20, 75, 110, 450],
    })
    ranges = {
        "Age": [0, 120],
        "Mean_HR": [20, 250],
    }
    cleaned = apply_valid_ranges(df, ranges)
    assert np.isnan(cleaned.loc[0, "Age"])
    assert cleaned.loc[1, "Age"] == 25
    assert np.isnan(cleaned.loc[3, "Age"])
    assert np.isnan(cleaned.loc[0, "Mean_HR"])
    assert cleaned.loc[1, "Mean_HR"] == 75
    assert np.isnan(cleaned.loc[3, "Mean_HR"])


def test_encode_ordinal_features():
    df = pd.DataFrame({"Acuity_Level": ["Low", "Medium", "High", "Critical"]})
    ordinal_cfg = {"Acuity_Level": {"Low": 1, "Medium": 2, "High": 3, "Critical": 4}}
    encoded = encode_ordinal_features(df, ordinal_cfg)
    assert list(encoded["Acuity_Level"]) == [1, 2, 3, 4]


def test_create_comorbidity_flags():
    df = pd.DataFrame({
        "Comorbidities": ["Hypertension", "Diabetes", "Both", "COPD", "CKD", "No_Comorbidity"]
    })
    flags_cfg = {
        "Has_Hypertension": ["Hypertension", "Both"],
        "Has_Diabetes": ["Diabetes", "Both"],
        "Has_COPD": ["COPD"],
        "Has_CKD": ["CKD"],
    }
    flagged = create_comorbidity_flags(df, flags_cfg)
    assert flagged.loc[0, "Has_Hypertension"] == 1
    assert flagged.loc[0, "Has_Diabetes"] == 0
    assert flagged.loc[2, "Has_Hypertension"] == 1
    assert flagged.loc[2, "Has_Diabetes"] == 1
    assert flagged.loc[3, "Has_COPD"] == 1
    assert flagged.loc[4, "Has_CKD"] == 1
    assert flagged.loc[5, "Has_Hypertension"] == 0


def test_target_encode_decode():
    classes = ["ROUTINE_CARE", "ENHANCED_MONITORING", "RESPIRATORY_SUPPORT", "SEPSIS_HOUR1", "CRITICAL_CARE"]
    labels = pd.Series(["ROUTINE_CARE", "CRITICAL_CARE", "SEPSIS_HOUR1"])
    encoded = encode_target(labels, classes)
    assert list(encoded) == [0, 4, 3]

    decoded = decode_target(encoded, classes)
    assert list(decoded) == ["ROUTINE_CARE", "CRITICAL_CARE", "SEPSIS_HOUR1"]


def test_preprocessor_pipeline():
    params = {
        "features": {
            "numeric": ["Age", "Mean_HR"],
            "categorical": ["Gender", "Department"],
            "ordinal": {"Acuity_Level": {"Low": 1, "Medium": 2, "High": 3, "Critical": 4}},
            "comorbidity_flags": {"Has_Hypertension": ["Hypertension", "Both"]},
        }
    }
    df = pd.DataFrame({
        "Age": [45.0, np.nan, 70.0],
        "Mean_HR": [80.0, 95.0, 110.0],
        "Acuity_Level": [1, 2, 3],
        "Has_Hypertension": [0, 1, 1],
        "Gender": ["M", "F", "M"],
        "Department": ["Ward", "ICU", "ER"],
    })
    preprocessor = build_preprocessor(params)
    transformed = preprocessor.fit_transform(df)
    assert transformed.shape[0] == 3
    # Check that no NaNs remain after imputation
    assert not np.isnan(transformed).any()
