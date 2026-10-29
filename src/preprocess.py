"""
Data Preprocessing Pipeline
============================
Loads raw Excel data, cleans dirty categoricals, handles outliers using
physiologically valid ranges, creates derived features, performs a
time-based train / val / test split, and saves processed CSVs.

Usage
-----
    python src/preprocess.py [--params params.yaml]
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────
# Config
# ────────────────────────────────────────────────────────────────────

def load_params(path: str = "params.yaml") -> dict:
    """Load pipeline configuration from *params.yaml*."""
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# ────────────────────────────────────────────────────────────────────
# Data loading
# ────────────────────────────────────────────────────────────────────

def load_raw_data(params: dict) -> pd.DataFrame:
    """Read the raw Excel file specified in params."""
    raw_path = params["data"]["raw_path"]
    sheet = params["data"]["sheet"]
    logger.info("Loading %s  (sheet=%s)", raw_path, sheet)
    df = pd.read_excel(raw_path, sheet_name=sheet)
    logger.info("Loaded %d rows × %d columns", *df.shape)
    return df


# ────────────────────────────────────────────────────────────────────
# Categorical cleaning helpers
# ────────────────────────────────────────────────────────────────────

def clean_gender(series: pd.Series) -> pd.Series:
    """Standardise Gender to ``M`` / ``F``."""
    mapping: dict[str, str] = {}
    for val in series.dropna().unique():
        clean = str(val).strip().lower()
        if clean in ("m", "male"):
            mapping[val] = "M"
        elif clean in ("f", "female"):
            mapping[val] = "F"
        else:
            mapping[val] = val  # keep unknown as-is
    return series.map(mapping)


def clean_department(series: pd.Series) -> pd.Series:
    """Standardise Department to ``Ward`` / ``ICU`` / ``ER``."""
    mapping: dict[str, str] = {}
    for val in series.dropna().unique():
        clean = str(val).strip().lower()
        if clean == "ward":
            mapping[val] = "Ward"
        elif clean == "icu":
            mapping[val] = "ICU"
        elif clean in ("er", "emergency"):
            mapping[val] = "ER"
        else:
            mapping[val] = val
    return series.map(mapping)


def clean_insurance(series: pd.Series) -> pd.Series:
    """Standardise Insurance to ``Government`` / ``Private`` / ``Self-Pay``."""
    mapping: dict[str, str] = {}
    for val in series.dropna().unique():
        # Normalise separators + whitespace for robust matching
        clean = (
            str(val).strip().lower()
            .replace("-", "")
            .replace("_", "")
            .replace(" ", "")
        )
        if clean in ("government", "govt"):
            mapping[val] = "Government"
        elif clean in ("private", "priv"):
            mapping[val] = "Private"
        elif clean == "selfpay":
            mapping[val] = "Self-Pay"
        else:
            mapping[val] = val
    return series.map(mapping)


def clean_comorbidities(series: pd.Series) -> pd.Series:
    """Standardise Comorbidities categories."""
    mapping: dict[str, str] = {}
    for val in series.dropna().unique():
        clean = str(val).strip().lower().replace(" ", "_")
        if clean == "no_comorbidity":
            mapping[val] = "No_Comorbidity"
        elif clean == "hypertension":
            mapping[val] = "Hypertension"
        elif clean == "diabetes":
            mapping[val] = "Diabetes"
        elif clean == "copd":
            mapping[val] = "COPD"
        elif clean == "ckd":
            mapping[val] = "CKD"
        elif clean == "both":
            mapping[val] = "Both"
        else:
            mapping[val] = val
    return series.map(mapping)


def clean_icu_admitted(series: pd.Series) -> pd.Series:
    """Standardise ICU_Admitted to 'Yes' / 'No'."""
    mapping: dict[str, str] = {}
    for val in series.dropna().unique():
        clean = str(val).strip().lower()
        if clean in ("yes", "y", "1", "true"):
            mapping[val] = "Yes"
        elif clean in ("no", "n", "0", "false"):
            mapping[val] = "No"
        else:
            mapping[val] = str(val).strip().capitalize()
    return series.map(mapping).fillna("No")


def clean_blood_type(series: pd.Series) -> pd.Series:
    """Standardise Blood_Type to uppercase (e.g. A+, O-, etc.)."""
    return series.astype(str).str.strip().str.upper()


def clean_yes_no(series: pd.Series, default: str = "No") -> pd.Series:
    """Standardise binary flags to 'Yes' / 'No'."""
    mapping: dict[str, str] = {}
    for val in series.dropna().unique():
        clean = str(val).strip().lower()
        if clean in ("yes", "y", "1", "true"):
            mapping[val] = "Yes"
        elif clean in ("no", "n", "0", "false"):
            mapping[val] = "No"
        else:
            mapping[val] = str(val).strip().capitalize()
    return series.map(mapping).fillna(default)


def clean_status(series: pd.Series) -> pd.Series:
    """Standardise Status to 'Discharged' / 'Deceased'."""
    return series.astype(str).str.strip().str.capitalize()


def clean_discharge_status(series: pd.Series) -> pd.Series:
    """Standardise Discharge_Status to Title case ('Home', 'Rehab', 'Transfer', 'Expired')."""
    return series.astype(str).str.strip().str.capitalize()


def clean_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all categorical standardisation rules."""
    df = df.copy()
    if "Gender" in df.columns:
        df["Gender"] = clean_gender(df["Gender"])
    if "Department" in df.columns:
        df["Department"] = clean_department(df["Department"])
    if "Insurance" in df.columns:
        df["Insurance"] = clean_insurance(df["Insurance"])
    if "Comorbidities" in df.columns:
        df["Comorbidities"] = clean_comorbidities(df["Comorbidities"])
    if "ICU_Admitted" in df.columns:
        df["ICU_Admitted"] = clean_icu_admitted(df["ICU_Admitted"])
    if "Blood_Type" in df.columns:
        df["Blood_Type"] = clean_blood_type(df["Blood_Type"])
    if "Status" in df.columns:
        df["Status"] = clean_status(df["Status"])
    if "Discharge_Status" in df.columns:
        df["Discharge_Status"] = clean_discharge_status(df["Discharge_Status"])
    if "ICU_Transfer" in df.columns:
        df["ICU_Transfer"] = clean_yes_no(df["ICU_Transfer"])
    if "Mortality" in df.columns:
        df["Mortality"] = clean_yes_no(df["Mortality"])
    if "Readmission" in df.columns:
        df["Readmission"] = clean_yes_no(df["Readmission"])

    cols_to_log = [c for c in ["Gender", "Department", "Insurance", "Comorbidities", "ICU_Admitted", "Blood_Type", "Status", "Discharge_Status", "ICU_Transfer", "Mortality", "Readmission"] if c in df.columns]

    for col in cols_to_log:
        logger.info(
            "  %s → %d unique values: %s",
            col, df[col].nunique(), sorted(df[col].dropna().unique()),
        )
    return df


# ────────────────────────────────────────────────────────────────────
# Outlier / range handling
# ────────────────────────────────────────────────────────────────────

def apply_valid_ranges(df: pd.DataFrame, ranges: dict) -> pd.DataFrame:
    """Set values outside physiologically plausible ranges to ``NaN``.

    These are treated as data-entry errors, *not* outliers to drop.
    The imputer in the feature-engineering stage will fill them.
    """
    df = df.copy()
    total_clipped = 0
    for col, (lo, hi) in ranges.items():
        if col not in df.columns:
            continue
        mask = ~df[col].between(lo, hi) & df[col].notna()
        n_invalid = mask.sum()
        if n_invalid:
            logger.info("  %s: %d values outside [%s, %s] → NaN", col, n_invalid, lo, hi)
            df.loc[mask, col] = np.nan
            total_clipped += n_invalid
    logger.info("  Total values clipped to NaN: %d", total_clipped)
    return df


# ────────────────────────────────────────────────────────────────────
# Derived features
# ────────────────────────────────────────────────────────────────────

def encode_ordinal_features(df: pd.DataFrame, ordinal_cfg: dict) -> pd.DataFrame:
    """Ordinal-encode categorical columns using the mapping in *params.yaml*."""
    df = df.copy()
    for col, mapping in ordinal_cfg.items():
        if col not in df.columns:
            logger.warning("Column '%s' not found – skipping ordinal encoding", col)
            continue
        before_nulls = df[col].isna().sum()
        df[col] = df[col].map(mapping)
        new_nulls = df[col].isna().sum() - before_nulls
        if new_nulls > 0:
            logger.warning("  %s: %d values could not be mapped → NaN", col, new_nulls)
        logger.info("  Ordinal encoded %s  (%s)", col, mapping)
    return df


def create_comorbidity_flags(df: pd.DataFrame, flags_cfg: dict) -> pd.DataFrame:
    """Expand Comorbidities into binary 0/1 flag columns."""
    df = df.copy()
    comorb = df.get("Comorbidities", pd.Series(dtype=str))
    for flag_name, trigger_values in flags_cfg.items():
        df[flag_name] = comorb.isin(trigger_values).astype(int)
    logger.info("  Created %d comorbidity flags", len(flags_cfg))
    return df


# ────────────────────────────────────────────────────────────────────
# Time-based splitting
# ────────────────────────────────────────────────────────────────────

def time_based_split(
    df: pd.DataFrame,
    date_col: str,
    val_size: float,
    test_size: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data chronologically by *date_col*.

    Rows with a null date are assigned to the training set so they do
    not leak future information into validation / test.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    null_mask = df[date_col].isna()
    if null_mask.any():
        logger.info("  %d rows with null/unparseable %s → assigned to train", null_mask.sum(), date_col)

    df_dated = df.loc[~null_mask].sort_values(date_col).reset_index(drop=True)
    df_undated = df.loc[null_mask].reset_index(drop=True)

    n = len(df_dated)
    train_end = int(n * (1 - val_size - test_size))
    val_end = int(n * (1 - test_size))

    train = pd.concat([df_dated.iloc[:train_end], df_undated], ignore_index=True)
    val = df_dated.iloc[train_end:val_end].reset_index(drop=True)
    test = df_dated.iloc[val_end:].reset_index(drop=True)

    logger.info("  Split sizes → train=%d  val=%d  test=%d", len(train), len(val), len(test))
    return train, val, test


# ────────────────────────────────────────────────────────────────────
# Column selection
# ────────────────────────────────────────────────────────────────────

def select_columns(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Keep only the feature columns + target; drop everything else."""
    feat = params["features"]
    target_col = params["target"]["column"]

    keep: list[str] = []
    keep += feat["numeric"]
    keep += feat["categorical"]
    if "ordinal" in feat:
        keep += list(feat["ordinal"].keys())
    if "comorbidity_flags" in feat:
        keep += list(feat["comorbidity_flags"].keys())
    keep.append(target_col)

    # Guard against columns that don't exist (e.g. optional features)
    keep = [c for c in keep if c in df.columns]
    return df[keep]


# ────────────────────────────────────────────────────────────────────
# Main pipeline
# ────────────────────────────────────────────────────────────────────

def main(params_path: str = "params.yaml") -> None:
    params = load_params(params_path)

    # 1 – Load raw data
    df = load_raw_data(params)

    # 2 – Clean dirty categoricals
    logger.info("── Cleaning categoricals ──")
    df = clean_categoricals(df)

    # 3 – Apply valid ranges (entry-error detection)
    logger.info("── Applying valid ranges ──")
    df = apply_valid_ranges(df, params["valid_ranges"])

    # 4 – Ordinal encode (e.g. Acuity_Level: Low→1 … Critical→4)
    if "ordinal" in params["features"]:
        logger.info("── Ordinal encoding ──")
        df = encode_ordinal_features(df, params["features"]["ordinal"])

    # 5 – Comorbidity binary flags
    logger.info("── Creating comorbidity flags ──")
    df = create_comorbidity_flags(df, params["features"]["comorbidity_flags"])

    # 6 – Time-based split (needs Admission_Date, so do before column selection)
    logger.info("── Time-based split ──")
    train, val, test = time_based_split(
        df,
        date_col=params["split"]["date_column"],
        val_size=params["split"]["val_size"],
        test_size=params["split"]["test_size"],
    )

    # 7 – Select only feature + target columns
    train = select_columns(train, params)
    val = select_columns(val, params)
    test = select_columns(test, params)

    # 8 – Save
    out_dir = Path(params["data"]["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    train.to_csv(out_dir / "train.csv", index=False)
    val.to_csv(out_dir / "val.csv", index=False)
    test.to_csv(out_dir / "test.csv", index=False)

    logger.info("── Saved to %s ──", out_dir)
    logger.info("  train : %s", train.shape)
    logger.info("  val   : %s", val.shape)
    logger.info("  test  : %s", test.shape)


# ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess healthcare data")
    parser.add_argument("--params", default="params.yaml", help="Path to params.yaml")
    args = parser.parse_args()
    main(args.params)
