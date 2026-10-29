"""
FastAPI Serving Application
============================
Provides real-time and batch REST API endpoints for Healthcare Acuity Bundle
predictions.

Endpoints
---------
    GET  /health       → API health check
    GET  /model-info   → Metadata about current loaded model & features
    POST /predict      → Single patient acuity prediction
    POST /predict-batch → Batch patient acuity predictions
"""

from contextlib import asynccontextmanager
import json
import logging
from pathlib import Path
from typing import Any, Optional, Union

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")
logger = logging.getLogger("api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_artifacts()
    yield


app = FastAPI(
    title="Healthcare Acuity Bundle Prediction Service",
    description="MLOps API service for predicting clinical acuity bundles.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global artifacts
MODEL_ARTIFACTS = {
    "model": None,
    "preprocessor": None,
    "params": None,
    "classes": [],
    "train_metrics": {},
    "eval_metrics": {},
}


class PatientRecord(BaseModel):
    Age: Optional[float] = Field(default=55.0, description="Patient age in years")
    Risk_Score: Optional[float] = Field(default=50.0, description="Admission risk score (0-100)")
    Mean_HR: Optional[float] = Field(default=80.0, description="Mean Heart Rate")
    Max_HR: Optional[float] = Field(default=95.0, description="Max Heart Rate")
    Std_HR: Optional[float] = Field(default=8.0, description="Std dev of Heart Rate")
    Mean_SpO2: Optional[float] = Field(default=97.0, description="Mean SpO2 percentage")
    Min_SpO2: Optional[float] = Field(default=94.0, description="Minimum SpO2 percentage")
    Mean_BP_Sys: Optional[float] = Field(default=120.0, description="Mean Systolic Blood Pressure")
    Min_BP_Sys: Optional[float] = Field(default=105.0, description="Minimum Systolic Blood Pressure")
    Max_Temp: Optional[float] = Field(default=37.0, description="Maximum Temperature in Celsius")
    Mean_RR: Optional[float] = Field(default=18.0, description="Mean Respiration Rate")
    Max_RR: Optional[float] = Field(default=22.0, description="Max Respiration Rate")
    Mean_GCS: Optional[float] = Field(default=15.0, description="Glasgow Coma Scale Mean (3-15)")
    Min_GCS: Optional[float] = Field(default=15.0, description="Glasgow Coma Scale Min (3-15)")
    Hypoxia_Episodes: Optional[float] = Field(default=0.0, description="Number of hypoxia episodes")
    Tachycardia_Episodes: Optional[float] = Field(default=0.0, description="Number of tachycardia episodes")
    Vitals_Count: Optional[float] = Field(default=10.0, description="Number of vital records taken")
    LOS: Optional[float] = Field(default=5.0, description="Length of Stay in days")

    Gender: Optional[str] = Field(default="M", description="Gender (M, F, Male, Female)")
    Department: Optional[str] = Field(default="Ward", description="Department (Ward, ICU, ER)")
    Diagnosis: Optional[str] = Field(default="Infection", description="Primary admission diagnosis")
    Insurance: Optional[str] = Field(default="Private", description="Insurance type (Private, Government, Self-Pay)")
    Blood_Type: Optional[str] = Field(default="O+", description="Blood Type (A+, O-, etc.)")
    ICU_Admitted: Optional[str] = Field(default="No", description="ICU Admitted flag (Yes, No)")
    Acuity_Level: Optional[Union[str, int]] = Field(default="Medium", description="Initial Acuity Level (Low, Medium, High, Critical)")
    Comorbidities: Optional[str] = Field(default="No_Comorbidity", description="Comorbidities (e.g. Hypertension, Diabetes, Both)")
    Status: Optional[str] = Field(default="Discharged", description="Patient status (Discharged, Deceased)")
    Discharge_Status: Optional[str] = Field(default="Home", description="Discharge status (Home, Rehab, Transfer, Expired)")
    ICU_Transfer: Optional[str] = Field(default="No", description="ICU Transfer flag (Yes, No)")
    Mortality: Optional[str] = Field(default="No", description="Mortality flag (Yes, No)")
    Readmission: Optional[str] = Field(default="No", description="Readmission flag (Yes, No)")


class PredictionResponse(BaseModel):
    predicted_acuity_bundle: str
    predicted_code: int
    probabilities: dict[str, float]


class BatchPredictionRequest(BaseModel):
    patients: list[PatientRecord]


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]


def clean_and_prepare_input(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Apply necessary cleaning and feature expansion so raw inputs match the preprocessor schema."""
    df = df.copy()

    # Clean gender
    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].astype(str).str.strip().str.lower().map({
            "m": "M", "male": "M", "f": "F", "female": "F"
        }).fillna(df["Gender"])

    # Clean department
    if "Department" in df.columns:
        df["Department"] = df["Department"].astype(str).str.strip().str.lower().map({
            "ward": "Ward", "icu": "ICU", "er": "ER", "emergency": "ER"
        }).fillna(df["Department"])

    # Clean insurance
    if "Insurance" in df.columns:
        clean_ins = df["Insurance"].astype(str).str.strip().str.lower().str.replace("-", "").str.replace("_", "").str.replace(" ", "")
        df["Insurance"] = clean_ins.map({
            "government": "Government", "govt": "Government",
            "private": "Private", "priv": "Private",
            "selfpay": "Self-Pay"
        }).fillna(df["Insurance"])

    # Clean Blood_Type
    if "Blood_Type" in df.columns:
        df["Blood_Type"] = df["Blood_Type"].astype(str).str.strip().str.upper()

    # Clean yes/no flags
    yes_no_cols = ["ICU_Admitted", "ICU_Transfer", "Mortality", "Readmission"]
    for col in yes_no_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.capitalize().map({
                "Yes": "Yes", "Y": "Yes", "1": "Yes", "True": "Yes",
                "No": "No", "N": "No", "0": "No", "False": "No"
            }).fillna("No")

    if "Status" in df.columns:
        df["Status"] = df["Status"].astype(str).str.strip().str.capitalize()

    if "Discharge_Status" in df.columns:
        df["Discharge_Status"] = df["Discharge_Status"].astype(str).str.strip().str.capitalize()

    # Ordinal encode Acuity_Level
    ordinal_map = params.get("features", {}).get("ordinal", {}).get("Acuity_Level", {
        "Low": 1, "Medium": 2, "High": 3, "Critical": 4
    })
    if "Acuity_Level" in df.columns:
        df["Acuity_Level"] = df["Acuity_Level"].map(lambda x: ordinal_map.get(str(x).capitalize(), x))
        df["Acuity_Level"] = pd.to_numeric(df["Acuity_Level"], errors="coerce")

    # Expand comorbidity flags
    flags_cfg = params.get("features", {}).get("comorbidity_flags", {
        "Has_Hypertension": ["Hypertension", "Both"],
        "Has_Diabetes": ["Diabetes", "Both"],
        "Has_COPD": ["COPD"],
        "Has_CKD": ["CKD"],
    })
    comorb = df.get("Comorbidities", pd.Series(dtype=str)).astype(str).str.strip()
    for flag_name, trigger_vals in flags_cfg.items():
        df[flag_name] = comorb.isin(trigger_vals).astype(int)

    # Ensure all required preprocessor columns exist in df (fill missing with NaN for imputer)
    feat_cfg = params.get("features", {})
    expected_cols = (
        feat_cfg.get("numeric", []) +
        feat_cfg.get("categorical", []) +
        list(feat_cfg.get("ordinal", {}).keys()) +
        list(feat_cfg.get("comorbidity_flags", {}).keys())
    )
    for col in expected_cols:
        if col not in df.columns:
            df[col] = np.nan

    return df


def load_artifacts():
    """Load model, preprocessor, and params into memory."""
    import yaml

    params_path = Path("params.yaml")
    if params_path.exists():
        with open(params_path, encoding="utf-8") as fh:
            MODEL_ARTIFACTS["params"] = yaml.safe_load(fh)
    else:
        MODEL_ARTIFACTS["params"] = {}

    params = MODEL_ARTIFACTS["params"]
    MODEL_ARTIFACTS["classes"] = params.get("target", {}).get("classes", [
        "ROUTINE_CARE", "ENHANCED_MONITORING", "RESPIRATORY_SUPPORT", "SEPSIS_HOUR1", "CRITICAL_CARE"
    ])

    model_path = Path(params.get("data", {}).get("model_path", "models/best_model.joblib"))
    prep_path = Path(params.get("data", {}).get("preprocessor_path", "models/preprocessor.joblib"))

    if prep_path.exists():
        MODEL_ARTIFACTS["preprocessor"] = joblib.load(prep_path)
        logger.info("Loaded preprocessor from %s", prep_path)
    else:
        logger.warning("Preprocessor not found at %s", prep_path)

    if model_path.exists():
        MODEL_ARTIFACTS["model"] = joblib.load(model_path)
        logger.info("Loaded model from %s", model_path)
    else:
        logger.warning("Model not found at %s", model_path)

    # Load metrics if available
    train_metrics_path = Path("models/train_metrics.json")
    if train_metrics_path.exists():
        with open(train_metrics_path, encoding="utf-8") as fh:
            MODEL_ARTIFACTS["train_metrics"] = json.load(fh)

    eval_metrics_path = Path("models/eval_metrics.json")
    if eval_metrics_path.exists():
        with open(eval_metrics_path, encoding="utf-8") as fh:
            MODEL_ARTIFACTS["eval_metrics"] = json.load(fh)


# Load artifacts at import time
load_artifacts()


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": MODEL_ARTIFACTS["model"] is not None,
        "preprocessor_loaded": MODEL_ARTIFACTS["preprocessor"] is not None,
    }


@app.get("/model-info")
def model_info():
    if MODEL_ARTIFACTS["model"] is None:
        raise HTTPException(status_code=503, detail="Model is not yet loaded.")
    return {
        "model_type": type(MODEL_ARTIFACTS["model"]).__name__,
        "target_classes": MODEL_ARTIFACTS["classes"],
        "train_metrics": MODEL_ARTIFACTS["train_metrics"],
        "eval_metrics": MODEL_ARTIFACTS["eval_metrics"],
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_patient(patient: PatientRecord):
    if MODEL_ARTIFACTS["model"] is None or MODEL_ARTIFACTS["preprocessor"] is None:
        raise HTTPException(status_code=503, detail="Model or Preprocessor not available.")

    df = pd.DataFrame([patient.model_dump()])
    df_prepared = clean_and_prepare_input(df, MODEL_ARTIFACTS["params"])

    try:
        X_trans = MODEL_ARTIFACTS["preprocessor"].transform(df_prepared)
        pred_code = int(MODEL_ARTIFACTS["model"].predict(X_trans)[0])
        pred_class = MODEL_ARTIFACTS["classes"][pred_code]

        probs = {}
        if hasattr(MODEL_ARTIFACTS["model"], "predict_proba"):
            proba_arr = MODEL_ARTIFACTS["model"].predict_proba(X_trans)[0]
            for idx, cls in enumerate(MODEL_ARTIFACTS["classes"]):
                if idx < len(proba_arr):
                    probs[cls] = float(proba_arr[idx])

        return PredictionResponse(
            predicted_acuity_bundle=pred_class,
            predicted_code=pred_code,
            probabilities=probs,
        )
    except Exception as exc:
        logger.error("Prediction failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}")


@app.post("/predict-batch", response_model=BatchPredictionResponse)
def predict_batch(request: BatchPredictionRequest):
    if MODEL_ARTIFACTS["model"] is None or MODEL_ARTIFACTS["preprocessor"] is None:
        raise HTTPException(status_code=503, detail="Model or Preprocessor not available.")

    records = [p.model_dump() for p in request.patients]
    df = pd.DataFrame(records)
    df_prepared = clean_and_prepare_input(df, MODEL_ARTIFACTS["params"])

    try:
        X_trans = MODEL_ARTIFACTS["preprocessor"].transform(df_prepared)
        preds = MODEL_ARTIFACTS["model"].predict(X_trans)

        probs_matrix = None
        if hasattr(MODEL_ARTIFACTS["model"], "predict_proba"):
            probs_matrix = MODEL_ARTIFACTS["model"].predict_proba(X_trans)

        responses = []
        for i, code in enumerate(preds):
            pred_code = int(code)
            pred_class = MODEL_ARTIFACTS["classes"][pred_code]
            probs = {}
            if probs_matrix is not None:
                for idx, cls in enumerate(MODEL_ARTIFACTS["classes"]):
                    if idx < probs_matrix.shape[1]:
                        probs[cls] = float(probs_matrix[i, idx])

            responses.append(
                PredictionResponse(
                    predicted_acuity_bundle=pred_class,
                    predicted_code=pred_code,
                    probabilities=probs,
                )
            )

        return BatchPredictionResponse(predictions=responses)
    except Exception as exc:
        logger.error("Batch prediction failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Batch inference error: {exc}")
