"""
API Integration Tests for FastAPI Serving Service
"""

import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

src_dir = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(src_dir))

from api.app import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["preprocessor_loaded"] is True


def test_model_info_endpoint():
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert "model_type" in data
    assert "target_classes" in data
    assert len(data["target_classes"]) == 5


def test_single_prediction_endpoint():
    sample_patient = {
        "Age": 68.0,
        "Risk_Score": 75.0,
        "Mean_HR": 105.0,
        "Max_HR": 130.0,
        "Std_HR": 12.0,
        "Mean_SpO2": 91.0,
        "Min_SpO2": 88.0,
        "Mean_BP_Sys": 145.0,
        "Min_BP_Sys": 120.0,
        "Max_Temp": 38.5,
        "Mean_RR": 24.0,
        "Max_RR": 28.0,
        "Mean_GCS": 13.0,
        "Min_GCS": 12.0,
        "Hypoxia_Episodes": 2.0,
        "Tachycardia_Episodes": 3.0,
        "Vitals_Count": 18.0,
        "Gender": "Male",
        "Department": "ER",
        "Diagnosis": "Infection",
        "Insurance": "Government",
        "Acuity_Level": "High",
        "Comorbidities": "Hypertension",
    }
    response = client.post("/predict", json=sample_patient)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_acuity_bundle" in data
    assert "predicted_code" in data
    assert "probabilities" in data
    assert data["predicted_acuity_bundle"] in [
        "ROUTINE_CARE", "ENHANCED_MONITORING", "RESPIRATORY_SUPPORT", "SEPSIS_HOUR1", "CRITICAL_CARE"
    ]
    # Sum of probabilities should be close to 1
    total_prob = sum(data["probabilities"].values())
    assert 0.99 <= total_prob <= 1.01


def test_batch_prediction_endpoint():
    sample_patients = {
        "patients": [
            {
                "Age": 45.0,
                "Risk_Score": 25.0,
                "Mean_HR": 72.0,
                "Max_HR": 85.0,
                "Std_HR": 5.0,
                "Mean_SpO2": 99.0,
                "Min_SpO2": 98.0,
                "Mean_BP_Sys": 115.0,
                "Min_BP_Sys": 110.0,
                "Max_Temp": 36.8,
                "Mean_RR": 16.0,
                "Max_RR": 18.0,
                "Mean_GCS": 15.0,
                "Min_GCS": 15.0,
                "Hypoxia_Episodes": 0.0,
                "Tachycardia_Episodes": 0.0,
                "Vitals_Count": 8.0,
                "Gender": "Female",
                "Department": "Ward",
                "Diagnosis": "Observation",
                "Insurance": "Private",
                "Acuity_Level": "Low",
                "Comorbidities": "No_Comorbidity",
            },
            {
                "Age": 82.0,
                "Risk_Score": 90.0,
                "Mean_HR": 120.0,
                "Max_HR": 145.0,
                "Std_HR": 15.0,
                "Mean_SpO2": 86.0,
                "Min_SpO2": 82.0,
                "Mean_BP_Sys": 85.0,
                "Min_BP_Sys": 70.0,
                "Max_Temp": 39.2,
                "Mean_RR": 32.0,
                "Max_RR": 38.0,
                "Mean_GCS": 9.0,
                "Min_GCS": 8.0,
                "Hypoxia_Episodes": 5.0,
                "Tachycardia_Episodes": 8.0,
                "Vitals_Count": 35.0,
                "Gender": "M",
                "Department": "ICU",
                "Diagnosis": "Sepsis",
                "Insurance": "Government",
                "Acuity_Level": "Critical",
                "Comorbidities": "Both",
            },
        ]
    }
    response = client.post("/predict-batch", json=sample_patients)
    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert len(data["predictions"]) == 2
