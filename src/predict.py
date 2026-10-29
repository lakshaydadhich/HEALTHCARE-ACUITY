"""
Inference / Prediction Module
==============================
Provides batch and real-time prediction using the serialized preprocessor
and trained model.

Usage (CLI)
-----------
    python src/predict.py --input data/processed/test.csv --output predictions.csv
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Union

import joblib
import numpy as np
import pandas as pd
import yaml

# Ensure sibling modules are importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from feature_engineering import decode_target, load_params

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


class AcuityPredictor:
    """Encapsulates model loading and inference logic."""

    def __init__(self, params_path: str = "params.yaml"):
        self.params = load_params(params_path)
        self.classes = self.params["target"]["classes"]
        preprocessor_path = Path(self.params["data"]["preprocessor_path"])
        model_path = Path(self.params["data"]["model_path"])

        if not preprocessor_path.exists():
            raise FileNotFoundError(f"Preprocessor not found at {preprocessor_path}")
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")

        self.preprocessor = joblib.load(preprocessor_path)
        self.model = joblib.load(model_path)
        logger.info("AcuityPredictor initialized with %s and %s", preprocessor_path, model_path)

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run inference on input DataFrame.

        Returns DataFrame containing predicted class and class probabilities.
        """
        target_col = self.params["target"]["column"]
        X = df.drop(columns=[target_col], errors="ignore")

        # Transform features
        X_trans = self.preprocessor.transform(X)

        # Predict classes
        pred_enc = self.model.predict(X_trans)
        pred_labels = decode_target(pred_enc, self.classes)

        result_df = pd.DataFrame({
            "predicted_acuity_bundle": pred_labels,
            "predicted_code": pred_enc,
        })

        # Add probability distribution if supported
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X_trans)
            for idx, cls in enumerate(self.classes):
                if idx < proba.shape[1]:
                    result_df[f"prob_{cls}"] = proba[:, idx]

        return result_df

    def predict_single(self, record: dict[str, Any]) -> dict[str, Any]:
        """Predict for a single patient dictionary."""
        df = pd.DataFrame([record])
        res = self.predict(df)
        return res.to_dict(orient="records")[0]


def main():
    parser = argparse.ArgumentParser(description="Run inference using trained Acuity model")
    parser.add_argument("--params", default="params.yaml", help="Path to params.yaml")
    parser.add_argument("--input", required=True, help="Path to input CSV")
    parser.add_argument("--output", default="predictions.csv", help="Path to output CSV")
    args = parser.parse_args()

    predictor = AcuityPredictor(args.params)
    df_in = pd.read_csv(args.input)
    preds = predictor.predict(df_in)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df = pd.concat([df_in.reset_index(drop=True), preds], axis=1)
    out_df.to_csv(output_path, index=False)
    logger.info("Predictions saved → %s (%d rows)", output_path, len(out_df))


if __name__ == "__main__":
    main()
