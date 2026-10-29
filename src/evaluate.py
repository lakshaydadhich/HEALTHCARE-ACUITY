"""
Model Evaluation Pipeline
==========================
Evaluates the trained model on the hold-out test set.
Computes comprehensive classification metrics (accuracy, weighted/macro F1,
precision, recall, confusion matrix, per-class report) and logs them to
MLflow and local JSON/PNG artifacts.

Usage
-----
    python src/evaluate.py [--params params.yaml]
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# Ensure sibling modules are importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from feature_engineering import decode_target, encode_target, load_params, prepare_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def setup_mlflow(params: dict):
    """Configure MLflow tracking URI and experiment."""
    try:
        import mlflow

        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", params["mlflow"]["tracking_uri"])
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(params["mlflow"]["experiment_name"])
        logger.info("MLflow configured: URI=%s, Experiment=%s", tracking_uri, params["mlflow"]["experiment_name"])
        return mlflow
    except Exception as exc:
        logger.warning("MLflow setup warning: %s. Continuing with local artifacts only.", exc)
        return None


def plot_confusion_matrix(cm: np.ndarray, classes: list[str], output_path: Path):
    """Generate and save a styled confusion matrix heatmap."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        cbar=True,
    )
    plt.title("Acuity Bundle – Confusion Matrix (Test Set)", fontsize=14, fontweight="bold")
    plt.xlabel("Predicted Class", fontsize=11)
    plt.ylabel("True Class", fontsize=11)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info("Confusion matrix plot saved → %s", output_path)


def main(params_path: str = "params.yaml"):
    params = load_params(params_path)
    mlflow = setup_mlflow(params)

    # 1. Load data
    processed_dir = Path(params["data"]["processed_dir"])
    test_path = processed_dir / "test.csv"
    if not test_path.exists():
        raise FileNotFoundError(f"Test data not found at {test_path}. Run preprocess.py first.")

    df_test = pd.read_csv(test_path)
    target_col = params["target"]["column"]
    classes = params["target"]["classes"]

    X_test, y_test = prepare_data(df_test, target_col)
    y_test_enc = encode_target(y_test, classes)

    # 2. Load preprocessor and model
    preprocessor_path = Path(params["data"]["preprocessor_path"])
    model_path = Path(params["data"]["model_path"])

    if not preprocessor_path.exists():
        raise FileNotFoundError(f"Preprocessor not found at {preprocessor_path}. Run train.py first.")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}. Run train.py first.")

    logger.info("Loading preprocessor from %s", preprocessor_path)
    preprocessor = joblib.load(preprocessor_path)

    logger.info("Loading model from %s", model_path)
    model = joblib.load(model_path)

    # 3. Predict
    X_test_t = preprocessor.transform(X_test)
    y_pred_enc = model.predict(X_test_t)
    y_pred = decode_target(y_pred_enc, classes)

    # 4. Metrics
    acc = float(accuracy_score(y_test_enc, y_pred_enc))
    f1_weighted = float(f1_score(y_test_enc, y_pred_enc, average="weighted", zero_division=0))
    f1_macro = float(f1_score(y_test_enc, y_pred_enc, average="macro", zero_division=0))
    prec_weighted = float(precision_score(y_test_enc, y_pred_enc, average="weighted", zero_division=0))
    rec_weighted = float(recall_score(y_test_enc, y_pred_enc, average="weighted", zero_division=0))

    report_dict = classification_report(y_test_enc, y_pred_enc, target_names=classes, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test_enc, y_pred_enc)

    metrics = {
        "test_accuracy": acc,
        "test_f1_weighted": f1_weighted,
        "test_f1_macro": f1_macro,
        "test_precision_weighted": prec_weighted,
        "test_recall_weighted": rec_weighted,
        "per_class": {
            cls: {
                "precision": report_dict[cls]["precision"],
                "recall": report_dict[cls]["recall"],
                "f1-score": report_dict[cls]["f1-score"],
                "support": report_dict[cls]["support"],
            }
            for cls in classes if cls in report_dict
        },
    }

    logger.info("=" * 60)
    logger.info("EVALUATION RESULTS (TEST SET):")
    logger.info("Accuracy:           %.4f", acc)
    logger.info("Weighted F1:        %.4f", f1_weighted)
    logger.info("Macro F1:           %.4f", f1_macro)
    logger.info("Weighted Precision: %.4f", prec_weighted)
    logger.info("Weighted Recall:    %.4f", rec_weighted)
    logger.info("=" * 60)

    # 5. Save local evaluation artifacts
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    eval_metrics_path = models_dir / "eval_metrics.json"
    with open(eval_metrics_path, "w") as fh:
        json.dump(metrics, fh, indent=2)
    logger.info("Saved evaluation metrics → %s", eval_metrics_path)

    cm_path = models_dir / "confusion_matrix.png"
    plot_confusion_matrix(cm, classes, cm_path)

    # 6. Log to MLflow if available
    if mlflow:
        try:
            with mlflow.start_run(run_name="model_evaluation"):
                mlflow.log_metric("test_accuracy", acc)
                mlflow.log_metric("test_f1_weighted", f1_weighted)
                mlflow.log_metric("test_f1_macro", f1_macro)
                mlflow.log_metric("test_precision_weighted", prec_weighted)
                mlflow.log_metric("test_recall_weighted", rec_weighted)
                mlflow.log_artifact(str(eval_metrics_path))
                mlflow.log_artifact(str(cm_path))
                logger.info("Logged evaluation metrics and plots to MLflow.")
        except Exception as exc:
            logger.warning("Failed to log to MLflow: %s", exc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate acuity model on test set")
    parser.add_argument("--params", default="params.yaml", help="Path to params.yaml")
    args = parser.parse_args()
    main(args.params)
