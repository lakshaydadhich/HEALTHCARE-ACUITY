"""
Model Training Pipeline
========================
Trains multiple classifiers (Logistic Regression, Random Forest,
XGBoost) with ``RandomizedSearchCV`` hyperparameter tuning.  Every
candidate is logged to **MLflow** so experiments are fully
reproducible and comparable.

Usage
-----
    python src/train.py [--params params.yaml]
"""

import argparse
import json
import logging
import os
import sys
import warnings
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier

# Ensure sibling modules are importable when running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent))

from feature_engineering import (
    build_preprocessor,
    encode_target,
    get_feature_lists,
    get_feature_names,
    load_params,
    prepare_data,
)

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────
# MLflow setup
# ────────────────────────────────────────────────────────────────────

def setup_mlflow(params: dict) -> None:
    """Configure the MLflow tracking URI and experiment name."""
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI",
        params["mlflow"]["tracking_uri"],
    )
    try:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(params["mlflow"]["experiment_name"])
    except Exception as exc:
        logger.warning("Could not connect to MLflow server at %s (%s). Falling back to sqlite:///mlflow.db", tracking_uri, exc)
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_experiment(params["mlflow"]["experiment_name"])
    logger.info("MLflow tracking URI : %s", mlflow.get_tracking_uri())
    logger.info("MLflow experiment   : %s", params["mlflow"]["experiment_name"])


# ────────────────────────────────────────────────────────────────────
# Model definitions
# ────────────────────────────────────────────────────────────────────

def get_model_configs(params: dict) -> dict:
    """Build a dict of ``{name: {model, params}}`` from *params.yaml*."""
    tcfg = params["train"]
    rs = tcfg["random_state"]
    model_defs = tcfg["models"]
    configs: dict = {}

    if "logistic_regression" in model_defs:
        p = model_defs["logistic_regression"]
        configs["LogisticRegression"] = {
            "model": LogisticRegression(
                random_state=rs,
                class_weight="balanced",
                max_iter=1000,
            ),
            "params": {k: v for k, v in p.items() if k not in ("max_iter", "penalty")},
        }

    if "random_forest" in model_defs:
        p = model_defs["random_forest"]
        configs["RandomForest"] = {
            "model": RandomForestClassifier(
                random_state=rs,
                class_weight="balanced",
                n_jobs=-1,
            ),
            "params": p,
        }

    if "xgboost" in model_defs:
        p = model_defs["xgboost"]
        configs["XGBoost"] = {
            "model": XGBClassifier(
                random_state=rs,
                objective="multi:softprob",
                eval_metric="mlogloss",
                n_jobs=-1,
                verbosity=0,
            ),
            "params": p,
        }

    return configs


# ────────────────────────────────────────────────────────────────────
# Single-model training
# ────────────────────────────────────────────────────────────────────

def train_single_model(
    name: str,
    config: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    params: dict,
) -> dict:
    """Run ``RandomizedSearchCV`` for one model and log everything to
    MLflow as a *nested* run."""
    tcfg = params["train"]

    with mlflow.start_run(run_name=name, nested=True):
        logger.info("Training %s …", name)

        search = RandomizedSearchCV(
            estimator=config["model"],
            param_distributions=config["params"],
            n_iter=min(tcfg["n_iter"], _param_space_size(config["params"])),
            cv=tcfg["cv_folds"],
            scoring=tcfg["scoring"],
            random_state=tcfg["random_state"],
            n_jobs=-1,
            verbose=0,
        )
        search.fit(X_train, y_train)

        best = search.best_estimator_
        best_params = search.best_params_

        # ── Evaluate ──
        y_train_pred = best.predict(X_train)
        y_val_pred = best.predict(X_val)

        train_f1 = f1_score(y_train, y_train_pred, average="weighted")
        val_f1 = f1_score(y_val, y_val_pred, average="weighted")
        val_acc = accuracy_score(y_val, y_val_pred)

        # ── Log to MLflow ──
        mlflow.log_params(best_params)
        mlflow.log_metric("train_f1_weighted", round(train_f1, 4))
        mlflow.log_metric("val_f1_weighted", round(val_f1, 4))
        mlflow.log_metric("val_accuracy", round(val_acc, 4))
        mlflow.log_metric("cv_best_score", round(search.best_score_, 4))
        try:
            mlflow.sklearn.log_model(best, name="model", serialization_format="cloudpickle")
        except Exception as exc:
            logger.warning("Could not log model artifact to MLflow: %s", exc)

        run_id = mlflow.active_run().info.run_id
        logger.info(
            "  %s → val_f1=%.4f  val_acc=%.4f  cv_best=%.4f  (run %s)",
            name, val_f1, val_acc, search.best_score_, run_id[:8],
        )

        return {
            "name": name,
            "model": best,
            "best_params": best_params,
            "train_f1": round(train_f1, 4),
            "val_f1": round(val_f1, 4),
            "val_accuracy": round(val_acc, 4),
            "cv_best_score": round(search.best_score_, 4),
            "run_id": run_id,
        }


def _param_space_size(param_grid: dict) -> int:
    """Upper bound on the number of distinct combos in a param grid."""
    size = 1
    for v in param_grid.values():
        size *= len(v) if isinstance(v, list) else 1
    return size


# ────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────

def main(params_path: str = "params.yaml") -> None:
    params = load_params(params_path)
    setup_mlflow(params)

    # ── Load processed splits ──
    processed = Path(params["data"]["processed_dir"])
    train_df = pd.read_csv(processed / "train.csv")
    val_df = pd.read_csv(processed / "val.csv")

    target_col = params["target"]["column"]
    classes = params["target"]["classes"]

    X_train, y_train = prepare_data(train_df, target_col)
    X_val, y_val = prepare_data(val_df, target_col)

    y_train_enc = encode_target(y_train, classes)
    y_val_enc = encode_target(y_val, classes)

    # ── Build & fit preprocessor ──
    preprocessor = build_preprocessor(params)
    X_train_t = preprocessor.fit_transform(X_train)
    X_val_t = preprocessor.transform(X_val)
    logger.info("Transformed feature matrix: %s", X_train_t.shape)

    # ── Save preprocessor ──
    prep_path = Path(params["data"]["preprocessor_path"])
    prep_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, prep_path)
    logger.info("Saved preprocessor → %s", prep_path)

    # ── Save feature names ──
    feat_names = get_feature_names(preprocessor)
    with open(processed / "feature_names.json", "w", encoding="utf-8") as fh:
        json.dump(feat_names, fh, indent=2)

    # ── Train all models ──
    model_configs = get_model_configs(params)
    results: list[dict] = []

    with mlflow.start_run(run_name="training_pipeline") as parent_run:
        mlflow.log_param("n_train", len(X_train))
        mlflow.log_param("n_val", len(X_val))
        mlflow.log_param("n_features", X_train_t.shape[1])
        mlflow.log_param("target", target_col)
        mlflow.log_param("classes", str(classes))

        for name, config in model_configs.items():
            result = train_single_model(
                name, config,
                X_train_t, y_train_enc,
                X_val_t, y_val_enc,
                params,
            )
            results.append(result)

        # ── Select best ──
        best = max(results, key=lambda r: r["val_f1"])
        mlflow.log_metric("best_val_f1", best["val_f1"])
        mlflow.log_param("best_model", best["name"])

    logger.info("")
    logger.info("🏆  Best model: %s  (val_f1 = %.4f)", best["name"], best["val_f1"])

    # ── Save best model ──
    model_path = Path(params["data"]["model_path"])
    joblib.dump(best["model"], model_path)
    logger.info("Saved best model → %s", model_path)

    # ── Save training metrics ──
    metrics = {
        "best_model": best["name"],
        "best_params": {k: str(v) for k, v in best["best_params"].items()},
        "train_f1_weighted": best["train_f1"],
        "val_f1_weighted": best["val_f1"],
        "val_accuracy": best["val_accuracy"],
        "cv_best_score": best["cv_best_score"],
        "parent_run_id": parent_run.info.run_id,
        "all_results": [
            {
                "name": r["name"],
                "val_f1_weighted": r["val_f1"],
                "val_accuracy": r["val_accuracy"],
                "run_id": r["run_id"],
            }
            for r in results
        ],
    }
    metrics_path = Path("models/train_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    logger.info("Saved training metrics → %s", metrics_path)


# ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train acuity models")
    parser.add_argument("--params", default="params.yaml")
    args = parser.parse_args()
    main(args.params)
