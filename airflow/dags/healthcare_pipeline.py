"""
Healthcare Acuity Bundle MLOps Pipeline DAG
=============================================
Orchestrates preprocessing, model training with MLflow tracking, and model
evaluation on a scheduled or triggered basis using Apache Airflow.
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator

default_args = {
    "owner": "mlops_team",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="healthcare_acuity_ml_pipeline",
    default_args=default_args,
    description="End-to-end MLOps pipeline for Acuity Bundle prediction",
    schedule_interval=timedelta(days=7),  # Weekly retrain
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["healthcare", "mlops", "acuity-prediction"],
) as dag:

    preprocess_step = BashOperator(
        task_id="preprocess_data",
        bash_command="python src/preprocess.py --params params.yaml",
    )

    train_step = BashOperator(
        task_id="train_models",
        bash_command="python src/train.py --params params.yaml",
    )

    evaluate_step = BashOperator(
        task_id="evaluate_model",
        bash_command="python src/evaluate.py --params params.yaml",
    )

    preprocess_step >> train_step >> evaluate_step
