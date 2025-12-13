import os
from datetime import datetime

from airflow.operators.bash import BashOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

from airflow import DAG

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))

with DAG(
    dag_id="drift_detection",
    start_date=datetime(2023, 1, 1),
    schedule_interval="@daily",
    catchup=False,
):
    check_drift = BashOperator(
        task_id="check_drift",
        bash_command="python /opt/src/src/drift_check.py",
        do_xcom_push=False,
        cwd=BASE,
    )

    trigger_retrain = TriggerDagRunOperator(
        task_id="trigger_retrain",
        trigger_dag_id="retrain_dag",
        trigger_rule="one_failed",
    )

    check_drift >> trigger_retrain
