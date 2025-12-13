from datetime import datetime

from airflow.operators.trigger_dagrun import TriggerDagRunOperator

from airflow import DAG

with DAG(
    dag_id="retrain_dag",
    start_date=datetime(2023, 1, 1),
    schedule_interval=None,
    catchup=False,
):
    trigger_pipeline = TriggerDagRunOperator(
        task_id="trigger_pipeline",
        trigger_dag_id="pipeline",
    )
