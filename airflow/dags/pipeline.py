import json
import os
import shutil

from airflow.exceptions import AirflowSkipException
from airflow.models import Variable
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

from airflow import DAG

# ----------- Конфигурация DAG -------------
DEFAULT_ARGS = {
    "owner": "airflow",
    "retries": 0,
}

dag = DAG(
    dag_id="pipeline",
    default_args=DEFAULT_ARGS,
    start_date=days_ago(1),
    schedule_interval=None,
    catchup=False,
    tags=["lab8"],
)

# ----------- Пути и переменные -------------


BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
SRC = os.path.join(BASE, "src")
DATA_DIR = os.path.join(BASE, "data")
MODELS_DIR = os.path.join(BASE, "models")
REPORTS_DIR = os.path.join(BASE, "reports")

# Пути к артефактам
RAW_DATA = os.path.join(DATA_DIR, "raw/dataset.csv")
PROCESSED_DATA = os.path.join(DATA_DIR, "processed/dataset.csv")
X_TRAIN = os.path.join(DATA_DIR, "split/X_train.csv")
X_TEST = os.path.join(DATA_DIR, "split/X_test.csv")
Y_TRAIN = os.path.join(DATA_DIR, "split/y_train.csv")
Y_TEST = os.path.join(DATA_DIR, "split/y_test.csv")
MODEL_PATH = os.path.join(MODELS_DIR, "xgboost_model.pkl")
EVAL_REPORT = os.path.join(REPORTS_DIR, "eval.json")

# Порог RMSE для регистрации модели
EVAL_THRESHOLD = float(Variable.get("EVAL_THRESHOLD", default_var=1000.0))  # пример

# ----------- Таски DAG -------------
# 1) download_data (если нужно, например dvc pull)
download_data = BashOperator(
    task_id="download_data",
    bash_command=f"echo 'No-op: данные уже в {RAW_DATA}'",  # можно заменить на dvc pull
    cwd=BASE,
    dag=dag,
)

check_mlflow = BashOperator(
    task_id="check_mlflow",
    bash_command="curl -s http://mlflow:5000 || exit 1",
    dag=dag,
)

# 2) preprocess
preprocess = BashOperator(
    task_id="preprocess",
    bash_command=f"python {os.path.join(SRC, 'preprocess.py')} --config {os.path.join(BASE, 'params.yaml')}",
    cwd=BASE,
    dag=dag,
)

# 3) split_data
split = BashOperator(
    task_id="split_data",
    bash_command=f"python {os.path.join(SRC, 'split_data.py')} --config {os.path.join(BASE, 'params.yaml')}",
    cwd=BASE,
    dag=dag,
)

# 4) train
train = BashOperator(
    task_id="train",
    bash_command=f"python {os.path.join(SRC, 'train.py')} --config {os.path.join(BASE, 'params.yaml')} --model xgboost",
    cwd=BASE,
    dag=dag,
)

# 5) evaluate
evaluate = BashOperator(
    task_id="evaluate",
    bash_command=(
        f"python {os.path.join(SRC, 'evaluate.py')} "
        f"--model_path {MODEL_PATH} "
        f"--test_data {X_TEST} "
        f"--target_data {Y_TEST} "
        f"--config {os.path.join(BASE, 'params.yaml')}"
    ),
    cwd=BASE,
    dag=dag,
)


def register_model(**kwargs):
    if not os.path.exists(EVAL_REPORT):
        raise FileNotFoundError(f"Eval report not found: {EVAL_REPORT}")

    with open(EVAL_REPORT, "r") as f:
        metrics = json.load(f)

    rmse = metrics.get("rmse")
    if rmse is None:
        raise ValueError("RMSE not found in eval report")

    if rmse <= EVAL_THRESHOLD:
        reg_dir = os.path.join(MODELS_DIR, "production")  # <- отдельная папка
        os.makedirs(reg_dir, exist_ok=True)
        target_path = os.path.join(reg_dir, os.path.basename(MODEL_PATH))

        if os.path.abspath(MODEL_PATH) != os.path.abspath(target_path):
            shutil.copy(MODEL_PATH, target_path)
            print(f"Model registered at {target_path}, RMSE={rmse:.4f}")
        else:
            print(f"Model already at target path: {target_path}, skipping copy")
    else:
        raise AirflowSkipException(f"RMSE {rmse:.4f} > threshold {EVAL_THRESHOLD}, skipping registration")


register = PythonOperator(
    task_id="register",
    python_callable=register_model,
    provide_context=True,
    dag=dag,
)

download_data >> check_mlflow >> preprocess >> split >> train >> evaluate >> register
