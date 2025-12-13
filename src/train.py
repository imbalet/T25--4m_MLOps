import argparse
import os
from datetime import datetime
from logging import getLogger

import joblib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import xgboost as xgb
import yaml
from feast import FeatureStore
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    root_mean_squared_error,
)

logger = getLogger(__name__)


def load_config(config_path="params.yaml"):
    """Загрузка конфигурации"""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ====================== NEW FUNCTION ============================
def load_data_from_feast(config):
    """
    Загрузка данных через Feast Feature Store.
    Требует чтобы entity_df содержала:
        car_id
        event_timestamp
    Если event_timestamp отсутствует — создаём искусственный.
    """
    from datetime import datetime

    split_dir = config["data"]["split_dir"]

    # Читаем entity таблицы
    entity_train = pd.read_csv(f"{split_dir}/X_train.csv")
    entity_test = pd.read_csv(f"{split_dir}/X_test.csv")

    entity_train["event_timestamp"] = pd.to_datetime(entity_train["event_timestamp"])
    entity_test["event_timestamp"] = pd.to_datetime(entity_test["event_timestamp"])

    # Проверяем обязательные поля
    if "car_id" not in entity_train.columns:
        raise ValueError("X_train.csv MUST contain 'car_id' column")
    if "car_id" not in entity_test.columns:
        raise ValueError("X_test.csv MUST contain 'car_id' column")

    # Feast требует event_timestamp → если нет, создаем фиктивный
    if "event_timestamp" not in entity_train.columns:
        entity_train["event_timestamp"] = datetime.utcnow()
    if "event_timestamp" not in entity_test.columns:
        entity_test["event_timestamp"] = datetime.utcnow()

    # Загружаем таргеты
    y_train = pd.read_csv(f"{split_dir}/y_train.csv").values.ravel()
    y_test = pd.read_csv(f"{split_dir}/y_test.csv").values.ravel()

    # Инициализируем Feature Store
    store = FeatureStore(repo_path="feature_repo")

    # Фичи
    feast_features = [
        "car_features:owners",
        "car_features:year",
        "car_features:region",
        "car_features:mileage",
        "car_features:mark",
        "car_features:model",
        "car_features:complectation",
        "car_features:steering_wheel",
        "car_features:gear_type",
        "car_features:engine",
        "car_features:transmission",
        "car_features:power",
        "car_features:displacement",
        "car_features:color",
        "car_features:body_type_type",
        "car_features:super_gen_name",
    ]

    # Получаем фичи
    X_train = store.get_historical_features(
        entity_df=entity_train,
        features=feast_features,
    ).to_df()

    X_test = store.get_historical_features(
        entity_df=entity_test,
        features=feast_features,
    ).to_df()

    # Удаляем служебные колонки Feast
    drop_cols = ["event_timestamp", "car_id"]
    X_train.drop(columns=[c for c in drop_cols if c in X_train], inplace=True)
    X_test.drop(columns=[c for c in drop_cols if c in X_test], inplace=True)

    return X_train, X_test, y_train, y_test


# ===============================================================


def calculate_metrics_regression(y_true, y_pred):
    """Расчет метрик регрессии"""
    return {
        "mse": mean_squared_error(y_true, y_pred),
        "rmse": root_mean_squared_error(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }


def plot_feature_importance(model, feature_names, model_name):
    """Визуализация важности признаков"""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:15]

        plt.figure(figsize=(12, 8))
        plt.title(f"Feature Importance - {model_name}")
        plt.bar(range(len(indices)), importances[indices])
        plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=45)
        plt.tight_layout()

        plot_path = f"reports/feature_importance_{model_name.lower()}.png"
        os.makedirs("reports", exist_ok=True)
        plt.savefig(plot_path)
        plt.close()

        return plot_path
    return None


def train_model(model_type, model_params, X_train, y_train, X_test, y_test, feature_names):
    """Обучение модели регрессии с MLflow"""

    with mlflow.start_run(run_name=f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        mlflow.log_params(model_params)
        mlflow.set_tag("model_type", model_type)
        mlflow.set_tag("dataset", "car_price")

        if model_type == "random_forest":
            model = RandomForestRegressor(**model_params)
            mlflow.sklearn.autolog()
        elif model_type == "xgboost":
            model = xgb.XGBRegressor(**model_params)
            mlflow.xgboost.autolog()
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        print(f"Training {model_type} model...")
        model.fit(X_train, y_train)

        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        train_metrics = calculate_metrics_regression(y_train, y_pred_train)
        test_metrics = calculate_metrics_regression(y_test, y_pred_test)

        for m, v in train_metrics.items():
            mlflow.log_metric(f"train_{m}", v)

        for m, v in test_metrics.items():
            mlflow.log_metric(f"test_{m}", v)

        mlflow.log_metric("overfitting_rmse", train_metrics["rmse"] - test_metrics["rmse"])

        fi_path = plot_feature_importance(model, feature_names, model_type)
        if fi_path:
            mlflow.log_artifact(fi_path)

        model_path = f"models/{model_type}_model.pkl"
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, model_path)
        mlflow.log_artifact(model_path)

        print(f"\n{model_type.upper()} RESULTS:")
        print("Train metrics:")
        for k, v in train_metrics.items():
            print(f"  {k}: {v:.4f}")

        print("Test metrics:")
        for k, v in test_metrics.items():
            print(f"  {k}: {v:.4f}")

        return model, test_metrics["rmse"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml", help="Config file path")
    parser.add_argument("--model", help="Specific model to train")
    args = parser.parse_args()

    config = load_config(args.config)

    # MLflow setup
    mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", None)
    if mlflow_tracking_uri:
        mlflow.set_tracking_uri(mlflow_tracking_uri)
    elif config["experiments"]["tracking_uri"]:
        mlflow.set_tracking_uri(config["experiments"]["tracking_uri"])

    experiment_name = config["experiments"]["experiment_name"]
    experiment = mlflow.get_experiment_by_name(experiment_name)

    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id

    print(f"experiment {experiment_id}")
    mlflow.set_experiment(experiment_name)

    # LOAD DATA THROUGH FEAST
    print("Loading data via Feast...")
    X_train, X_test, y_train, y_test = load_data_from_feast(config)
    feature_names = X_train.columns.tolist()

    print(f"Train shape: {X_train.shape}")
    print(f"Test shape: {X_test.shape}")

    results = {}
    models_to_train = [args.model] if args.model else config["models"].keys()

    for model_type in models_to_train:
        if model_type not in config["models"]:
            print(f"Warning: unknown model {model_type}")
            continue

        model_params = config["models"][model_type]

        # если параметр — список, берём первое значение
        base_params = {k: (v[0] if isinstance(v, list) else v) for k, v in model_params.items()}

        model, score = train_model(model_type, base_params, X_train, y_train, X_test, y_test, feature_names)
        results[model_type] = score

    print("\n" + "=" * 50)
    print("MODEL COMPARISON:")
    print("=" * 50)

    for model_type, score in sorted(results.items(), key=lambda x: x[1]):
        print(f"{model_type:20}: RMSE = {score:.4f}")


if __name__ == "__main__":
    main()
