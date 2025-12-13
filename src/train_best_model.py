import argparse
import os
from datetime import datetime

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
import yaml
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


def load_config(config_path="params.yaml"):
    """Загрузка конфигурации"""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_data(config):
    """Загрузка данных"""
    split_dir = config["data"]["split_dir"]

    X_train = pd.read_csv(f"{split_dir}/X_train.csv")
    X_test = pd.read_csv(f"{split_dir}/X_test.csv")
    y_train = pd.read_csv(f"{split_dir}/y_train.csv").values.ravel()
    y_test = pd.read_csv(f"{split_dir}/y_test.csv").values.ravel()

    return X_train, X_test, y_train, y_test


def load_best_params_from_mlflow(experiment_name, model_type):
    """Загрузка лучших параметров из MLflow"""
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        print(f"Experiment {experiment_name} not found")
        return None

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.model_type = '{model_type}'",
        order_by=["metrics.best_cv_score DESC"],
        max_results=1,
    )

    if runs.empty:  # type: ignore
        print(f"No runs found for {model_type}")
        return None

    best_run = runs.iloc[0]  # type: ignore

    # Извлекаем лучшие параметры
    best_params = {}
    for col in best_run.index:
        if col.startswith("params.best_"):
            param_name = col.replace("params.best_", "")
            param_value = best_run[col]

            # Конвертируем типы
            if param_name in ["n_estimators", "max_depth", "min_samples_split"]:
                best_params[param_name] = int(float(param_value))  # type: ignore
            elif param_name in ["learning_rate"]:
                best_params[param_name] = float(param_value)  # type: ignore
            else:
                best_params[param_name] = param_value

    best_params["random_state"] = 42
    best_params = {
        **best_params,
        "tree_method": "hist",
        "device": "cuda",
    }
    return best_params


def create_shap_plots(model, X_test, model_name):
    """Создание SHAP графиков для регрессии"""
    try:
        explainer = shap.Explainer(model, X_test.sample(min(100, len(X_test))))
        X_sample = X_test.sample(min(50, len(X_test)))
        shap_values = explainer(X_sample)

        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_sample, show=False)
        plt.title(f"SHAP Summary Plot - {model_name}")

        os.makedirs("reports", exist_ok=True)
        shap_path = f"reports/shap_summary_{model_name.lower()}.png"
        plt.savefig(shap_path, bbox_inches="tight")
        plt.close()

        return shap_path
    except Exception as e:
        print(f"Could not create SHAP plots: {e}")
        return None


def train_final_model(model_type, best_params, X_train, y_train, X_test, y_test):
    """Обучение финальной модели регрессии с лучшими параметрами"""

    run_name = f"final_{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(best_params)
        mlflow.set_tag("model_type", model_type)
        mlflow.set_tag("task", "regression")
        mlflow.set_tag("model_stage", "final")

        # Создаем модель
        if model_type == "random_forest":
            model = RandomForestRegressor(**best_params)
        elif model_type == "xgboost":
            model = xgb.XGBRegressor(**best_params)
        else:
            raise ValueError("Unknown model type")

        print(f"Training final {model_type} model...")
        model.fit(X_train, y_train)

        # Предсказания
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        # Метрики
        train_rmse = mean_squared_error(y_train, y_pred_train)
        test_rmse = mean_squared_error(y_test, y_pred_test)
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)

        # Логируем метрики
        mlflow.log_metric("train_rmse", train_rmse)
        mlflow.log_metric("test_rmse", test_rmse)
        mlflow.log_metric("train_r2", train_r2)
        mlflow.log_metric("test_r2", test_r2)
        mlflow.log_metric("overfitting_gap", train_rmse - test_rmse)

        # Feature Importance
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            feature_names = X_train.columns.tolist()
            indices = np.argsort(importances)[::-1][:15]

            plt.figure(figsize=(12, 8))
            plt.title(f"Feature Importance - {model_type}")
            plt.bar(range(len(indices)), importances[indices])
            plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=45)
            plt.tight_layout()

            fi_path = f"reports/final_feature_importance_{model_type}.png"
            plt.savefig(fi_path)
            plt.close()
            mlflow.log_artifact(fi_path)

        # SHAP анализ
        shap_path = create_shap_plots(model, X_test, f"final_{model_type}")
        if shap_path:
            mlflow.log_artifact(shap_path)

        # Сохраняем модель
        os.makedirs("models", exist_ok=True)
        final_model_path = f"models/final_{model_type}_model.pkl"
        joblib.dump(model, final_model_path)
        mlflow.log_artifact(final_model_path)

        # Регистрируем модель
        if model_type == "random_forest":
            model_info = mlflow.sklearn.log_model(model, "model", registered_model_name="CarPrices_RF")  # type: ignore
        elif model_type == "xgboost":
            model_info = mlflow.xgboost.log_model(model, "model", registered_model_name="CarPrices_XGB")  # type: ignore
        else:
            raise ValueError("Unknown model type")

        print(f"\nFinal {model_type.upper()} Results:")
        print(f"Test RMSE: {test_rmse:.4f}")
        print(f"Test R2: {test_r2:.4f}")

        return model, test_rmse, model_info.model_uri


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml", help="Config file path")
    parser.add_argument(
        "--model",
        required=True,
        choices=["random_forest", "xgboost"],
        help="Model to train with best parameters",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    # MLflow setup
    if config["experiments"]["tracking_uri"]:
        mlflow.set_tracking_uri(config["experiments"]["tracking_uri"])
    mlflow.set_experiment(config["experiments"]["experiment_name"])

    # Load data
    X_train, X_test, y_train, y_test = load_data(config)

    # Load best hyperparameters
    hyperopt_experiment_name = f"{config['experiments']['experiment_name']}_hyperopt"
    best_params = load_best_params_from_mlflow(hyperopt_experiment_name, args.model)

    if best_params is None:
        print("No optimized parameters found. Using default parameters.")
        model_config = config["models"][args.model]
        best_params = {k: v[0] if isinstance(v, list) else v for k, v in model_config.items()}

    # Train final model
    model, test_rmse, model_uri = train_final_model(args.model, best_params, X_train, y_train, X_test, y_test)

    print("\nFinal model training completed!")
    print(f"Model URI: {model_uri}")
    print(f"Test RMSE: {test_rmse:.4f}")


if __name__ == "__main__":
    main()
