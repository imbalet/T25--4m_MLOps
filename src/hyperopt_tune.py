import argparse
import pickle
from datetime import datetime

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import pandas as pd
import xgboost as xgb
import yaml
from hyperopt import STATUS_OK, Trials, fmin, hp, space_eval, tpe
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score


def load_config(config_path="params.yaml"):
    """Загрузка конфигурации"""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_data(config):
    """Загрузка подготовленных данных"""
    split_dir = config["data"]["split_dir"]

    X_train = pd.read_csv(f"{split_dir}/X_train.csv")
    X_test = pd.read_csv(f"{split_dir}/X_test.csv")
    y_train = pd.read_csv(f"{split_dir}/y_train.csv").values.ravel()
    y_test = pd.read_csv(f"{split_dir}/y_test.csv").values.ravel()

    return X_train, X_test, y_train, y_test


# --------- Objective функции ---------


def objective_random_forest(params, X_train, y_train, cv_folds, scoring="neg_mean_squared_error"):
    """Objective для RandomForestRegressor"""
    params = {
        "n_estimators": int(params["n_estimators"]),
        "max_depth": int(params["max_depth"]) if params["max_depth"] != 0 else None,
        "min_samples_split": int(params["min_samples_split"]),
        "random_state": 42,
        "n_jobs": -1,
    }

    model = RandomForestRegressor(**params)
    scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring=scoring)

    with mlflow.start_run(nested=True):
        mlflow.log_params(params)
        mlflow.log_metric("cv_score_mean", scores.mean())
        mlflow.log_metric("cv_score_std", scores.std())
        mlflow.set_tag("model", "RandomForestRegressor")
        mlflow.set_tag("task", "regression")
        mlflow.set_tag("metric", scoring)

    # так как метрика отрицательная — возвращаем loss = -score
    return {"loss": -scores.mean(), "status": STATUS_OK}


def objective_xgboost(params, X_train, y_train, cv_folds, scoring="neg_mean_squared_error"):
    """Objective для XGBRegressor"""
    params = {
        "n_estimators": int(params["n_estimators"]),
        "max_depth": int(params["max_depth"]),
        "learning_rate": params["learning_rate"],
        "random_state": 42,
        "tree_method": "hist",
        "device": "cuda",
    }

    model = xgb.XGBRegressor(**params)
    scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring=scoring)

    with mlflow.start_run(nested=True):
        mlflow.log_params(params)
        mlflow.log_metric("cv_score_mean", scores.mean())
        mlflow.log_metric("cv_score_std", scores.std())
        mlflow.set_tag("model", "XGBRegressor")
        mlflow.set_tag("task", "regression")
        mlflow.set_tag("metric", scoring)

    return {"loss": -scores.mean(), "status": STATUS_OK}


# --------- Основная функция оптимизации ---------

objectives = {
    "random_forest": objective_random_forest,
    "xgboost": objective_xgboost,
}


def optimize_model(model_type, X_train, y_train, config):
    """Гиперпараметрическая оптимизация"""
    hyperopt_config = config["hyperparameter_tuning"]
    max_evals = hyperopt_config["max_evals"]
    cv_folds = hyperopt_config["cv_folds"]

    # Для регрессии используем RMSE (чем меньше, тем лучше)
    scoring = hyperopt_config.get("scoring", "neg_root_mean_squared_error")

    if model_type == "random_forest":
        space = {
            "n_estimators": hp.choice("n_estimators", [100, 200, 300, 500]),
            "max_depth": hp.choice("max_depth", [0, 5, 10, 15, 20]),  # 0 → None
            "min_samples_split": hp.choice("min_samples_split", [2, 5, 10]),
        }

    elif model_type == "xgboost":
        space = {
            "n_estimators": hp.choice("n_estimators", [100, 200, 300, 500]),
            "max_depth": hp.choice("max_depth", [3, 6, 9, 12]),
            "learning_rate": hp.choice("learning_rate", [0.01, 0.05, 0.1, 0.2]),
        }

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    print(f"Starting hyperparameter optimization for {model_type}...")
    print(f"Max evaluations: {max_evals}")

    with mlflow.start_run(run_name=f"hyperopt_{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        mlflow.set_tag("optimization_type", "hyperopt")
        mlflow.set_tag("model_type", model_type)
        mlflow.set_tag("task", "regression")
        mlflow.log_param("max_evals", max_evals)
        mlflow.log_param("cv_folds", cv_folds)
        mlflow.log_param("scoring", scoring)

        trials = Trials()

        best = fmin(
            fn=(lambda params: objectives[model_type](params, X_train, y_train, cv_folds, scoring)),
            space=space,
            algo=tpe.suggest,
            max_evals=max_evals,
            trials=trials,
        )

        best_params = space_eval(space, best)
        best_score = -trials.best_trial["result"]["loss"]  # type: ignore

        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
        mlflow.log_metric("best_cv_score", best_score)

        with open(f"experiments/hyperopt_trials_{model_type}.pkl", "wb") as f:
            pickle.dump(trials, f)

        print(f"Best parameters for {model_type}: {best_params}")
        print(f"Best CV score (RMSE): {best_score:.4f}")

        return best_params, best_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml", help="Config file path")
    parser.add_argument(
        "--model",
        required=True,
        choices=["random_forest", "xgboost"],
        help="Model to optimize",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    if config["experiments"]["tracking_uri"]:
        mlflow.set_tracking_uri(config["experiments"]["tracking_uri"])

    experiment_name = f"{config['experiments']['experiment_name']}_hyperopt"
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        mlflow.create_experiment(experiment_name)
    mlflow.set_experiment(experiment_name)

    X_train, X_test, y_train, y_test = load_data(config)

    best_params, best_score = optimize_model(args.model, X_train, y_train, config)

    print("\nOptimization completed!")
    print(f"Best {args.model} parameters: {best_params}")
    print(f"Best CV score (RMSE): {best_score:.4f}")


if __name__ == "__main__":
    main()
