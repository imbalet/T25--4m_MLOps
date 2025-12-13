import argparse
import os

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml")
    args = parser.parse_args()

    config = load_config(args.config)

    df = pd.read_csv(config["data"]["processed"])
    target_col = "price"  # или другой целевой признак, если у тебя есть
    feature_cols = [c for c in df.columns if c != target_col]

    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config["preprocess"]["test_size"],
        random_state=config["preprocess"]["random_state"],
    )

    os.makedirs("data/split", exist_ok=True)
    os.makedirs("data/drift", exist_ok=True)

    X_train.to_csv("data/split/X_train.csv", index=False)
    X_test.to_csv("data/split/X_test.csv", index=False)
    y_train.to_csv("data/split/y_train.csv", index=False)
    y_test.to_csv("data/split/y_test.csv", index=False)

    sample = X_train.sample(n=5000, random_state=42)
    sample.to_csv("data/drift/reference.csv", index=False)
    sample.to_csv("data/drift/production.csv", index=False)
