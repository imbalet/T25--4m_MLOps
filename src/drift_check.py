import json
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

PSI_THRESHOLD = 0.2
CAT_JS_THRESHOLD = 0.1  # порог для категориальных фич
METRIC_THRESHOLD = 0.05  # падение метрики на 5%


# ----------------------------------------------------
# PSI — ТОЛЬКО для числовых
# ----------------------------------------------------
def psi(expected, actual, buckets=10):
    expected = np.array(expected)
    actual = np.array(actual)

    quantiles = np.linspace(0, 1, buckets + 1)
    expected_bins = np.quantile(expected, quantiles)
    actual_bins = np.quantile(actual, quantiles)

    psi_value = 0
    for i in range(buckets):
        exp_perc = np.mean((expected >= expected_bins[i]) & (expected < expected_bins[i + 1]))
        act_perc = np.mean((actual >= actual_bins[i]) & (actual < actual_bins[i + 1]))

        exp_perc = max(exp_perc, 0.0001)
        act_perc = max(act_perc, 0.0001)

        psi_value += (exp_perc - act_perc) * np.log(exp_perc / act_perc)

    return psi_value


# ----------------------------------------------------
# JS Divergence — ДЛЯ КАТЕГОРИАЛЬНЫХ
# ----------------------------------------------------
def jensen_shannon(p, q):
    # выравнивание словарей
    keys = set(p.keys()).union(set(q.keys()))
    p_vec = np.array([p.get(k, 0) for k in keys], dtype=float)
    q_vec = np.array([q.get(k, 0) for k in keys], dtype=float)

    # нормализация в вероятности
    p_vec /= p_vec.sum()
    q_vec /= q_vec.sum()

    m = 0.5 * (p_vec + q_vec)

    def kl(a, b):
        mask = a > 0
        return np.sum(a[mask] * np.log(a[mask] / b[mask]))

    return 0.5 * kl(p_vec, m) + 0.5 * kl(q_vec, m)


def categorical_drift(train_col, prod_col):
    p = train_col.value_counts(normalize=True).to_dict()
    q = prod_col.value_counts(normalize=True).to_dict()
    return jensen_shannon(p, q)


# ----------------------------------------------------
# MAIN
# ----------------------------------------------------
def main():
    train = pd.read_csv("data/drift/reference.csv")
    prod = pd.read_csv("data/drift/production.csv")
    model = pd.read_pickle("models/production/xgboost_model.pkl")

    drift_report = {"feature_drift": {}, "metric_drift": None}
    drift_detected = False

    # Определение типов
    numeric_cols = train.select_dtypes(include=[np.number]).columns
    categorical_cols = train.select_dtypes(exclude=[np.number]).columns

    # ----------------------------------------------------
    # FEATURE DRIFT
    # ----------------------------------------------------
    # ЧИСЛОВЫЕ
    for col in numeric_cols:
        if col == "target":
            continue
        value = psi(train[col].values, prod[col].values)
        drift_report["feature_drift"][col] = value
        if value > PSI_THRESHOLD:
            drift_detected = True

    # КАТЕГОРИАЛЬНЫЕ
    for col in categorical_cols:
        value = categorical_drift(train[col], prod[col])
        drift_report["feature_drift"][col] = value
        if value > CAT_JS_THRESHOLD:
            drift_detected = True

    # ----------------------------------------------------
    # PERFORMANCE DRIFT (если в проде есть target)
    # ----------------------------------------------------
    if "target" in prod.columns:
        y_true = prod["target"]
        X = prod.drop(columns=["target"])

        y_pred = model.predict_proba(X)[:, 1]
        current_auc = roc_auc_score(y_true, y_pred)

        prev_auc = json.load(open("reports/prev_auc.json"))["auc"]
        drift_report["metric_drift"] = float(prev_auc - current_auc)

        if prev_auc - current_auc > METRIC_THRESHOLD:
            drift_detected = True

        json.dump({"auc": current_auc}, open("reports/prev_auc.json", "w"))

    # ----------------------------------------------------
    # SAVE REPORT + EXIT CODE
    # ----------------------------------------------------
    json.dump(drift_report, open("reports/drift_report.json", "w"), indent=2)

    if drift_detected:
        print("DRIFT DETECTED")
        sys.exit(1)
    else:
        print("NO DRIFT")
        sys.exit(0)


if __name__ == "__main__":
    main()
