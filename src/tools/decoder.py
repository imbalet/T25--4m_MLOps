import numpy as np
import pandas as pd


def apply_encoders_and_scaler(data: dict, encoders, scaler):
    df = pd.DataFrame([data]) if isinstance(data, dict) else pd.DataFrame(data)
    df_transformed = df.copy()

    for col, encoder in encoders.items():
        if col not in df_transformed.columns:
            raise KeyError(f"Отсутствует категориальный столбец '{col}' в данных")

        df_transformed[col] = df_transformed[col].astype(str)
        unknown_values = set(df_transformed[col]) - set(encoder.classes_)
        if unknown_values:
            raise ValueError(f"В колонке '{col}' найдены неизвестные категории: {unknown_values}")

        df_transformed[col] = encoder.transform(df_transformed[col])

    if not hasattr(scaler, "mean_"):
        raise TypeError("Переданный скейлер некорректен или не обучен")

    try:
        num_cols = scaler.feature_names_in_
    except AttributeError:
        num_cols = df_transformed.select_dtypes(include=[np.number]).columns

    df_transformed["price"] = 0

    missing_cols = [col for col in num_cols if col not in df_transformed.columns]
    if missing_cols:
        raise KeyError(f"Отсутствуют числовые колонки для масштабирования: {missing_cols}")

    df_transformed[num_cols] = scaler.transform(df_transformed[num_cols])

    return df_transformed.drop("price", axis=1)
