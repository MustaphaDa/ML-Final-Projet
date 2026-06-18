"""Feature preprocessing and regression metrics."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def get_numeric_columns(df: pd.DataFrame, exclude: list[str] | None = None) -> list[str]:
    exclude = set(exclude or [])
    return [col for col in df.select_dtypes(include=["number"]).columns if col not in exclude]


def get_categorical_columns(
    df: pd.DataFrame,
    exclude: list[str] | None = None,
    max_cardinality: int = 50,
) -> list[str]:
    exclude = set(exclude or [])
    candidates = df.select_dtypes(include=["object", "category"]).columns
    return [col for col in candidates if col not in exclude and df[col].nunique() <= max_cardinality]


def build_tabular_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": r2_score(y_true, y_pred),
    }


def print_metrics(metrics: dict[str, float], label: str = "Model") -> None:
    print(f"{label} metrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")
