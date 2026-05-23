"""Shared training utilities for the wine quality examples."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
from joblib import dump
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor

DATA_URL = (
    "http://archive.ics.uci.edu/ml/machine-learning-databases/"
    "wine-quality/winequality-red.csv"
)
RANDOM_STATE = 123456


@dataclass(frozen=True)
class Metrics:
    """Regression metrics for a trained model."""

    mse: float
    mae: float
    r2: float


@dataclass(frozen=True)
class ModelResult:
    """Trained estimator and its train/test metrics."""

    estimator: object
    train_metrics: Metrics
    test_metrics: Metrics


def load_data(url: str = DATA_URL) -> pd.DataFrame:
    """Load the red wine quality data set."""

    return pd.read_csv(url, sep=";")


def split_features_target(df: pd.DataFrame):
    """Separate features from the target column."""

    y = df["quality"]
    x = df.drop(columns=["quality"])
    return x, y


def make_train_test_split(df: pd.DataFrame):
    """Create the train/test split used across the assignment."""

    x, y = split_features_target(df)
    return train_test_split(
        x,
        y,
        test_size=0.25,
        random_state=RANDOM_STATE,
    )


def evaluate(estimator, x, y) -> Metrics:
    """Evaluate an estimator with common regression metrics."""

    y_pred = estimator.predict(x)
    return Metrics(
        mse=mean_squared_error(y, y_pred),
        mae=mean_absolute_error(y, y_pred),
        r2=r2_score(y, y_pred),
    )


def fit_and_evaluate(estimator, x_train, x_test, y_train, y_test) -> ModelResult:
    """Fit an estimator and return metrics for train and test data."""

    estimator.fit(x_train, y_train)
    return ModelResult(
        estimator=estimator,
        train_metrics=evaluate(estimator, x_train, y_train),
        test_metrics=evaluate(estimator, x_test, y_test),
    )


def train_best_knn(
    neighbors: Iterable[int] = (1, 3, 5, 7, 9, 11, 15),
    df: pd.DataFrame | None = None,
) -> ModelResult:
    """Train KNN candidates and keep the one with the lowest test MSE."""

    data = load_data() if df is None else df
    x_train, x_test, y_train, y_test = make_train_test_split(data)
    results = [
        fit_and_evaluate(
            KNeighborsRegressor(n_neighbors=n), x_train, x_test, y_train, y_test
        )
        for n in neighbors
    ]
    return min(results, key=lambda result: result.test_metrics.mse)


def train_best_elasticnet(
    params: Iterable[tuple[float, float]] = (
        (0.5, 0.5),
        (0.2, 0.2),
        (0.1, 0.1),
        (0.1, 0.05),
        (0.3, 0.2),
    ),
    df: pd.DataFrame | None = None,
) -> ModelResult:
    """Train ElasticNet candidates and keep the one with the lowest test MSE."""

    data = load_data() if df is None else df
    x_train, x_test, y_train, y_test = make_train_test_split(data)
    results = [
        fit_and_evaluate(
            ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=12345),
            x_train,
            x_test,
            y_train,
            y_test,
        )
        for alpha, l1_ratio in params
    ]
    return min(results, key=lambda result: result.test_metrics.mse)


def save_estimator(estimator, path: str | Path = "models/estimator.pkl") -> Path:
    """Persist the trained estimator."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dump(estimator, output_path)
    return output_path


def print_result(result: ModelResult) -> None:
    """Print model metrics in the original homework format."""

    print()
    print(result.estimator, ":", sep="")
    print()
    print("Metricas de entrenamiento:")
    print(f"  MSE: {result.train_metrics.mse}")
    print(f"  MAE: {result.train_metrics.mae}")
    print(f"  R2: {result.train_metrics.r2}")
    print()
    print("Metricas de testing:")
    print(f"  MSE: {result.test_metrics.mse}")
    print(f"  MAE: {result.test_metrics.mae}")
    print(f"  R2: {result.test_metrics.r2}")
