"""Leakage-safe training on the bundled Wisconsin breast cancer dataset."""

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier


@dataclass(frozen=True)
class Result:
    model: str
    parameters: dict[str, Any]
    metrics: dict[str, float]
    train_count: int
    test_count: int


def dataset_summary() -> dict[str, Any]:
    data = load_breast_cancer()
    return {"rows": len(data.data), "features": len(data.feature_names), "classes": list(data.target_names)}


def train_model(model: str, *, test_size: float = 0.2, random_state: int = 42,
                c: float = 1.0, n_estimators: int = 100, max_depth: int | None = None) -> Result:
    if not 0.1 <= test_size <= 0.4:
        raise ValueError("test_size must be between 0.1 and 0.4")
    data = load_breast_cancer()
    x_train, x_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=test_size, random_state=random_state,
        stratify=data.target,
    )
    if model == "Logistic regression":
        if c <= 0:
            raise ValueError("C must be positive")
        estimator = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=2000))
        parameters = {"C": c}
    elif model == "Random forest":
        if n_estimators < 1 or (max_depth is not None and max_depth < 1):
            raise ValueError("Invalid random forest parameters")
        estimator = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                           random_state=random_state, n_jobs=-1)
        parameters = {"n_estimators": n_estimators, "max_depth": max_depth}
    else:
        raise ValueError(f"Unsupported model: {model}")
    estimator.fit(x_train, y_train)
    predictions = estimator.predict(x_test)
    scores = estimator.predict_proba(x_test)[:, 1]
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, scores)),
    }
    return Result(model, parameters, metrics, len(y_train), len(y_test))
