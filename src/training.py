"""Leakage-safe training on the bundled dataset or a validated uploaded CSV.

Every metric treats encoded class 1 as positive. For the bundled dataset this
means benign; for an upload it means the class selected by the user.

Evaluation rules this module follows:

* The data is split once into a training part and a held-out test part
  (stratified, so both parts keep the same benign/malignant mix).
* Imputation and any scaling are fitted on the training part only.
* Every reported score is computed on the held-out test part, which the model
  never saw while learning.
* The same settings (including the random seed) always give the same split and
  the same scores.
"""

import math
import numbers
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.datasets import DatasetSpec

POSITIVE_CLASS = 1
POSITIVE_CLASS_NAME = "benign"

SUPPORTED_MODELS = ("Logistic regression", "Random forest")
MIN_TEST_SIZE, MAX_TEST_SIZE = 0.1, 0.4
MAX_SEED = 2**32 - 1  # the largest seed scikit-learn accepts
MAX_TREES = 1000  # keeps one click in the dashboard from running for minutes


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


def _is_integer(value: Any) -> bool:
    return isinstance(value, numbers.Integral) and not isinstance(value, bool)


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, numbers.Real) and not isinstance(value, bool) and math.isfinite(value)


def _validate_settings(model: str, test_size: Any, random_state: Any, c: Any,
                       n_estimators: Any, max_depth: Any) -> None:
    """Raise ValueError with a plain message if any setting cannot be used.

    Only ValueError is raised (never TypeError) because the dashboard shows
    ValueError messages to the user and would crash on anything else.
    Settings that belong to the other model are ignored, as before.
    """
    if model not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model: {model!r}. Choose one of: {', '.join(SUPPORTED_MODELS)}")
    if not _is_finite_number(test_size) or not MIN_TEST_SIZE <= test_size <= MAX_TEST_SIZE:
        raise ValueError(f"test_size must be a number between {MIN_TEST_SIZE} and {MAX_TEST_SIZE}, got {test_size!r}")
    if not _is_integer(random_state) or not 0 <= random_state <= MAX_SEED:
        raise ValueError(f"random_state must be a whole number from 0 to {MAX_SEED}, got {random_state!r}")
    if model == "Logistic regression":
        if not _is_finite_number(c) or c <= 0:
            raise ValueError(f"C must be a positive number, got {c!r}")
    else:
        if not _is_integer(n_estimators) or not 1 <= n_estimators <= MAX_TREES:
            raise ValueError(f"n_estimators must be a whole number from 1 to {MAX_TREES}, got {n_estimators!r}")
        if max_depth is not None and (not _is_integer(max_depth) or max_depth < 1):
            raise ValueError(f"max_depth must be None or a whole number of at least 1, got {max_depth!r}")


def split_indices(target: np.ndarray, test_size: float, random_state: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (train_rows, test_rows): a seeded, stratified split of row numbers.

    The two sets never overlap, and together they cover every row exactly once.
    """
    rows = np.arange(len(target))
    train_rows, test_rows = train_test_split(
        rows, test_size=test_size, random_state=random_state, stratify=target,
    )
    return train_rows, test_rows


def build_estimator(model: str, *, random_state: int, c: float, n_estimators: int,
                    max_depth: int | None):
    """Create an untrained model. Settings must already be validated."""
    if model == "Logistic regression":
        # Imputation and scaling are fitted on training rows only.
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             LogisticRegression(C=c, max_iter=2000))
    return make_pipeline(SimpleImputer(strategy="median"),
                         RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                                random_state=random_state, n_jobs=-1))


def _score_on_held_out(estimator: Any, x_test: np.ndarray, y_test: np.ndarray) -> dict[str, float]:
    """Score a trained model on rows it never saw. Encoded class 1 is positive."""
    predictions = estimator.predict(x_test)
    # Column 1 of predict_proba is the probability of encoded class 1.
    probabilities = estimator.predict_proba(x_test)[:, POSITIVE_CLASS]
    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, pos_label=POSITIVE_CLASS, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, pos_label=POSITIVE_CLASS, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, pos_label=POSITIVE_CLASS, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
    }


def train_model(model: str, *, test_size: float = 0.2, random_state: int = 42,
                c: float = 1.0, n_estimators: int = 100, max_depth: int | None = None,
                dataset: DatasetSpec | None = None) -> Result:
    """Train one model and report scores measured on held-out test data.

    Raises ValueError for any setting that is not valid.
    """
    _validate_settings(model, test_size, random_state, c, n_estimators, max_depth)
    # Turn numpy numbers into plain Python ones so the settings are JSON serializable.
    test_size, random_state, c = float(test_size), int(random_state), float(c)
    n_estimators = int(n_estimators)
    max_depth = None if max_depth is None else int(max_depth)

    if dataset is None:
        data = load_breast_cancer()
        features, target = data.data, data.target
    else:
        features, target = dataset.features.to_numpy(dtype=float), dataset.target
    train_rows, test_rows = split_indices(target, test_size, random_state)
    x_train, y_train = features[train_rows], target[train_rows]
    x_test, y_test = features[test_rows], target[test_rows]

    estimator = build_estimator(model, random_state=random_state, c=c,
                                n_estimators=n_estimators, max_depth=max_depth)
    estimator.fit(x_train, y_train)
    metrics = _score_on_held_out(estimator, x_test, y_test)

    if model == "Logistic regression":
        parameters: dict[str, Any] = {"C": c}
    else:
        parameters = {"n_estimators": n_estimators, "max_depth": max_depth}
    # Saving the split settings lets anyone repeat a saved experiment exactly.
    parameters.update({"test_size": test_size, "random_state": random_state})
    if dataset is not None:
        parameters["dataset"] = dataset.metadata()
    return Result(model, parameters, metrics, len(train_rows), len(test_rows))
