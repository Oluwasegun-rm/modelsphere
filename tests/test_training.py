import dataclasses
import json
import math

import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import roc_auc_score

from src.tracking import list_experiments, save_experiment
from src.training import (
    MAX_SEED,
    MAX_TREES,
    POSITIVE_CLASS,
    POSITIVE_CLASS_NAME,
    Result,
    build_estimator,
    dataset_summary,
    split_indices,
    train_model,
)

MODELS = ("Logistic regression", "Random forest")


def test_dataset_and_models():
    assert dataset_summary()["rows"] == 569
    for model in MODELS:
        result = train_model(model, n_estimators=10)
        assert result.train_count + result.test_count == 569
        assert set(result.metrics) == {"accuracy", "precision", "recall", "f1", "roc_auc"}
        assert all(0 <= score <= 1 for score in result.metrics.values())


def test_bad_model_rejected():
    with pytest.raises(ValueError):
        train_model("unsupported")


# --- Positive class -------------------------------------------------------

def test_positive_class_is_benign():
    """Class 1 is benign, so precision/recall/F1 are about finding benign cases."""
    data = load_breast_cancer()
    assert POSITIVE_CLASS == 1
    assert data.target_names[POSITIVE_CLASS] == POSITIVE_CLASS_NAME == "benign"


# --- Settings are checked --------------------------------------------------

@pytest.mark.parametrize("model, settings", [
    ("Logistic regression", {"test_size": 0.05}),
    ("Logistic regression", {"test_size": 0.5}),
    ("Logistic regression", {"test_size": float("nan")}),
    ("Logistic regression", {"test_size": "0.2"}),
    ("Logistic regression", {"test_size": True}),
    ("Logistic regression", {"random_state": -1}),
    ("Logistic regression", {"random_state": 1.5}),
    ("Logistic regression", {"random_state": True}),
    ("Logistic regression", {"random_state": MAX_SEED + 1}),
    ("Logistic regression", {"c": 0}),
    ("Logistic regression", {"c": -1.0}),
    ("Logistic regression", {"c": float("nan")}),
    ("Logistic regression", {"c": float("inf")}),
    ("Logistic regression", {"c": "1"}),
    ("Logistic regression", {"c": True}),
    ("Random forest", {"n_estimators": 0}),
    ("Random forest", {"n_estimators": -5}),
    ("Random forest", {"n_estimators": 2.5}),
    ("Random forest", {"n_estimators": MAX_TREES + 1}),
    ("Random forest", {"n_estimators": True}),
    ("Random forest", {"max_depth": 0}),
    ("Random forest", {"max_depth": -3}),
    ("Random forest", {"max_depth": 2.5}),
    ("Random forest", {"max_depth": "5"}),
])
def test_invalid_settings_raise_value_error(model, settings):
    # The dashboard only catches ValueError, so bad input must never raise anything else.
    with pytest.raises(ValueError):
        train_model(model, **settings)


def test_edge_values_that_are_allowed():
    train_model("Logistic regression", test_size=0.1, random_state=0, c=0.01)
    train_model("Logistic regression", test_size=0.4, random_state=MAX_SEED, c=100.0)
    train_model("Random forest", n_estimators=1, max_depth=1)
    train_model("Random forest", n_estimators=5, max_depth=None)


# --- Results are repeatable ------------------------------------------------

@pytest.mark.parametrize("model", MODELS)
def test_same_settings_give_same_results(model):
    first = train_model(model, random_state=7, n_estimators=25)
    second = train_model(model, random_state=7, n_estimators=25)
    assert first == second


def test_seed_controls_the_split():
    target = load_breast_cancer().target
    train_a, test_a = split_indices(target, 0.2, 1)
    train_b, test_b = split_indices(target, 0.2, 1)
    _, test_other_seed = split_indices(target, 0.2, 2)
    assert np.array_equal(train_a, train_b) and np.array_equal(test_a, test_b)
    assert not np.array_equal(test_a, test_other_seed)


def test_saved_parameters_can_repeat_the_run():
    """Parameters store the seed and split size and stay JSON serializable."""
    # numpy numbers are what some widgets return; they must still be saved cleanly.
    result = train_model("Random forest", test_size=np.float64(0.25), random_state=np.int64(3),
                         n_estimators=np.int64(20), max_depth=np.int64(4))
    assert json.loads(json.dumps(result.parameters)) == {
        "n_estimators": 20, "max_depth": 4, "test_size": 0.25, "random_state": 3,
    }
    repeat = train_model("Random forest", **{
        "test_size": result.parameters["test_size"], "random_state": result.parameters["random_state"],
        "n_estimators": result.parameters["n_estimators"], "max_depth": result.parameters["max_depth"],
    })
    assert repeat == result


# --- Scores come from unseen data -------------------------------------------

@pytest.mark.parametrize("test_size", [0.1, 0.2, 0.4])
def test_split_is_separate_complete_and_stratified(test_size):
    target = load_breast_cancer().target
    train_rows, test_rows = split_indices(target, test_size, 42)
    assert set(train_rows).isdisjoint(test_rows)  # no row is used for both training and testing
    assert sorted(np.concatenate([train_rows, test_rows])) == list(range(len(target)))
    assert len(test_rows) == math.ceil(test_size * len(target))
    # Stratified: the share of benign cases is nearly the same in both parts.
    overall = target.mean()
    assert abs(target[train_rows].mean() - overall) < 0.01
    assert abs(target[test_rows].mean() - overall) < 0.01


@pytest.mark.parametrize("model", MODELS)
def test_reported_scores_come_from_the_test_rows(model):
    """Recompute the scores by hand: train on training rows, score on test rows."""
    data = load_breast_cancer()
    result = train_model(model, test_size=0.2, random_state=42, n_estimators=25)

    train_rows, test_rows = split_indices(data.target, 0.2, 42)
    estimator = build_estimator(model, random_state=42, c=1.0, n_estimators=25, max_depth=None)
    estimator.fit(data.data[train_rows], data.target[train_rows])
    x_test, y_test = data.data[test_rows], data.target[test_rows]

    assert result.metrics["accuracy"] == pytest.approx(estimator.score(x_test, y_test))
    expected_auc = roc_auc_score(y_test, estimator.predict_proba(x_test)[:, POSITIVE_CLASS])
    assert result.metrics["roc_auc"] == pytest.approx(expected_auc)
    assert result.test_count == len(test_rows)
    assert result.train_count == len(train_rows)


def test_reported_accuracy_is_not_the_training_accuracy():
    """A random forest memorizes its training rows (100% there), so a reported score
    below that shows the numbers were not measured on the training data."""
    data = load_breast_cancer()
    train_rows, _ = split_indices(data.target, 0.2, 42)
    estimator = build_estimator("Random forest", random_state=42, c=1.0, n_estimators=100, max_depth=None)
    estimator.fit(data.data[train_rows], data.target[train_rows])
    training_accuracy = estimator.score(data.data[train_rows], data.target[train_rows])

    result = train_model("Random forest", random_state=42, n_estimators=100)
    assert training_accuracy == 1.0
    assert result.metrics["accuracy"] < training_accuracy


# --- The dashboard can still use it ------------------------------------------

def test_result_shape_and_call_match_the_dashboard(tmp_path):
    """app.py calls train_model with these keywords and saves the Result."""
    assert [f.name for f in dataclasses.fields(Result)] == [
        "model", "parameters", "metrics", "train_count", "test_count",
    ]
    result = train_model("Logistic regression", test_size=0.2, random_state=42, c=1.0,
                         n_estimators=100, max_depth=None)
    database = tmp_path / "experiments.db"
    run_id = save_experiment(result, database)
    saved = list_experiments(database)
    assert saved[0]["id"] == run_id
    assert saved[0]["model"] == "Logistic regression"
    assert saved[0]["metrics"] == result.metrics
    assert saved[0]["parameters"] == result.parameters
