"""Tests for Optuna tuning: both models work, results are repeatable and
JSON-safe, and the held-out test rows never take part in the search."""

import importlib.util
import json
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_breast_cancer

from src import tuning
from src.datasets import prepare_dataset
from src.tracking import list_experiments, save_experiment
from src.training import Result, split_indices, train_model
from src.tuning import DEFAULT_TRIALS, MAX_TRIALS, tune_model

MODELS = ("Logistic regression", "Random forest")
requires_optuna = pytest.mark.skipif(importlib.util.find_spec("optuna") is None,
                                     reason="optuna is not installed")


def uploaded_dataset():
    """A small two-feature dataset like a student CSV, with a real signal in it."""
    rng = np.random.default_rng(0)
    label = np.array(["yes"] * 60 + ["no"] * 60)
    frame = pd.DataFrame({
        "size": np.where(label == "yes", 6.0, 4.0) + rng.normal(0, 1.5, 120),
        "texture": rng.normal(10, 3, 120),
        "label": label,
    })
    return prepare_dataset(frame, name="samples.csv", target_name="label",
                           feature_names=["size", "texture"], positive_label="yes")


def selected_settings(result: Result) -> dict:
    """The model settings tuning chose, in the form train_model accepts."""
    parameters = result.parameters
    if result.model == "Logistic regression":
        return {"c": parameters["C"]}
    return {"n_estimators": parameters["n_estimators"], "max_depth": parameters["max_depth"]}


@requires_optuna
@pytest.mark.parametrize("model", MODELS)
def test_both_models_return_a_normal_result(model):
    result = tune_model(model, n_trials=4)
    assert isinstance(result, Result)
    assert result.model == model
    assert set(result.metrics) == {"accuracy", "precision", "recall", "f1", "roc_auc"}
    assert all(0 <= score <= 1 for score in result.metrics.values())
    assert result.train_count + result.test_count == 569

    details = result.parameters["tuning"]
    assert details["trials"] == 4
    assert details["cv_folds"] == 3
    assert details["validated_on"] == "training rows only"
    assert 0 <= details["validation_score"] <= 1
    # Split settings are still saved so the dashboard can match comparable runs.
    assert result.parameters["test_size"] == 0.2 and result.parameters["random_state"] == 42
    # Everything must survive being written as JSON (what tracking does).
    assert json.loads(json.dumps(result.parameters)) == result.parameters


@requires_optuna
@pytest.mark.parametrize("model", MODELS)
def test_same_inputs_choose_the_same_settings(model):
    assert tune_model(model, n_trials=4, random_state=7) == tune_model(model, n_trials=4, random_state=7)


@requires_optuna
@pytest.mark.parametrize("model", MODELS)
def test_test_rows_are_never_used_while_searching(model, monkeypatch):
    """Record every row cross-validation sees: all must be training rows."""
    data = load_breast_cancer()
    train_rows, test_rows = split_indices(data.target, 0.2, 42)
    test_rows_as_bytes = {data.data[i].tobytes() for i in test_rows}
    train_rows_as_bytes = {data.data[i].tobytes() for i in train_rows}
    seen = []
    real_cross_val_score = tuning.cross_val_score

    def spy(estimator, x, y, **kwargs):
        seen.extend(row.tobytes() for row in np.asarray(x))
        return real_cross_val_score(estimator, x, y, **kwargs)

    monkeypatch.setattr(tuning, "cross_val_score", spy)
    tune_model(model, n_trials=3, test_size=0.2, random_state=42)

    assert seen, "the search should have run cross-validation"
    assert set(seen) <= train_rows_as_bytes
    assert set(seen).isdisjoint(test_rows_as_bytes)


@requires_optuna
@pytest.mark.parametrize("model", MODELS)
def test_reported_scores_are_one_final_fit_scored_on_the_test_rows(model):
    """The reported metrics must equal a normal train_model run with the chosen
    settings, so the test rows are scored once, after the choice is made."""
    result = tune_model(model, n_trials=3)
    plain = train_model(model, **selected_settings(result))
    assert result.metrics == plain.metrics
    assert (result.train_count, result.test_count) == (plain.train_count, plain.test_count)


@requires_optuna
def test_chosen_settings_stay_inside_the_search_ranges():
    low, high = tuning.LOGISTIC_C_RANGE
    assert low <= tune_model("Logistic regression", n_trials=4).parameters["C"] <= high
    forest = tune_model("Random forest", n_trials=4).parameters
    assert tuning.FOREST_TREES_RANGE[0] <= forest["n_estimators"] <= tuning.FOREST_TREES_RANGE[1]
    assert tuning.FOREST_DEPTH_RANGE[0] <= forest["max_depth"] <= tuning.FOREST_DEPTH_RANGE[1]


@requires_optuna
@pytest.mark.parametrize("model", MODELS)
def test_uploaded_dataset_keeps_its_description_and_saves_through_tracking(model, tmp_path):
    dataset = uploaded_dataset()
    result = tune_model(model, n_trials=3, dataset=dataset)
    assert result.parameters["dataset"] == dataset.metadata()
    assert result.train_count + result.test_count == len(dataset.target)
    assert all(0 <= score <= 1 for score in result.metrics.values())

    database = tmp_path / "experiments.db"
    run_id = save_experiment(result, database)
    saved = list_experiments(database)[0]
    assert saved["id"] == run_id
    assert saved["parameters"] == result.parameters
    assert saved["metrics"] == result.metrics
    # Only a description of the data is saved, never its rows.
    assert "rows" not in saved["parameters"]["dataset"]


@requires_optuna
def test_built_in_result_saves_through_tracking(tmp_path):
    result = tune_model("Logistic regression", n_trials=3)
    database = tmp_path / "experiments.db"
    save_experiment(result, database)
    assert list_experiments(database)[0]["parameters"]["tuning"]["trials"] == 3


def test_default_search_is_small_enough_for_the_public_app():
    assert 10 <= DEFAULT_TRIALS <= 15
    assert DEFAULT_TRIALS <= MAX_TRIALS <= 30


@pytest.mark.parametrize("settings", [
    {"n_trials": 0}, {"n_trials": -1}, {"n_trials": MAX_TRIALS + 1}, {"n_trials": 2.5},
    {"n_trials": "5"}, {"n_trials": True},
    {"cv_folds": 1}, {"cv_folds": 6}, {"cv_folds": 2.5}, {"cv_folds": True},
    {"test_size": 0.9}, {"test_size": float("nan")}, {"random_state": -1}, {"random_state": 1.5},
])
def test_invalid_settings_raise_value_error_before_any_search(settings, monkeypatch):
    # The dashboard only catches ValueError, and nothing should be searched first.
    monkeypatch.setattr(tuning, "_search", lambda *a, **k: pytest.fail("search should not start"))
    with pytest.raises(ValueError):
        tune_model("Logistic regression", **settings)


def test_unsupported_model_is_rejected():
    with pytest.raises(ValueError):
        tune_model("Neural network")


def test_missing_optuna_gives_a_plain_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "optuna", None)  # makes "import optuna" fail
    with pytest.raises(ValueError, match="optuna"):
        tune_model("Logistic regression", n_trials=1)
