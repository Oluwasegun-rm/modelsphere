"""Automatic (Optuna) tuning for the two baseline models.

What "tuning" means here: each model has settings (how strongly to regularize,
how many trees...). Instead of picking them by hand, Optuna tries a small number
of combinations ("trials") and keeps the one that scored best.

How the held-out test rows stay out of it:

1. The data is split ONCE into training rows and test rows, exactly as
   ``train_model`` does it (same seed, same stratified split).
2. Every trial is judged by cross-validation on the TRAINING rows only: the
   training rows are cut into folds, and each fold takes a turn as the
   "practice test" while the model learns from the other folds. Imputation and
   scaling live inside the model pipeline, so they are fitted inside each fold.
3. The best settings are then handed to ``train_model``, which fits them on all
   training rows and scores the untouched test rows once.

The function returns the normal ``Result`` used by the app and by experiment
tracking. The tuning details are stored as plain JSON-safe values under
``result.parameters["tuning"]``. This module does not use Streamlit or SQLite.
"""

import dataclasses
from typing import Any

import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.datasets import DatasetSpec
from src.training import (
    Result,
    _is_integer,
    _validate_settings,
    build_estimator,
    load_features_and_target,
    split_indices,
    train_model,
)

DEFAULT_TRIALS = 12
MAX_TRIALS = 30  # keeps one click in the public app from running for minutes
DEFAULT_CV_FOLDS = 3
MIN_CV_FOLDS, MAX_CV_FOLDS = 2, 5
VALIDATION_METRIC = "roc_auc"

# The ranges Optuna may search. Kept small and sensible for a class demo.
LOGISTIC_C_RANGE = (0.001, 100.0)  # searched on a log scale
FOREST_TREES_RANGE = (20, 150)
FOREST_DEPTH_RANGE = (2, 20)


def _import_optuna():
    try:
        import optuna
    except ImportError as exc:  # the dashboard shows ValueError messages
        raise ValueError("Tuning needs the 'optuna' package. Run: pip install -r requirements.txt") from exc
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    return optuna


def _validate_tuning_settings(n_trials: Any, cv_folds: Any, y_train: np.ndarray) -> None:
    if not _is_integer(n_trials) or not 1 <= n_trials <= MAX_TRIALS:
        raise ValueError(f"n_trials must be a whole number from 1 to {MAX_TRIALS}, got {n_trials!r}")
    if not _is_integer(cv_folds) or not MIN_CV_FOLDS <= cv_folds <= MAX_CV_FOLDS:
        raise ValueError(f"cv_folds must be a whole number from {MIN_CV_FOLDS} to {MAX_CV_FOLDS}, got {cv_folds!r}")
    smallest_class = int(np.bincount(y_train).min())
    if cv_folds > smallest_class:
        raise ValueError(f"cv_folds ({cv_folds}) cannot be larger than the smallest class in the "
                         f"training rows ({smallest_class}). Use fewer folds or more data.")


def _search(model: str, x_train: np.ndarray, y_train: np.ndarray, *, n_trials: int,
            cv_folds: int, random_state: int) -> tuple[dict[str, Any], float, int]:
    """Look for good settings using ONLY the training rows it is given.

    Returns (best settings, best cross-validated score, number of finished trials).
    This function never sees the test rows, so it cannot tune on them.
    """
    optuna = _import_optuna()
    folds = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    def objective(trial) -> float:
        if model == "Logistic regression":
            settings = {"c": trial.suggest_float("c", *LOGISTIC_C_RANGE, log=True)}
            estimator = build_estimator(model, random_state=random_state, c=settings["c"],
                                        n_estimators=100, max_depth=None)
        else:
            settings = {
                "n_estimators": trial.suggest_int("n_estimators", *FOREST_TREES_RANGE),
                "max_depth": trial.suggest_int("max_depth", *FOREST_DEPTH_RANGE),
            }
            estimator = build_estimator(model, random_state=random_state, c=1.0, **settings)
        scores = cross_val_score(estimator, x_train, y_train, cv=folds,
                                 scoring=VALIDATION_METRIC, error_score="raise")
        return float(np.mean(scores))

    # A fixed sampler seed makes the same inputs choose the same settings again.
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=random_state))
    study.optimize(objective, n_trials=n_trials)
    finished = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    return dict(study.best_params), float(study.best_value), len(finished)


def tune_model(model: str, *, n_trials: int = DEFAULT_TRIALS, cv_folds: int = DEFAULT_CV_FOLDS,
               test_size: float = 0.2, random_state: int = 42,
               dataset: DatasetSpec | None = None) -> Result:
    """Tune one model with Optuna, then report scores on held-out test rows.

    Inputs: the model name ("Logistic regression" or "Random forest"), how many
    trials to try, how many cross-validation folds, the split settings, and an
    optional uploaded ``DatasetSpec`` (leave it out for the built-in dataset).

    Returns the usual ``Result``. ``metrics`` come from the untouched test rows.
    ``parameters`` holds the selected settings, the split settings, the dataset
    description (for uploads), and a ``"tuning"`` entry with the method, the
    trial count, the cross-validation setup and the training-only validation score.

    Raises ValueError for any setting that cannot be used.
    """
    # Reuse the training checks; the model-specific values are placeholders here.
    _validate_settings(model, test_size, random_state, 1.0, 100, None)
    test_size, random_state = float(test_size), int(random_state)

    features, target = load_features_and_target(dataset)
    train_rows, _ = split_indices(target, test_size, random_state)  # the test rows are not used here
    x_train, y_train = features[train_rows], target[train_rows]
    _validate_tuning_settings(n_trials, cv_folds, y_train)

    best, validation_score, finished = _search(
        model, x_train, y_train, n_trials=int(n_trials), cv_folds=int(cv_folds),
        random_state=random_state)

    # Fit the chosen settings on all training rows and score the test rows once.
    final = train_model(model, test_size=test_size, random_state=random_state,
                        dataset=dataset, **best)
    tuning = {
        "method": "Optuna TPE",
        "validation_metric": VALIDATION_METRIC,
        "validation_score": validation_score,
        "trials": finished,
        "cv_folds": int(cv_folds),
        "validated_on": "training rows only",
    }
    return dataclasses.replace(final, parameters={**final.parameters, "tuning": tuning})
