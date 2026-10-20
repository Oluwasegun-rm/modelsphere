"""PyTorch model uses the shared holdout and returns trackable test scores."""

import pytest

from src.neural import MODEL_NAME, train_neural_model


def test_small_neural_model_returns_held_out_scores():
    result = train_neural_model(epochs=3, hidden_units=8, random_state=42)

    assert result.model == MODEL_NAME
    assert (result.train_count, result.test_count) == (455, 114)
    assert result.parameters == {
        "epochs": 3, "hidden_units": 8, "learning_rate": 0.001,
        "test_size": 0.2, "random_state": 42,
    }
    assert set(result.metrics) == {"accuracy", "precision", "recall", "f1", "roc_auc"}
    assert all(0 <= score <= 1 for score in result.metrics.values())


def test_neural_settings_are_bounded():
    with pytest.raises(ValueError, match="Epochs"):
        train_neural_model(epochs=1000)
