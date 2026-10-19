import pytest

from src.training import dataset_summary, train_model


def test_dataset_and_models():
    assert dataset_summary()["rows"] == 569
    for model in ("Logistic regression", "Random forest"):
        result = train_model(model, n_estimators=10)
        assert result.train_count + result.test_count == 569
        assert set(result.metrics) == {"accuracy", "precision", "recall", "f1", "roc_auc"}
        assert all(0 <= score <= 1 for score in result.metrics.values())


def test_bad_model_rejected():
    with pytest.raises(ValueError):
        train_model("unsupported")
