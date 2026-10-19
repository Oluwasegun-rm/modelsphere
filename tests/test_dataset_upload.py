"""The uploaded-data path uses the same training and tracking contract as the demo."""

import numpy as np
import pandas as pd
import pytest

from src.datasets import MAX_CSV_BYTES, prepare_dataset, read_csv
from src.tracking import list_experiments, save_experiment
from src.training import train_model


def example_frame() -> pd.DataFrame:
    frame = pd.DataFrame({
        "size": [float(i % 20) for i in range(60)],
        "texture": [float((i * 3) % 17) for i in range(60)],
        "label": ["yes"] * 30 + ["no"] * 30,
        "record_id": [f"record-{i}" for i in range(60)],
    })
    frame.loc[0, "size"] = np.nan
    frame.loc[1, "label"] = None
    return frame


def test_csv_inspection_and_training_with_missing_values(tmp_path):
    frame = read_csv(example_frame().to_csv(index=False).encode("utf-8"))
    dataset = prepare_dataset(frame, name="samples.csv", target_name="label",
                              feature_names=["size", "texture"], positive_label="yes")
    assert len(dataset.target) == 59
    assert dataset.excluded_target_rows == 1
    assert dataset.missing_by_feature["size"] == 1
    assert dataset.metadata()["features"] == ["size", "texture"]

    for model in ("Logistic regression", "Random forest"):
        result = train_model(model, dataset=dataset, n_estimators=10)
        assert result.parameters["dataset"]["id"] == dataset.dataset_id
        assert result.parameters["dataset"]["positive_label"] == "yes"
        assert result.train_count + result.test_count == 59
        assert all(0 <= value <= 1 for value in result.metrics.values())
        saved_id = save_experiment(result, tmp_path / "runs.db")
        assert list_experiments(tmp_path / "runs.db")[0]["id"] == saved_id


def test_dataset_identity_tracks_selection_and_positive_class():
    frame = example_frame()
    first = prepare_dataset(frame, name="a.csv", target_name="label",
                            feature_names=["size", "texture"], positive_label="yes")
    renamed = prepare_dataset(frame, name="b.csv", target_name="label",
                              feature_names=["size", "texture"], positive_label="yes")
    other_class = prepare_dataset(frame, name="a.csv", target_name="label",
                                  feature_names=["size", "texture"], positive_label="no")
    fewer_features = prepare_dataset(frame, name="a.csv", target_name="label",
                                     feature_names=["size"], positive_label="yes")
    assert first.dataset_id == renamed.dataset_id
    assert len({first.dataset_id, other_class.dataset_id, fewer_features.dataset_id}) == 3


def test_reject_unsupported_data_with_explanations():
    frame = example_frame()
    with pytest.raises(ValueError, match="numeric"):
        prepare_dataset(frame, name="a.csv", target_name="label",
                        feature_names=["record_id"], positive_label="yes")
    frame["empty"] = np.nan
    with pytest.raises(ValueError, match="entirely empty"):
        prepare_dataset(frame, name="a.csv", target_name="label",
                        feature_names=["empty"], positive_label="yes")
    frame.loc[2, "label"] = "maybe"
    with pytest.raises(ValueError, match="exactly two"):
        prepare_dataset(frame, name="a.csv", target_name="label",
                        feature_names=["size"], positive_label="yes")


def test_reject_empty_or_oversized_upload():
    with pytest.raises(ValueError, match="empty"):
        read_csv(b"")
    with pytest.raises(ValueError, match="smaller than 5 MB"):
        read_csv(b"x" * (MAX_CSV_BYTES + 1))
