"""Small, inspectable CSV input for binary classification experiments."""

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO

import numpy as np
import pandas as pd

MAX_CSV_BYTES = 5 * 1024 * 1024
MAX_ROWS = 10_000
MAX_FEATURES = 50
BUILTIN_DATASET_ID = "breast-cancer-wisconsin-v1"


@dataclass(frozen=True)
class DatasetSpec:
    features: pd.DataFrame
    target: np.ndarray
    dataset_id: str
    name: str
    target_name: str
    positive_label: str
    feature_names: tuple[str, ...]
    excluded_target_rows: int
    duplicate_rows: int
    missing_by_feature: dict[str, int]

    def metadata(self) -> dict:
        """Only descriptive details are saved; uploaded rows stay in memory."""
        return {
            "id": self.dataset_id,
            "name": self.name,
            "target": self.target_name,
            "positive_label": self.positive_label,
            "features": list(self.feature_names),
        }


def read_csv(contents: bytes) -> pd.DataFrame:
    """Read a small UTF-8 CSV, raising a useful error for unsupported files."""
    if not contents:
        raise ValueError("The CSV file is empty.")
    if len(contents) > MAX_CSV_BYTES:
        raise ValueError("Use a CSV file smaller than 5 MB for this demo.")
    try:
        frame = pd.read_csv(BytesIO(contents), encoding="utf-8-sig")
    except (UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise ValueError("Could not read this CSV. Save it as a UTF-8 CSV with a header row.") from exc
    if frame.empty or len(frame.columns) < 2:
        raise ValueError("The CSV needs data rows and at least two columns.")
    if len(frame) > MAX_ROWS:
        raise ValueError("Use at most 10,000 rows for this demo.")
    return frame


def prepare_dataset(frame: pd.DataFrame, *, name: str, target_name: str,
                    feature_names: list[str], positive_label: str) -> DatasetSpec:
    """Validate selected columns and encode the chosen positive class as 1."""
    if target_name not in frame.columns:
        raise ValueError("Choose a target column from the CSV.")
    if not feature_names or len(feature_names) > MAX_FEATURES:
        raise ValueError("Choose between 1 and 50 numeric feature columns.")
    if len(set(feature_names)) != len(feature_names) or target_name in feature_names:
        raise ValueError("Choose distinct feature columns, excluding the target.")
    for column in feature_names:
        if column not in frame.columns or not pd.api.types.is_numeric_dtype(frame[column]):
            raise ValueError(f"Feature '{column}' must be a numeric column.")

    labels = frame[target_name].astype("string").str.strip().replace("", pd.NA)
    valid_target = labels.notna()
    excluded = int((~valid_target).sum())
    clean_labels = labels.loc[valid_target]
    classes = clean_labels.unique().tolist()
    if len(classes) != 2:
        raise ValueError("The selected target must have exactly two non-empty classes.")
    if positive_label not in classes:
        raise ValueError("Choose one of the target's two classes as positive.")
    class_counts = clean_labels.value_counts()
    if len(clean_labels) < 40 or int(class_counts.min()) < 10:
        raise ValueError("Use at least 40 labelled rows, with at least 10 in each class.")

    features = frame.loc[valid_target, feature_names].copy()
    missing_by_feature: dict[str, int] = {}
    for column in feature_names:
        features[column] = pd.to_numeric(features[column]).astype(float)
        values = features[column].to_numpy()
        if np.isinf(values).any():
            raise ValueError(f"Feature '{column}' contains infinity; replace it before upload.")
        missing = int(features[column].isna().sum())
        if missing == len(features):
            raise ValueError(f"Feature '{column}' is entirely empty; deselect it.")
        missing_by_feature[column] = missing

    duplicates = int(pd.concat([features, clean_labels.rename(target_name)], axis=1).duplicated().sum())
    fingerprint_source = (target_name + "\n" + positive_label + "\n" +
                          pd.concat([features, clean_labels.rename(target_name)], axis=1)
                          .to_csv(index=False))
    fingerprint = sha256(fingerprint_source.encode("utf-8")).hexdigest()
    safe_name = str(name).replace("\\", "/").split("/")[-1][:80] or "uploaded.csv"
    return DatasetSpec(
        features=features,
        target=(clean_labels == positive_label).astype(int).to_numpy(),
        dataset_id=fingerprint,
        name=safe_name,
        target_name=target_name,
        positive_label=positive_label,
        feature_names=tuple(feature_names),
        excluded_target_rows=excluded,
        duplicate_rows=duplicates,
        missing_by_feature=missing_by_feature,
    )
