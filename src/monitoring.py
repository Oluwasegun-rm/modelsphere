"""A simple, transparent data-drift signal for numeric input features.

Question answered: *do the numbers in a new batch of data look different from a
reference batch?*  Typical use: the reference is the data a model was trained
on and the new batch is data it is about to be used on.

    report = detect_numeric_drift(reference_frame, new_frame)
    report.status    # "warning", "watch", "no_signal" or "not_enough_data"
    report.message   # one plain-language sentence
    report.features  # one FeatureDrift row per compared feature

This module needs only pandas and numpy. It does not use Streamlit or SQLite and
it never stores or sends data rows.

How each feature is compared
----------------------------
Every numeric feature is checked on its own with the two-sample
Kolmogorov-Smirnov (KS) gap. Sort the reference values and the new values; at
each point, find the share of reference values at or below it and the share of
new values at or below it. The KS gap is the largest difference between those
two shares. It runs from 0 (same distribution shape) to 1 (no overlap).

A feature is flagged as "drift" when its KS gap is larger than a threshold, and
the threshold is the larger of two numbers:

* ``1.358 * sqrt((n + m) / (n * m))``, where n and m are the numbers of usable
  values in the two batches. This is the standard 5% cut-off for "bigger than
  random sampling luck", so small batches need a bigger gap before they are
  flagged.
* ``MIN_EFFECT`` (0.10), so that a very large batch does not flag a difference
  too small to matter.

Overall result
--------------
* ``warning``: at least ``WARNING_SHARE`` (20%) of the checked features are
  flagged AND at least ``MIN_FLAGGED_FOR_WARNING`` (2) features are flagged (or
  the only checked feature is flagged). These are rules of thumb, not statistical
  results. They exist because a few features can look different by chance alone:
  without the two-feature minimum, one chance flag among 3-5 features gave a false
  overall warning 11-19% of the time on identical data. With it, measured false
  warnings on identical random data were about 5% or less. On random halves of the
  built-in dataset (no real drift) it was roughly 4-8% across repeated runs, because
  its 30 features are strongly linked and chance flags come in clusters. An
  occasional false warning is therefore expected. A partial real shift (8 of 30 features moved by
  0.75 standard deviations) was caught in every trial.
* ``watch``: some features are flagged but not enough for a warning, or columns of
  the reference batch were missing from the new batch.
* ``no_signal``: no checked feature is flagged.
* ``not_enough_data``: no feature could be checked.

Missing values and other edge cases
-----------------------------------
* NaN and infinite values are left out of each feature's comparison and counted
  in ``reference_missing`` / ``new_missing``. A change in how often values are
  missing is reported through those counts but is not scored as drift.
* A feature needs at least ``MIN_SAMPLES`` (30) usable values in each batch,
  otherwise it is reported as ``not_enough_data`` and not counted either way.
* A constant feature (every value identical) is handled without dividing by
  zero: the mean shift in standard deviations is left empty (None). If the new
  batch has a different value or spread, the KS gap still flags it.
* Only columns present in both batches and numeric in both are compared. Columns
  missing from the new batch, extra columns, and non-numeric columns are listed
  in the report rather than silently ignored. If no column can be compared, a
  ValueError explains why.

What this does NOT tell you
---------------------------
It is a drift signal on input features only. It is not a prediction of future
drift, not a measurement of model accuracy (that needs labels for the new batch),
and not production monitoring. Features are checked one at a time, so changes in
how features relate to each other are not detected.
"""

import json
import math
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

MIN_SAMPLES = 30
KS_CRITICAL_COEFFICIENT = 1.358  # 5% significance level for the two-sample KS test
MIN_EFFECT = 0.10
WARNING_SHARE = 0.20
MIN_FLAGGED_FOR_WARNING = 2

LIMITS = (
    "This compares the distribution of each numeric input feature in a new batch with a "
    "reference batch. It is a drift signal, not a prediction of future drift, not a "
    "measurement of model accuracy (that needs labels for the new batch), and not "
    "production monitoring. Features are checked one at a time, so changes in how "
    "features relate to each other are not detected."
)


@dataclass(frozen=True)
class FeatureDrift:
    """The comparison result for one feature."""

    feature: str
    status: str  # "drift", "no_drift" or "not_enough_data"
    ks_statistic: float | None
    threshold: float | None
    reference_mean: float | None
    new_mean: float | None
    mean_shift_in_std: float | None  # (new mean - reference mean) / reference std
    reference_count: int  # usable values
    new_count: int
    reference_missing: int
    new_missing: int
    note: str


@dataclass(frozen=True)
class DriftReport:
    """Per-feature results plus an overall summary. ``to_dict()`` is JSON-safe."""

    status: str  # "warning", "watch", "no_signal" or "not_enough_data"
    message: str
    reference_rows: int
    new_rows: int
    features_compared: int
    features_checked: int
    features_flagged: int
    share_flagged: float | None
    features: tuple[FeatureDrift, ...]
    missing_in_new: tuple[str, ...]
    extra_in_new: tuple[str, ...]
    skipped_non_numeric: tuple[str, ...]
    limits: str = LIMITS

    def to_dict(self) -> dict:
        return json.loads(json.dumps(asdict(self), allow_nan=False))


def ks_statistic(reference: np.ndarray, new: np.ndarray) -> float:
    """Largest gap between the two empirical distributions (0 to 1). Ties are handled."""
    reference = np.sort(np.asarray(reference, dtype=float))
    new = np.sort(np.asarray(new, dtype=float))
    if reference.size == 0 or new.size == 0:
        raise ValueError("Both batches need at least one value.")
    points = np.concatenate([reference, new])
    reference_share = np.searchsorted(reference, points, side="right") / reference.size
    new_share = np.searchsorted(new, points, side="right") / new.size
    return float(np.max(np.abs(reference_share - new_share)))


def ks_threshold(reference_count: int, new_count: int) -> float:
    """The gap a feature must exceed to be flagged (see the module docstring)."""
    luck = KS_CRITICAL_COEFFICIENT * math.sqrt((reference_count + new_count) / (reference_count * new_count))
    return max(luck, MIN_EFFECT)


def _is_numeric(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)


def _usable_values(series: pd.Series) -> np.ndarray:
    """Finite values only; NaN, pd.NA and infinity count as missing."""
    values = series.to_numpy(dtype=float, na_value=np.nan)
    return values[np.isfinite(values)]


def _check_frame(frame, label: str) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"The {label} batch must be a pandas DataFrame.")
    if frame.shape[0] == 0 or frame.shape[1] == 0:
        raise ValueError(f"The {label} batch is empty.")
    if frame.columns.duplicated().any():
        raise ValueError(f"The {label} batch has duplicate column names.")


def _choose_features(reference: pd.DataFrame, new: pd.DataFrame, features):
    """Return (features to compare, missing in new, extra in new, skipped non-numeric)."""
    missing_in_new = tuple(str(c) for c in reference.columns if c not in new.columns)
    extra_in_new = tuple(str(c) for c in new.columns if c not in reference.columns)

    if features is not None:
        chosen = list(features)
        if not chosen or len(set(chosen)) != len(chosen):
            raise ValueError("features must be a non-empty list of distinct column names.")
        absent = [str(c) for c in chosen if c not in reference.columns or c not in new.columns]
        if absent:
            raise ValueError(f"These features are not in both batches: {', '.join(absent)}.")
        text = [str(c) for c in chosen if not (_is_numeric(reference[c]) and _is_numeric(new[c]))]
        if text:
            raise ValueError(f"These features are not numeric in both batches: {', '.join(text)}.")
        return chosen, missing_in_new, extra_in_new, ()

    shared = [c for c in reference.columns if c in new.columns]
    chosen = [c for c in shared if _is_numeric(reference[c]) and _is_numeric(new[c])]
    skipped = tuple(str(c) for c in shared if c not in chosen)
    if not chosen:
        raise ValueError(
            "The two batches have no numeric column in common, so nothing can be compared. "
            f"Reference columns: {[str(c) for c in reference.columns]}; "
            f"new columns: {[str(c) for c in new.columns]}."
        )
    return chosen, missing_in_new, extra_in_new, skipped


def _compare_feature(name, reference_series: pd.Series, new_series: pd.Series) -> FeatureDrift:
    ref = _usable_values(reference_series)
    cur = _usable_values(new_series)
    ref_missing = len(reference_series) - ref.size
    new_missing = len(new_series) - cur.size
    ref_mean = float(ref.mean()) if ref.size else None
    new_mean = float(cur.mean()) if cur.size else None
    common = dict(feature=str(name), reference_count=int(ref.size), new_count=int(cur.size),
                  reference_missing=int(ref_missing), new_missing=int(new_missing),
                  reference_mean=ref_mean, new_mean=new_mean)

    if ref.size < MIN_SAMPLES or cur.size < MIN_SAMPLES:
        short = [f"{label} has {count}" for label, count in (("reference", ref.size), ("new batch", cur.size))
                 if count < MIN_SAMPLES]
        return FeatureDrift(status="not_enough_data", ks_statistic=None, threshold=None,
                            mean_shift_in_std=None,
                            note=f"Needs at least {MIN_SAMPLES} usable values in each batch; "
                                 f"{' and '.join(short)}.", **common)

    gap = ks_statistic(ref, cur)
    threshold = ks_threshold(ref.size, cur.size)
    ref_constant = bool(np.all(ref == ref[0]))
    new_constant = bool(np.all(cur == cur[0]))
    ref_std = float(ref.std(ddof=1))
    shift = (new_mean - ref_mean) / ref_std if ref_std > 0 else None

    notes = []
    if ref_constant and new_constant and ref[0] == cur[0]:
        notes.append("Constant in both batches.")
    elif ref_constant:
        notes.append("Constant in the reference batch, so the shift in standard deviations is undefined.")
    elif new_constant:
        notes.append("Constant in the new batch.")
    return FeatureDrift(status="drift" if gap > threshold else "no_drift", ks_statistic=gap,
                        threshold=threshold, mean_shift_in_std=shift, note=" ".join(notes), **common)


def detect_numeric_drift(reference: pd.DataFrame, new: pd.DataFrame, *, features=None) -> DriftReport:
    """Compare a new batch with a reference batch, feature by feature.

    ``reference`` and ``new`` are DataFrames with the same numeric feature columns
    (for an uploaded dataset, pass ``DatasetSpec.features``). ``features`` optionally
    limits the comparison to named columns, which must then exist in both batches.
    The inputs are not modified.
    """
    _check_frame(reference, "reference")
    _check_frame(new, "new")
    chosen, missing_in_new, extra_in_new, skipped = _choose_features(reference, new, features)

    results = tuple(_compare_feature(c, reference[c], new[c]) for c in chosen)
    checked = [r for r in results if r.status != "not_enough_data"]
    flagged = [r for r in checked if r.status == "drift"]
    share = len(flagged) / len(checked) if checked else None

    if not checked:
        status = "not_enough_data"
        message = (f"Not enough data: no feature had at least {MIN_SAMPLES} usable values in both "
                   "batches, so no drift check could be made.")
    elif share >= WARNING_SHARE and len(flagged) >= min(MIN_FLAGGED_FOR_WARNING, len(checked)):
        status = "warning"
        message = (f"Drift warning: {len(flagged)} of {len(checked)} checked features ({share:.0%}) "
                   "look different from the reference batch. This compares inputs only; it does "
                   "not show that model accuracy changed.")
    elif flagged:
        status = "watch"
        message = (f"Watch: {len(flagged)} of {len(checked)} checked features differ from the reference "
                   f"batch. A warning needs at least {WARNING_SHARE:.0%} of features and at least "
                   f"{MIN_FLAGGED_FOR_WARNING} features.")
    else:
        status = "no_signal"
        message = (f"No drift signal: none of the {len(checked)} checked features differ clearly from the "
                   "reference batch. This does not prove the model is still accurate.")

    extras = []
    unchecked = len(results) - len(checked)
    if unchecked and checked:
        extras.append(f"{unchecked} feature(s) had too few usable values to check.")
    if missing_in_new:
        extras.append(f"{len(missing_in_new)} reference column(s) are missing from the new batch "
                      f"({', '.join(missing_in_new)}) and could not be compared.")
        if status == "no_signal":
            status = "watch"
            message = (f"Watch: none of the {len(checked)} compared features differ clearly from the "
                       "reference batch, but the new batch is missing reference columns. This does "
                       "not prove the model is still accurate.")
    if extra_in_new:
        extras.append(f"{len(extra_in_new)} extra column(s) in the new batch were ignored.")
    if skipped:
        extras.append(f"{len(skipped)} shared column(s) were skipped because they are not numeric in both batches.")
    if extras:
        message = message + " " + " ".join(extras)

    return DriftReport(
        status=status, message=message, reference_rows=int(len(reference)), new_rows=int(len(new)),
        features_compared=len(results), features_checked=len(checked), features_flagged=len(flagged),
        share_flagged=share, features=results, missing_in_new=missing_in_new,
        extra_in_new=extra_in_new, skipped_non_numeric=skipped,
    )
