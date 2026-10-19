import json
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy.stats import ks_2samp
from sklearn.datasets import load_breast_cancer

from src.datasets import prepare_dataset
from src.monitoring import (KS_CRITICAL_COEFFICIENT, MIN_EFFECT, MIN_SAMPLES, WARNING_SHARE,
                            detect_numeric_drift, ks_statistic, ks_threshold)


def make_frame(rng, rows, columns=("a", "b", "c", "d", "e"), shifts=None):
    """Normal data per column; `shifts` moves named columns by that many standard deviations."""
    shifts = shifts or {}
    return pd.DataFrame({name: rng.normal(shifts.get(name, 0.0), 1.0, rows) for name in columns})


def feature(report, name):
    return next(item for item in report.features if item.feature == name)


# ---------- the measure itself ----------

def test_ks_statistic_simple_cases():
    assert ks_statistic([1, 2, 3], [1, 2, 3]) == 0.0
    assert ks_statistic([1, 2, 3], [10, 11, 12]) == 1.0
    assert ks_statistic([1, 2, 3, 4], [3, 4, 5, 6]) == pytest.approx(0.5)


def test_ks_statistic_matches_scipy_even_with_many_ties():
    rng = np.random.default_rng(0)
    for _ in range(200):
        a = np.round(rng.normal(size=rng.integers(5, 80)), 1)  # rounding creates repeated values
        b = np.round(rng.normal(0.3, 1.0, size=rng.integers(5, 80)), 1)
        assert ks_statistic(a, b) == pytest.approx(ks_2samp(a, b).statistic, abs=1e-12)


def test_threshold_grows_for_small_batches_and_has_a_floor():
    assert ks_threshold(30, 30) == pytest.approx(KS_CRITICAL_COEFFICIENT * np.sqrt(60 / 900))
    assert ks_threshold(30, 30) > ks_threshold(300, 300) > ks_threshold(1000, 1000)
    assert ks_threshold(5000, 5000) == MIN_EFFECT  # huge batches never flag tiny gaps


# ---------- shifted versus similar batches ----------

def test_similar_batches_give_no_signal():
    rng = np.random.default_rng(1)
    reference = make_frame(rng, 400)
    # A random subsample of the very same data: any gap is pure sampling noise.
    report = detect_numeric_drift(reference, reference.sample(300, random_state=1))
    assert report.status == "no_signal"
    assert report.features_flagged == 0
    assert report.features_checked == 5
    assert "does not prove" in report.message


def test_independent_draws_from_one_distribution_do_not_warn():
    rng = np.random.default_rng(1)
    report = detect_numeric_drift(make_frame(rng, 400), make_frame(rng, 300))
    assert report.status != "warning"  # one chance flag at most; the rate test below counts them
    assert report.features_flagged <= 1


def test_clearly_shifted_features_are_named_and_trigger_a_warning():
    rng = np.random.default_rng(2)
    reference = make_frame(rng, 400)
    new = make_frame(rng, 300, shifts={"b": 2.0, "d": -2.5})
    report = detect_numeric_drift(reference, new)
    assert report.status == "warning"
    flagged = {f.feature for f in report.features if f.status == "drift"}
    assert {"b", "d"} <= flagged  # the shifted ones are always found
    assert flagged <= {"a", "b", "c", "d", "e"} and len(flagged) <= 3  # at most one chance flag besides them
    assert feature(report, "b").mean_shift_in_std == pytest.approx(2.0, abs=0.35)
    assert feature(report, "d").mean_shift_in_std == pytest.approx(-2.5, abs=0.35)
    assert feature(report, "a").status == "no_drift"
    assert report.share_flagged == pytest.approx(len(flagged) / 5)


def test_overall_levels_one_flagged_is_watch_and_twenty_percent_is_warning():
    rng = np.random.default_rng(3)
    columns = tuple("abcdefghij")
    reference = make_frame(rng, 300, columns)
    one = detect_numeric_drift(reference, make_frame(rng, 300, columns, shifts={"a": 3.0}))
    two = detect_numeric_drift(reference, make_frame(rng, 300, columns, shifts={"a": 3.0, "b": 3.0}))
    assert (one.features_flagged, one.status) == (1, "watch")
    assert two.features_flagged == 2 and two.share_flagged == pytest.approx(WARNING_SHARE)
    assert two.status == "warning"  # exactly 20% counts as a warning


def test_one_flagged_feature_among_a_few_is_only_a_watch():
    """Regression test: without a two-feature minimum, 1 of 5 (20%) falsely warned 19% of the time."""
    rng = np.random.default_rng(23)
    reference = make_frame(rng, 300)
    one = detect_numeric_drift(reference, make_frame(rng, 300, shifts={"a": 3.0}))
    assert one.features_flagged == 1 and one.share_flagged == pytest.approx(0.2)
    assert one.status == "watch"
    assert "at least 2 features" in one.message
    two = detect_numeric_drift(reference, make_frame(rng, 300, shifts={"a": 3.0, "b": 3.0}))
    assert two.status == "warning"


def test_a_single_checked_feature_that_drifts_can_warn():
    rng = np.random.default_rng(24)
    report = detect_numeric_drift(make_frame(rng, 200, ("x",)), make_frame(rng, 200, ("x",), {"x": 3.0}))
    assert (report.features_checked, report.features_flagged, report.status) == (1, 1, "warning")


def test_false_warning_rate_on_identical_data_stays_low_for_any_feature_count():
    rng = np.random.default_rng(25)
    for columns in (1, 2, 3, 5, 10):
        names = tuple(f"f{i}" for i in range(columns))
        warned = sum(
            detect_numeric_drift(make_frame(rng, 100, names), make_frame(rng, 100, names)).status == "warning"
            for _ in range(300))
        assert warned / 300 < 0.10, columns  # measured at most about 5%


def test_identical_distributions_rarely_flag_a_feature():
    rng = np.random.default_rng(4)
    flagged = checked = 0
    for _ in range(200):
        report = detect_numeric_drift(make_frame(rng, 100, tuple("abcdefghij")),
                                      make_frame(rng, 100, tuple("abcdefghij")))
        flagged += report.features_flagged
        checked += report.features_checked
    assert flagged / checked < 0.08  # a 5% rule; measured about 4%


def test_batch_size_matters_small_batches_need_a_bigger_gap():
    rng = np.random.default_rng(5)
    hits = {}
    for rows in (30, 2000):
        hits[rows] = sum(
            detect_numeric_drift(make_frame(rng, rows, ("x",)), make_frame(rng, rows, ("x",), {"x": 0.5})).features_flagged
            for _ in range(100))
    assert hits[30] < 60      # a half-std shift is usually not provable with 30 rows
    assert hits[2000] == 100  # but is always caught with 2,000
    tiny = detect_numeric_drift(make_frame(rng, 20000, ("x",)), make_frame(rng, 20000, ("x",), {"x": 0.05}))
    assert tiny.features_flagged == 0  # too small to matter, even though 20,000 rows could detect it


# ---------- missing values ----------

def test_missing_values_are_left_out_and_counted():
    rng = np.random.default_rng(6)
    reference = make_frame(rng, 300, ("a", "b"))
    new = make_frame(rng, 300, ("a", "b"), shifts={"a": 2.0})
    reference.loc[:59, "a"] = np.nan
    new.loc[:29, "a"] = np.nan
    report = detect_numeric_drift(reference, new)
    a = feature(report, "a")
    assert (a.reference_missing, a.new_missing) == (60, 30)
    assert (a.reference_count, a.new_count) == (240, 270)
    expected = ks_2samp(reference["a"].dropna(), new["a"].dropna()).statistic
    assert a.ks_statistic == pytest.approx(expected)
    assert a.status == "drift"
    assert feature(report, "b").reference_missing == 0


def test_infinite_values_and_nullable_columns_count_as_missing():
    rng = np.random.default_rng(7)
    reference = make_frame(rng, 200, ("a", "b"))
    new = make_frame(rng, 200, ("a", "b"))
    reference.loc[:9, "a"] = np.inf
    new["b"] = new["b"].astype("Float64")
    new.loc[:19, "b"] = pd.NA
    report = detect_numeric_drift(reference, new)
    assert feature(report, "a").reference_missing == 10
    assert feature(report, "b").new_missing == 20
    assert report.features_checked == 2


def test_a_feature_missing_everywhere_in_the_new_batch_does_not_stop_the_others():
    rng = np.random.default_rng(8)
    reference = make_frame(rng, 200, ("a", "b"))
    new = make_frame(rng, 200, ("a", "b"))
    new["b"] = np.nan
    report = detect_numeric_drift(reference, new)
    assert feature(report, "b").status == "not_enough_data"
    assert feature(report, "b").ks_statistic is None
    assert feature(report, "a").status == "no_drift"
    assert (report.features_compared, report.features_checked) == (2, 1)
    assert "too few usable values" in report.message


# ---------- small batches ----------

def test_minimum_sample_size_boundary():
    rng = np.random.default_rng(9)
    reference = make_frame(rng, 200, ("a",))
    just_enough = detect_numeric_drift(reference, make_frame(rng, MIN_SAMPLES, ("a",)))
    too_few = detect_numeric_drift(reference, make_frame(rng, MIN_SAMPLES - 1, ("a",)))
    assert just_enough.features_checked == 1
    assert too_few.features_checked == 0
    assert too_few.status == "not_enough_data"
    assert too_few.share_flagged is None
    assert str(MIN_SAMPLES) in too_few.message


def test_tiny_batch_never_raises_a_false_alarm():
    rng = np.random.default_rng(10)
    report = detect_numeric_drift(make_frame(rng, 200), make_frame(rng, 5, shifts={"a": 5.0}))
    assert report.status == "not_enough_data"
    assert report.features_flagged == 0


# ---------- constant columns ----------

def test_constant_in_both_batches_is_fine_and_raises_no_warnings():
    reference = pd.DataFrame({"a": [4.0] * 50})
    new = pd.DataFrame({"a": [4.0] * 40})
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no divide-by-zero or similar
        report = detect_numeric_drift(reference, new)
    a = feature(report, "a")
    assert (a.status, a.ks_statistic, a.mean_shift_in_std) == ("no_drift", 0.0, None)
    assert "Constant in both" in a.note
    assert report.status == "no_signal"


def test_constant_reference_with_a_different_value_or_new_spread_is_flagged():
    rng = np.random.default_rng(11)
    reference = pd.DataFrame({"a": [4.0] * 100, "b": [4.0] * 100})
    new = pd.DataFrame({"a": [9.0] * 100, "b": rng.normal(4.0, 1.0, 100)})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        report = detect_numeric_drift(reference, new)
    assert (feature(report, "a").status, feature(report, "a").ks_statistic) == ("drift", 1.0)
    assert feature(report, "b").status == "drift"  # reference never varied, new batch does
    assert feature(report, "a").mean_shift_in_std is None
    assert "Constant in the reference" in feature(report, "a").note


def test_constant_new_batch_against_a_varying_reference_is_flagged():
    rng = np.random.default_rng(12)
    reference = pd.DataFrame({"a": rng.normal(0, 1, 200)})
    new = pd.DataFrame({"a": [0.0] * 60})
    report = detect_numeric_drift(reference, new)
    assert feature(report, "a").status == "drift"
    assert "Constant in the new batch" in feature(report, "a").note
    assert feature(report, "a").mean_shift_in_std is not None


# ---------- mismatched columns ----------

def test_column_order_does_not_matter():
    rng = np.random.default_rng(13)
    reference = make_frame(rng, 200)
    new = make_frame(rng, 200, shifts={"c": 2.0})
    straight = detect_numeric_drift(reference, new)
    flipped = detect_numeric_drift(reference, new[list(reversed(new.columns))])
    assert [f.ks_statistic for f in straight.features] == [f.ks_statistic for f in flipped.features]


def test_columns_missing_from_the_new_batch_are_reported_and_prevent_an_all_clear():
    rng = np.random.default_rng(14)
    reference = make_frame(rng, 200, ("a", "b", "c"))
    new = make_frame(rng, 200, ("a", "b"))
    report = detect_numeric_drift(reference, new)
    assert report.missing_in_new == ("c",)
    assert report.features_compared == 2
    assert report.status == "watch"  # nothing drifted, but a column vanished
    assert "missing from the new batch" in report.message and "(c)" in report.message


def test_extra_and_non_numeric_columns_are_listed_and_skipped():
    rng = np.random.default_rng(15)
    reference = make_frame(rng, 200, ("a", "b"))
    reference["label"] = "x"
    reference["flag"] = True
    new = make_frame(rng, 200, ("a", "b", "extra"))
    new["label"] = "y"
    new["flag"] = False
    report = detect_numeric_drift(reference, new)
    assert report.extra_in_new == ("extra",)
    assert set(report.skipped_non_numeric) == {"label", "flag"}
    assert [f.feature for f in report.features] == ["a", "b"]
    assert report.status == "no_signal"


def test_numeric_column_that_arrives_as_text_is_skipped_not_guessed():
    rng = np.random.default_rng(16)
    reference = make_frame(rng, 100, ("a", "b"))
    new = make_frame(rng, 100, ("a", "b"))
    new["b"] = new["b"].astype(str)
    report = detect_numeric_drift(reference, new)
    assert report.skipped_non_numeric == ("b",)
    assert [f.feature for f in report.features] == ["a"]  # "b" is not converted or compared


def _similar_batches_with_text_in_new(rng, reverse=False):
    """Same data twice (so no real drift), except column 'b' has numbers in one batch and text in the other."""
    reference = make_frame(rng, 300, ("a", "b", "c"))
    new = reference.sample(200, random_state=0).reset_index(drop=True)
    new["b"] = new["b"].round(2).astype(str)
    new.loc[3, "b"] = "N/A"  # the kind of stray entry that turns a whole column into text
    return (new, reference) if reverse else (reference, new)


def test_numeric_in_reference_but_text_in_new_is_at_least_a_watch_and_names_the_column():
    reference, new = _similar_batches_with_text_in_new(np.random.default_rng(40))
    report = detect_numeric_drift(reference, new)
    assert report.features_flagged == 0  # nothing that WAS compared drifted...
    assert report.status == "watch"      # ...but that must not look like an all clear
    assert report.type_mismatch == ("b",)
    assert "b" in report.skipped_non_numeric
    assert [f.feature for f in report.features] == ["a", "c"]
    assert "Column 'b' is numeric in one batch but not in the other" in report.message
    assert "could not be compared" in report.message
    assert not report.message.startswith("No drift signal")


def test_text_in_reference_but_numeric_in_new_is_treated_the_same_way():
    reference, new = _similar_batches_with_text_in_new(np.random.default_rng(41), reverse=True)
    report = detect_numeric_drift(reference, new)
    assert (report.status, report.type_mismatch) == ("watch", ("b",))
    assert "'b'" in report.message


def test_several_mismatched_columns_are_all_named():
    rng = np.random.default_rng(42)
    reference = make_frame(rng, 200, ("a", "b", "c"))
    new = reference.sample(150, random_state=0).reset_index(drop=True)
    new["b"], new["c"] = new["b"].astype(str), new["c"].astype(str)
    report = detect_numeric_drift(reference, new)
    assert report.type_mismatch == ("b", "c")
    assert report.status == "watch"
    assert "Columns 'b', 'c' are numeric in one batch" in report.message


def test_a_text_label_that_is_text_in_both_batches_is_skipped_without_changing_the_status():
    rng = np.random.default_rng(43)
    reference = make_frame(rng, 300, ("a", "b"))
    reference["passed"] = rng.choice(["yes", "no"], 300)
    new = reference.sample(200, random_state=0).reset_index(drop=True)
    new["passed"] = rng.choice(["yes", "no"], 200)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # also no Python warning of any kind
        report = detect_numeric_drift(reference, new)
    assert report.status == "no_signal"
    assert report.type_mismatch == ()
    assert report.skipped_non_numeric == ("passed",)
    assert [f.feature for f in report.features] == ["a", "b"]
    assert "could not be compared" not in report.message
    assert "numeric in either batch" in report.message  # mentioned, but only as information


def test_a_real_warning_stays_a_warning_and_still_names_the_mismatched_column():
    rng = np.random.default_rng(44)
    reference = make_frame(rng, 300, ("a", "b", "c", "d"))
    new = make_frame(rng, 300, ("a", "b", "c", "d"), shifts={"a": 3.0, "b": 3.0})
    new["d"] = new["d"].astype(str)
    report = detect_numeric_drift(reference, new)
    assert report.status == "warning"  # not downgraded by the mismatch
    assert report.type_mismatch == ("d",)
    assert "Column 'd' is numeric in one batch" in report.message


def test_mismatched_type_and_missing_column_are_both_reported():
    rng = np.random.default_rng(45)
    reference = make_frame(rng, 300, ("a", "b", "c"))
    new = reference[["a", "b"]].sample(200, random_state=0).reset_index(drop=True)
    new["b"] = new["b"].astype(str)
    report = detect_numeric_drift(reference, new)
    assert report.status == "watch"
    assert report.type_mismatch == ("b",) and report.missing_in_new == ("c",)
    assert "'b'" in report.message and "(c)" in report.message


def test_when_the_only_shared_numeric_column_changes_type_the_error_names_it():
    rng = np.random.default_rng(46)
    reference = make_frame(rng, 50, ("a",))
    new = reference.astype(str)
    with pytest.raises(ValueError, match="numeric in only one batch: a"):
        detect_numeric_drift(reference, new)


def test_no_shared_numeric_columns_is_an_error_that_names_the_columns():
    rng = np.random.default_rng(17)
    with pytest.raises(ValueError, match="no numeric column in common"):
        detect_numeric_drift(make_frame(rng, 50, ("a", "b")), make_frame(rng, 50, ("x", "y")))


def test_explicit_feature_list():
    rng = np.random.default_rng(18)
    reference, new = make_frame(rng, 200), make_frame(rng, 200, shifts={"a": 3.0, "b": 3.0})
    report = detect_numeric_drift(reference, new, features=["b", "c"])
    assert [f.feature for f in report.features] == ["b", "c"]
    with pytest.raises(ValueError, match="not in both batches: nope"):
        detect_numeric_drift(reference, new, features=["a", "nope"])
    with pytest.raises(ValueError):
        detect_numeric_drift(reference, new, features=[])
    with pytest.raises(ValueError):
        detect_numeric_drift(reference, new, features=["a", "a"])


# ---------- bad inputs and output ----------

def test_bad_inputs_get_clear_errors():
    rng = np.random.default_rng(19)
    good = make_frame(rng, 50)
    with pytest.raises(TypeError, match="DataFrame"):
        detect_numeric_drift(good.to_numpy(), good)
    with pytest.raises(ValueError, match="empty"):
        detect_numeric_drift(good, good.iloc[0:0])
    with pytest.raises(ValueError, match="duplicate column"):
        detect_numeric_drift(good, pd.concat([good, good[["a"]]], axis=1))


def test_inputs_are_not_modified():
    rng = np.random.default_rng(20)
    reference, new = make_frame(rng, 100), make_frame(rng, 100)
    reference.loc[:4, "a"] = np.nan
    before = (reference.copy(), new.copy())
    detect_numeric_drift(reference, new)
    pd.testing.assert_frame_equal(reference, before[0])
    pd.testing.assert_frame_equal(new, before[1])


def test_report_is_json_safe_and_states_its_limits():
    rng = np.random.default_rng(21)
    reference = make_frame(rng, 100)
    reference["b"] = np.nan  # missing data must not leak NaN into the output
    report = detect_numeric_drift(reference, make_frame(rng, 100, shifts={"a": 2.5}))
    payload = json.loads(json.dumps(report.to_dict(), allow_nan=False))
    assert payload["status"] == report.status
    assert len(payload["features"]) == 5
    assert payload["type_mismatch"] == []  # the new field is part of the JSON output
    assert "not a prediction of future drift" in payload["limits"]
    assert "not a measurement of model accuracy" in payload["limits"]
    assert "does not show that model accuracy changed" in report.message


# ---------- real data ----------

def _breast_cancer():
    data = load_breast_cancer(as_frame=True)
    return data.data, data.target


def test_real_data_known_shift_is_a_warning():
    features, target = _breast_cancer()
    report = detect_numeric_drift(features[target == 1], features[target == 0])  # benign vs malignant
    assert report.status == "warning"
    assert report.features_flagged >= 20 and report.features_checked == 30


def test_real_data_random_halves_rarely_warn():
    features, _ = _breast_cancer()
    rng = np.random.default_rng(22)
    warned = 0
    for _ in range(100):
        order = rng.permutation(len(features))
        half = len(features) // 2
        warned += detect_numeric_drift(features.iloc[order[:half]], features.iloc[order[half:]]).status == "warning"
    assert warned < 15  # measured about 6-8%: chance flags cluster because the features are correlated


def test_works_directly_with_dataset_spec_features():
    features, target = _breast_cancer()
    frame = features.copy()
    frame["diagnosis"] = np.where(target == 1, "benign", "malignant")
    spec = prepare_dataset(frame, name="breast.csv", target_name="diagnosis",
                           feature_names=list(features.columns), positive_label="benign")
    report = detect_numeric_drift(spec.features.iloc[:300], spec.features.iloc[300:])
    assert report.features_compared == 30
    assert report.status in {"no_signal", "watch", "warning"}
