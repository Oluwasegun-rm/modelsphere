import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest

from src import tracking
from src.training import Result, train_model
from src.tracking import list_experiments, save_experiment


def make_result(model="Logistic regression", parameters=None, accuracy=0.9) -> Result:
    """A small hand-made run, so these tests do not need to train a model."""
    return Result(
        model=model,
        parameters={"C": 1.0} if parameters is None else parameters,
        metrics={"accuracy": accuracy, "precision": 0.8, "recall": 0.7, "f1": 0.75, "roc_auc": 0.95},
        train_count=455,
        test_count=114,
    )


def test_save_and_compare(tmp_path):
    path = tmp_path / "runs.db"
    first = save_experiment(train_model("Logistic regression"), path)
    second = save_experiment(train_model("Random forest", n_estimators=10), path)
    rows = list_experiments(path)
    assert [row["id"] for row in rows] == [second, first]
    assert rows[0]["parameters"]["n_estimators"] == 10


def test_empty_history_lists_nothing(tmp_path):
    assert list_experiments(tmp_path / "new.db") == []


def test_many_runs_all_saved_newest_first(tmp_path):
    path = tmp_path / "runs.db"
    ids = [save_experiment(make_result(accuracy=i / 100), path) for i in range(25)]
    assert len(set(ids)) == 25  # every run gets its own number
    rows = list_experiments(path)
    assert [row["id"] for row in rows] == sorted(ids, reverse=True)
    assert rows[0]["metrics"]["accuracy"] == pytest.approx(0.24)
    assert rows[-1]["metrics"]["accuracy"] == pytest.approx(0.0)


def test_saved_values_come_back_unchanged(tmp_path):
    path = tmp_path / "runs.db"
    run = make_result("Random forest", {"n_estimators": 50, "max_depth": None})
    save_experiment(run, path)
    row = list_experiments(path)[0]
    assert row["model"] == "Random forest"
    assert row["parameters"] == {"n_estimators": 50, "max_depth": None}
    assert row["metrics"] == run.metrics
    assert (row["train_count"], row["test_count"]) == (455, 114)
    assert row["created_at"].endswith("+00:00")  # saved in UTC


def test_history_is_still_there_on_the_next_call(tmp_path):
    path = tmp_path / "runs.db"
    save_experiment(make_result(), path)
    assert len(list_experiments(path)) == 1  # a fresh connection sees the saved run
    save_experiment(make_result(), path)
    assert len(list_experiments(path)) == 2


def test_default_location_comes_from_environment(tmp_path, monkeypatch):
    """The dashboard calls save/list with no path, so this is how it finds the database."""
    path = tmp_path / "nested" / "folder" / "app.db"
    monkeypatch.setenv("MODELSPHERE_DB_PATH", str(path))
    run_id = save_experiment(make_result())
    assert path.exists()  # missing folders are created
    assert [row["id"] for row in list_experiments()] == [run_id]


def test_odd_text_is_stored_as_text_not_run_as_sql(tmp_path):
    path = tmp_path / "runs.db"
    nasty = "x'); DROP TABLE experiments; --"
    save_experiment(make_result(model=nasty), path)
    save_experiment(make_result(), path)
    rows = list_experiments(path)
    assert len(rows) == 2
    assert rows[1]["model"] == nasty


@pytest.mark.parametrize("bad_metrics", [float("nan"), float("inf")])
def test_unreadable_numbers_are_refused_and_nothing_is_saved(tmp_path, bad_metrics):
    path = tmp_path / "runs.db"
    save_experiment(make_result(), path)
    bad = make_result(accuracy=bad_metrics)
    with pytest.raises(ValueError):
        save_experiment(bad, path)
    assert len(list_experiments(path)) == 1


def test_settings_that_are_not_plain_values_are_refused(tmp_path):
    path = tmp_path / "runs.db"
    with pytest.raises(ValueError):
        save_experiment(make_result(parameters={"model_object": object()}), path)
    assert not path.exists()  # a refused run does not even create the database
    assert list_experiments(path) == []


def test_connections_are_closed_after_each_call(tmp_path, monkeypatch):
    opened = []
    real_connect = sqlite3.connect

    def spy(*args, **kwargs):
        connection = real_connect(*args, **kwargs)
        opened.append(connection)
        return connection

    monkeypatch.setattr(tracking.sqlite3, "connect", spy)
    path = tmp_path / "runs.db"
    save_experiment(make_result(), path)
    list_experiments(path)
    assert len(opened) == 2
    for connection in opened:
        with pytest.raises(sqlite3.ProgrammingError):  # only a closed connection raises this
            connection.execute("SELECT 1")


def test_saving_at_the_same_time_loses_nothing(tmp_path):
    path = tmp_path / "runs.db"
    save_experiment(make_result(), path)  # set up the table first
    with ThreadPoolExecutor(max_workers=8) as pool:
        ids = list(pool.map(lambda i: save_experiment(make_result(accuracy=i / 100), path), range(40)))
    assert len(set(ids)) == 40
    assert len(list_experiments(path)) == 41