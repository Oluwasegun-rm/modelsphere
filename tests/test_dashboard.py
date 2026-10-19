"""Check the dashboard's train, save, and comparison flow."""

from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_dashboard_records_comparable_runs(tmp_path, monkeypatch):
    monkeypatch.setenv("MODELSPHERE_DB_PATH", str(tmp_path / "experiments.db"))

    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(app_path).run(timeout=15)
    assert not app.exception

    next(button for button in app.button if button.label == "Train and save").click().run(timeout=15)
    assert not app.exception
    assert app.dataframe[0].value.iloc[0]["test fraction"] == 0.2
    assert app.dataframe[0].value.iloc[0]["seed"] == 42

    next(button for button in app.button if button.label == "Train and save").click().run(timeout=15)
    assert not app.exception
    assert any("Automatic comparison" in message.value for message in app.info)


def test_dashboard_tunes_and_saves_a_run(tmp_path, monkeypatch):
    monkeypatch.setenv("MODELSPHERE_DB_PATH", str(tmp_path / "experiments.db"))

    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(app_path).run(timeout=15)
    app.checkbox[0].check().run(timeout=15)
    assert not app.exception

    next(button for button in app.button if button.label == "Tune, train and save").click().run(timeout=30)
    assert not app.exception
    assert any("Experiment #" in message.value for message in app.success)
    assert app.dataframe[0].value.iloc[0]["method"] == "Optuna"


def test_dashboard_shows_a_simulated_drift_warning(tmp_path, monkeypatch):
    monkeypatch.setenv("MODELSPHERE_DB_PATH", str(tmp_path / "experiments.db"))

    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(app_path).run(timeout=15)
    next(box for box in app.selectbox if box.label == "Example").set_value("Simulated shift").run(timeout=15)
    next(button for button in app.button if button.label == "Check drift").click().run(timeout=15)

    assert not app.exception
    assert any("Drift warning" in message.value for message in app.warning)
