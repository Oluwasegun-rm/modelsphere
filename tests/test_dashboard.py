"""Check the dashboard's train, save, and comparison flow."""

from streamlit.testing.v1 import AppTest


def test_dashboard_records_comparable_runs(tmp_path, monkeypatch):
    monkeypatch.setenv("MODELSPHERE_DB_PATH", str(tmp_path / "experiments.db"))

    app = AppTest.from_file("app.py").run()
    assert not app.exception

    app.button[0].click().run()
    assert not app.exception
    assert app.dataframe[0].value.iloc[0]["test fraction"] == 0.2
    assert app.dataframe[0].value.iloc[0]["seed"] == 42

    app.button[0].click().run()
    assert not app.exception
    assert any("Automatic comparison" in message.value for message in app.info)
