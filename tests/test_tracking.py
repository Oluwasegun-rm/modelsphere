from src.training import train_model
from src.tracking import list_experiments, save_experiment


def test_save_and_compare(tmp_path):
    path = tmp_path / "runs.db"
    first = save_experiment(train_model("Logistic regression"), path)
    second = save_experiment(train_model("Random forest", n_estimators=10), path)
    rows = list_experiments(path)
    assert [row["id"] for row in rows] == [second, first]
    assert rows[0]["parameters"]["n_estimators"] == 10
