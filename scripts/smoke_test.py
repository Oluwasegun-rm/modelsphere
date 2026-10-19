"""Run a quick non-UI training and persistence check."""

from tempfile import TemporaryDirectory
from pathlib import Path

from src.training import train_model
from src.tracking import list_experiments, save_experiment

with TemporaryDirectory() as directory:
    path = Path(directory) / "demo.db"
    result = train_model("Logistic regression")
    run_id = save_experiment(result, path)
    assert list_experiments(path)[0]["id"] == run_id
    print(f"Smoke test passed: accuracy={result.metrics['accuracy']:.3f}, saved run={run_id}")
