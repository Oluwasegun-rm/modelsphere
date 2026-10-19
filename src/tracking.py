"""SQLite experiment history. No personal or uploaded data is stored."""

import json
import os
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

from src.training import Result

# If another save has the database locked, wait this long instead of failing at once.
BUSY_TIMEOUT_SECONDS = 30


def database_path() -> Path:
    return Path(os.environ.get("MODELSPHERE_DB_PATH", ".local/modelsphere.db"))


def _connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=BUSY_TIMEOUT_SECONDS)
    try:
        connection.execute("""CREATE TABLE IF NOT EXISTS experiments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            model TEXT NOT NULL,
            parameters TEXT NOT NULL,
            metrics TEXT NOT NULL,
            train_count INTEGER NOT NULL,
            test_count INTEGER NOT NULL
        )""")
    except Exception:
        connection.close()
        raise
    return connection


def _to_json(value, label: str) -> str:
    """Turn settings/metrics into JSON text, refusing anything that could not be read back."""
    try:
        return json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Experiment {label} must be plain numbers, text, true/false or empty: {exc}") from exc


def save_experiment(result: Result, path: Path | None = None) -> int:
    # Check the values first so a bad run never creates a half-saved row.
    parameters = _to_json(result.parameters, "parameters")
    metrics = _to_json(result.metrics, "metrics")
    with closing(_connect(path or database_path())) as connection:
        with connection:  # commits the save, or undoes it if something goes wrong
            cursor = connection.execute(
                "INSERT INTO experiments (created_at, model, parameters, metrics, train_count, test_count) VALUES (?, ?, ?, ?, ?, ?)",
                (datetime.now(timezone.utc).isoformat(timespec="seconds"), result.model,
                 parameters, metrics, result.train_count, result.test_count),
            )
        return int(cursor.lastrowid)


def list_experiments(path: Path | None = None) -> list[dict]:
    with closing(_connect(path or database_path())) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute("SELECT * FROM experiments ORDER BY id DESC").fetchall()
    return [{**dict(row), "parameters": json.loads(row["parameters"]),
             "metrics": json.loads(row["metrics"])} for row in rows]