"""SQLite experiment history. No personal or uploaded data is stored."""

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from src.training import Result


def database_path() -> Path:
    return Path(os.environ.get("MODELSPHERE_DB_PATH", ".local/modelsphere.db"))


def _connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute("""CREATE TABLE IF NOT EXISTS experiments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at TEXT NOT NULL,
        model TEXT NOT NULL,
        parameters TEXT NOT NULL,
        metrics TEXT NOT NULL,
        train_count INTEGER NOT NULL,
        test_count INTEGER NOT NULL
    )""")
    return connection


def save_experiment(result: Result, path: Path | None = None) -> int:
    with _connect(path or database_path()) as connection:
        cursor = connection.execute(
            "INSERT INTO experiments (created_at, model, parameters, metrics, train_count, test_count) VALUES (?, ?, ?, ?, ?, ?)",
            (datetime.now(timezone.utc).isoformat(timespec="seconds"), result.model,
             json.dumps(result.parameters), json.dumps(result.metrics), result.train_count, result.test_count),
        )
        return int(cursor.lastrowid)


def list_experiments(path: Path | None = None) -> list[dict]:
    with _connect(path or database_path()) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute("SELECT * FROM experiments ORDER BY id DESC").fetchall()
    return [{**dict(row), "parameters": json.loads(row["parameters"]),
             "metrics": json.loads(row["metrics"])} for row in rows]
