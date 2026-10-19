# Upload a CSV for a classification experiment

The built-in Breast Cancer Wisconsin dataset remains the one-click demo.
Choose **Upload CSV** to try a small dataset of your own. Use a UTF-8 CSV with a
header row, at most 5 MB and 10,000 rows. This version supports a target with
exactly two classes and 1–50 selected numeric feature columns.

The app shows the first 10 rows, column types, missing-value counts, and target
class counts before training. Choose the target column, positive class, and
features. Both classes need at least 10 labelled rows, and at least 40 labelled
rows are needed overall. Text feature columns are visible but are not used in
this version; a text target is supported.

Rows with a blank target are excluded, and the count is shown. Missing numeric
feature values are filled with training-set medians inside each model pipeline.
Exact duplicate rows are counted and kept; decide whether they are legitimate
observations before interpreting results. An entirely empty selected feature or
an infinity value causes a clear error. Review selected features for IDs or
columns that reveal the answer after the fact; the app cannot detect every
source of data leakage.

Only the chosen dataset description, content fingerprint, settings, and metrics
are saved in SQLite. The uploaded rows are kept in memory for the active app
session and are not saved by ModelSphere. To repeat an old uploaded experiment
after a restart, upload the same CSV and select the same target, class, and
features again. Runs from different dataset selections are not mixed in the
comparison table. Local SQLite history on Streamlit Community Cloud may reset
when the app restarts.

## Shared code contract

`src/datasets.py` reads and validates an uploaded CSV and returns a
`DatasetSpec` with numeric features, an encoded binary target, and descriptive
metadata. `src/training.py` accepts it through the optional `dataset` argument
to `train_model`; omitting that argument keeps the built-in demo behavior.
Training returns the existing `Result` shape. For uploaded data,
`result.parameters["dataset"]` contains the dataset fingerprint, display name,
target, positive class, and chosen features. `src/tracking.py` saves that
JSON-serializable result without storing uploaded rows or changing the SQLite
schema. New models and tuning should keep this interface and fit preprocessing
only inside the training folds.
