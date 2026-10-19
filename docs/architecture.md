# Architecture

The Streamlit interface calls `train_model`, which loads scikit-learn's bundled dataset, makes a stratified holdout split, fits one estimator, and returns test metrics. `save_experiment` writes model settings and metrics to SQLite. `list_experiments` renders the comparison table. No model binary or row-level patient data is stored.

## Extension points

- Training engine: add cross-validation or optional Optuna tuning without using the holdout set to select parameters.
- Tracking: add labels, export, and retention policy if needed.
- Monitoring: compare new feature batches with a reference distribution; call it a simple drift signal rather than production monitoring.
