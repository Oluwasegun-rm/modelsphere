# ModelSphere architecture and code walkthrough

ModelSphere answers a classroom question: **how do model settings affect performance on unseen data?** It trains a classifier, evaluates it on a held-out test set, and saves the settings and scores for comparison. This is an educational demo, not a medical decision tool.

## Data flow

```text
Streamlit controls (app.py)
    | built-in data OR inspected CSV (src/datasets.py)
    | model, settings, test fraction, seed
    v
train_model (src/training.py) OR tune_model (src/tuning.py)
    | load bundled Wisconsin dataset or accept DatasetSpec
    | stratified train/test split
    | fit preprocessing and model on training rows only
    | tune on training folds only when requested
    | predict on untouched test rows
    v
Result: settings, test metrics, row counts, optional dataset metadata
    |                          |
    | display scores           | save_experiment
    v                          v
Streamlit metric cards      SQLite file (src/tracking.py)
                               |
                               | list_experiments
                               v
                          comparison table (app.py)
```

The UI coordinates the workflow. `src/datasets.py` validates uploads and returns a `DatasetSpec` with numeric features, binary labels, and metadata. Training and tuning do not know about Streamlit or SQLite; tracking accepts a small `Result` rather than a model or dataset. Uploaded rows are not written to SQLite. Separately, `src/monitoring.py` compares the training features with a new batch and returns a drift report without saving rows.

## File-by-file roles

| File or folder | Role |
| --- | --- |
| `app.py` | Collects settings, starts a run, and displays metrics and saved runs. |
| `src/datasets.py` | Reads and validates a small CSV; provides a `DatasetSpec` and fingerprint. |
| `src/training.py` | Loads data, splits it, chooses an algorithm, fits it, and calculates test metrics. Defines `Result`. |
| `src/tuning.py` | Uses bounded Optuna cross-validation on training rows and returns the usual `Result`. |
| `src/monitoring.py` | Compares numeric feature distributions in reference and new batches. |
| `src/tracking.py` | Creates a SQLite table if needed, inserts run summaries, and lists them newest first. |
| `src/__init__.py` | Makes `src` importable as a Python package. |
| `tests/test_training.py` | Checks dataset size, both models, metric ranges, and invalid model handling. |
| `tests/test_tracking.py` | Saves two runs in a temporary database and checks order and settings. |
| `scripts/smoke_test.py` | Checks the complete train-save-read path without the UI. |
| `requirements.txt` | Lists packages needed locally and on deployment. |
| `.github/workflows/test.yml` | Runs pytest on pushes and pull requests. |
| `.github/workflows/container.yml` | Builds the Docker image, exercises training and saving, and checks app health. |
| `Dockerfile` | Packages one instance of the Streamlit app. |
| `docs/` | Holds setup, architecture, presentation, team tasks, deployment notes, and demo GIF. |
| `.local/modelsphere.db` | Runtime database, ignored by Git; each local installation has its own history. |

## Dataset and experimental design

The [Breast Cancer Wisconsin (Diagnostic) dataset](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic) is bundled with scikit-learn. It has **569 rows**, **30 numeric input features**, and a binary target: **malignant = 0**, **benign = 1**. The CSV path added by PR #10 allows a small UTF-8 file, a selected two-class target, and 1–50 numeric features. See [upload guidance](upload.md) for validation and limits.

By default, `train_test_split` puts 80% of rows in training and 20% in testing, using seed 42. `stratify=data.target` keeps class proportions similar in both subsets. The seed makes a repeat run with the same settings reproducible. Changing the seed changes which observations are held out. For a fair side-by-side comparison, keep the test fraction and seed fixed.

Only training rows are passed to `.fit()`. Uploaded missing numeric values are imputed with training-set medians inside the model pipeline. The test rows are used for evaluation. A holdout score estimates performance on unseen examples from that dataset, but one split cannot establish performance on new hospitals or patient populations.

## Algorithms and preprocessing

| Algorithm | Intuition | Controls | Reason for inclusion |
| --- | --- | --- | --- |
| Logistic regression | Learns feature weights and converts them to a class probability; it is a classifier despite its name. | `C`, the inverse regularization strength. Smaller `C` means stronger regularization. | A simple linear baseline. |
| Random forest | Trains many decision trees and combines their votes. | Number of trees and maximum tree depth. | A nonlinear contrast that can capture feature interactions. |
| Neural network (PyTorch) | Learns a small hidden representation, then a probability for class 1. | Epochs (1–20), hidden units (8/16/32), and learning rate. | Demonstrates the requested PyTorch/Lightning workflow on the same data. |

Logistic regression uses median imputation and `StandardScaler` inside a scikit-learn pipeline because feature scales vary. Random forest uses median imputation without scaling. The neural network uses the same training-only imputation and scaling, then a PyTorch model with one hidden layer, a binary cross-entropy loss, and a Lightning CPU training loop. Preprocessing is fit on **training data only**; the test data is transformed using training-set statistics. This prevents test data leakage. The neural network is a bounded course demonstration, not a large-scale AI model.

## Metrics and interpretation

The scikit-learn models use `predict()` for labels and `predict_proba()[:, 1]` for the score of encoded class **1**. The neural network applies a sigmoid to its output and labels scores of at least 0.5 as class 1. For the built-in dataset, class 1 is **benign**. For an upload, the user chooses which target value becomes class 1. Precision, recall, F1, and ROC AUC use that positive class. This library convention is not a statement about medical importance.

| Metric | Meaning here |
| --- | --- |
| Accuracy | Fraction of all test cases classified correctly. |
| Precision | Of cases predicted benign, fraction actually benign. |
| Recall | Of actually benign cases, fraction predicted benign. |
| F1 | Harmonic mean of benign precision and benign recall. |
| ROC AUC | How well the benign probability ranks benign above malignant across thresholds. |

**Built-in displayed recall is not malignant recall.** A health application would need to study missed malignant cases separately. Cross-validation is used only when optional tuning is selected. The app does not calculate confidence intervals or save a deployable model.

## Tracking and comparison

`save_experiment` stores a UTC timestamp, model name, parameter settings, five metrics, and train/test counts. Parameterized SQL placeholders (`?`) insert values. JSON fields preserve the small settings and metrics dictionaries. For uploaded data, `Result.parameters["dataset"]` records a fingerprint, target, positive class, and chosen features. `list_experiments` reads newest first and `app.py` shows runs matching the current dataset selection. It stores no row-level dataset records, predictions, or model binaries.

SQLite is a single file on the machine running the app. Cloud storage may disappear when the app restarts, so this is not a durable shared experiment service. Keep the same seed and test fraction when comparing model choices or hyperparameters.

## Why these technologies

- **Python:** One language for data loading, training, and UI.
- **Streamlit:** An interactive dashboard without a separate frontend or API server.
- **scikit-learn:** Established implementations for data loading, splitting, preprocessing, models, and metrics.
- **Optuna:** Tries a bounded set of hyperparameters using training-only cross-validation. It is not reinforcement learning.
- **PyTorch and Lightning:** Build and train one small CPU neural classifier; its results use the same held-out evaluation and SQLite tracking as the baselines.
- **SQLite:** A built-in single-file database sufficient for demo runs.
- **pandas:** Converts saved run dictionaries to the comparison table.
- **pytest and GitHub Actions:** Check basic behavior locally and after changes.
- **Docker:** Packages one app instance; CI builds and smoke-checks it. It does not autoscale.

## Current extensions and next contracts

- **Optuna tuning (WaunCPU):** `src/tuning.py` accepts the same built-in or `DatasetSpec` data and returns a JSON-safe `Result`. It uses cross-validation on training rows only; the held-out test set is scored once after choosing settings. The dashboard keeps this optional so the manual run remains fast.
- **Drift signal (Eicap1-prog):** `src/monitoring.py` compares numeric features from reference training rows and a new batch. Its output explains the measure and threshold. The dashboard can demonstrate a synthetic shift or accept another CSV. Without labels for the new batch, it cannot measure accuracy degradation or predict future drift.
- **PyTorch/Lightning classifier (Oluwasegun):** `src/neural.py` returns the same `Result` shape and metrics using the same split, positive-class mapping, and training-only preprocessing. The dashboard offers its bounded CPU training controls. PyTorch hooks could later log internal outputs, but hooks alone do not detect drift.
- **LLM explanation (Oluwasegun):** Optional commentary on saved settings and aggregate metrics only. Keep the secret in deployment settings; send no uploaded rows. Label generated suggestions as interpretations rather than measured results.

LLM commentary remains planned. Dockerfile generation, Swarm/Kubernetes scaling, transformer-based drift prediction, alerts, and a complete automated deployment pipeline are also outside the current release. README and presentation claims must match the live app.
