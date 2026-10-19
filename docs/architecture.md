# ModelSphere architecture and code walkthrough

ModelSphere answers a classroom question: **how do model settings affect performance on unseen data?** It trains a classifier, evaluates it on a held-out test set, and saves the settings and scores for comparison. This is an educational demo, not a medical decision tool.

## Data flow

```text
Streamlit controls (app.py)
    | model, settings, test fraction, seed
    v
train_model (src/training.py)
    | load bundled Wisconsin dataset
    | stratified train/test split
    | fit preprocessing and model on training rows only
    | predict on untouched test rows
    v
Result: settings, test metrics, row counts
    |                          |
    | display scores           | save_experiment
    v                          v
Streamlit metric cards      SQLite file (src/tracking.py)
                               |
                               | list_experiments
                               v
                          comparison table (app.py)
```

The UI coordinates the workflow. The training module does not know about Streamlit or SQLite; the tracking module accepts a small `Result` rather than a model or dataset. This separation makes each part easier to explain and test.

## File-by-file roles

| File or folder | Role |
| --- | --- |
| `app.py` | Collects settings, starts a run, and displays metrics and saved runs. |
| `src/training.py` | Loads data, splits it, chooses an algorithm, fits it, and calculates test metrics. Defines `Result`. |
| `src/tracking.py` | Creates a SQLite table if needed, inserts run summaries, and lists them newest first. |
| `src/__init__.py` | Makes `src` importable as a Python package. |
| `tests/test_training.py` | Checks dataset size, both models, metric ranges, and invalid model handling. |
| `tests/test_tracking.py` | Saves two runs in a temporary database and checks order and settings. |
| `scripts/smoke_test.py` | Checks the complete train-save-read path without the UI. |
| `requirements.txt` | Lists packages needed locally and on deployment. |
| `.github/workflows/test.yml` | Runs pytest on pushes and pull requests. |
| `docs/` | Holds setup, architecture, presentation, team tasks, deployment notes, and demo GIF. |
| `.local/modelsphere.db` | Runtime database, ignored by Git; each local installation has its own history. |

## Dataset and experimental design

The [Breast Cancer Wisconsin (Diagnostic) dataset](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic) is bundled with scikit-learn. It has **569 rows**, **30 numeric input features**, and a binary target: **malignant = 0**, **benign = 1**. There is no CSV upload in the current app.

By default, `train_test_split` puts 80% of rows in training and 20% in testing, using seed 42. `stratify=data.target` keeps class proportions similar in both subsets. The seed makes a repeat run with the same settings reproducible. Changing the seed changes which observations are held out. For a fair side-by-side comparison, keep the test fraction and seed fixed.

Only training rows are passed to `.fit()`. The test rows are used for evaluation. A holdout score estimates performance on unseen examples from this dataset, but one split cannot establish performance on new hospitals or patient populations.

## Algorithms and preprocessing

| Algorithm | Intuition | Controls | Reason for inclusion |
| --- | --- | --- | --- |
| Logistic regression | Learns feature weights and converts them to a class probability; it is a classifier despite its name. | `C`, the inverse regularization strength. Smaller `C` means stronger regularization. | A simple linear baseline. |
| Random forest | Trains many decision trees and combines their votes. | Number of trees and maximum tree depth. | A nonlinear contrast that can capture feature interactions. |

Logistic regression uses a `StandardScaler` inside a scikit-learn pipeline because feature scales vary. The scaler is fit on **training data only**; the test data is transformed using training-set statistics. This prevents test data leakage. Random forest does not need scaling. Both use scikit-learn implementations so the course project can focus on experimental design and evaluation.

## Metrics and interpretation

`predict()` gives class labels; `predict_proba()[:, 1]` gives a probability score for class **1 = benign**. Precision, recall, F1, and ROC AUC therefore treat **benign as positive**. This library convention is not a statement about medical importance.

| Metric | Meaning here |
| --- | --- |
| Accuracy | Fraction of all test cases classified correctly. |
| Precision | Of cases predicted benign, fraction actually benign. |
| Recall | Of actually benign cases, fraction predicted benign. |
| F1 | Harmonic mean of benign precision and benign recall. |
| ROC AUC | How well the benign probability ranks benign above malignant across thresholds. |

**Displayed recall is not malignant recall.** A health application would need to study missed malignant cases separately. The app does not run cross-validation, tune hyperparameters, calculate confidence intervals, or save a deployable model.

## Tracking and comparison

`save_experiment` stores a UTC timestamp, model name, parameter settings, five metrics, and train/test counts. Parameterized SQL placeholders (`?`) insert values. JSON fields preserve the small settings and metrics dictionaries. `list_experiments` reads newest first and `app.py` shows a table. It stores no row-level dataset records, predictions, or model binaries.

SQLite is a single file on the machine running the app. Cloud storage may disappear when the app restarts, so this is not a durable shared experiment service. Keep the same seed and test fraction when comparing model choices or hyperparameters.

## Why these technologies

- **Python:** One language for data loading, training, and UI.
- **Streamlit:** An interactive dashboard without a separate frontend or API server.
- **scikit-learn:** Established implementations for data loading, splitting, preprocessing, models, and metrics.
- **SQLite:** A built-in single-file database sufficient for demo runs.
- **pandas:** Converts saved run dictionaries to the comparison table.
- **pytest and GitHub Actions:** Check basic behavior locally and after changes.

CSV upload, Optuna tuning, drift monitoring, and AI insights remain proposed extensions.
