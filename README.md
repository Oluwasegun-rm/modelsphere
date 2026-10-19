# Project Title: ModelSphere

ModelSphere is a compact machine learning experiment dashboard. It trains two classification models on the Breast Cancer Wisconsin dataset, evaluates them on a held-out test set, and saves results for side-by-side comparison. Built by team **Texas Toast** with Streamlit, scikit-learn, and SQLite.

## 👥 Team Members & Contributions

All three students have contributed commits to the repository. Their completed work and current areas of ownership are:

* **Oluwasegun Soji-John ([@Oluwasegun-rm](https://github.com/Oluwasegun-rm))** — dashboard comparison, integration, demo, deployment, and Docker packaging ([issue #1](https://github.com/Oluwasegun-rm/modelsphere/issues/1))
* **Dawaun Shields ([@WaunCPU](https://github.com/WaunCPU))** — training validation, repeatable runs, and held-out evaluation tests ([issue #2](https://github.com/Oluwasegun-rm/modelsphere/issues/2))
* **Eian Capers ([@Eicap1-prog](https://github.com/eicap1-prog))** — reliable SQLite experiment saving, simultaneous-save handling, and tracking tests ([issue #3](https://github.com/Oluwasegun-rm/modelsphere/issues/3))

The initial scaffold was created with Codex; it is not represented as a student's individual work. GitHub history shows each student's later contributions.

---

## 🚀 Live Demo

* **Application URL:** [ModelSphere on Streamlit Community Cloud](https://efpmqnyhgu7aepxpdqxxi2.streamlit.app/)

The public app was opened and both classifiers were trained successfully on October 19, 2026. Local SQLite history on Community Cloud may reset after an app restart.

### 🎥 Project Walkthrough

![Project Demo Walkthrough](docs/demo.gif)

The GIF shows actual frames from the live app: model selection, configuration, and saved results.

---

## 🛠️ Installation & Setup

### Prerequisites

Python 3.11 or 3.12, `pip`, and Git.

### Local Installation

```bash
git clone https://github.com/Oluwasegun-rm/modelsphere.git
cd modelsphere
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
streamlit run app.py
```

Run automated checks with `python -m pytest -q` and the non-UI smoke check with `python -m scripts.smoke_test`.

For a guided first run and a fair model comparison, follow the [local walkthrough](docs/local_walkthrough.md).
To build and run the same app in Docker, follow the [container guide](docs/container.md). The Docker image is checked in GitHub Actions; the public demo runs on Streamlit Community Cloud.

---

## 📊 Dataset

* **Source:** [UCI Breast Cancer Wisconsin (Diagnostic)](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic), accessed through [scikit-learn's bundled loader](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html).
* **Description:** 569 observations, 30 numeric features derived from breast mass images, and a binary malignant/benign label. In scikit-learn, class 0 is malignant and class 1 is benign.
* **Data pipeline:** `load_breast_cancer()` supplies the data with the package. The app makes a stratified 80/20 split by default; the random seed and test fraction are configurable. No data file needs to be downloaded separately.

This is an educational dataset and app, not a clinical diagnostic tool.

---

## 🧠 Model Architecture & Training

* **Algorithms:** Logistic regression with configurable regularization strength C; random forest with configurable tree count and maximum depth.
* **Frameworks:** scikit-learn for data, training, and evaluation; Streamlit for the interface; SQLite for experiment records.
* **Training process:** A stratified train/test split keeps the holdout data out of fitting. Logistic regression uses `StandardScaler` inside a training-only pipeline. Random forest uses a fixed random seed. The app saves settings and metrics, not model binaries or row-level data.

Hyperparameter tuning with Optuna, drift monitoring, CSV upload, and AI insights are proposed extensions. They are not implemented in the current release.

---

## 📈 Evaluation & Results

The following values were observed in the live demo with an 80/20 stratified split and seed 42 (455 training, 114 test rows). Test metrics use **benign (class 1)** as the positive class. Training metrics are intentionally not reported as generalization results.

| Model | Test accuracy | Test precision | Test recall | Test F1 | Test ROC AUC |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Logistic regression (C=1.0) | 0.982 | 0.986 | 0.986 | 0.986 | 0.995 |
| Random forest (100 trees, unlimited depth) | 0.956 | 0.959 | 0.972 | 0.966 | 0.994 |

These are results from one fixed split of a small, easy dataset; they do not establish clinical suitability. The dashboard reports all five metrics for each new run and displays saved runs in a comparison table.

---

## 🧑‍💻 Git Workflow & Guidelines

1. Create a feature branch for your issue, such as `feature/training-validation`.
2. Pull the latest `main` before starting and before opening a pull request.
3. Make genuine, descriptive commits from your own account; open a pull request early.
4. Run `python -m pytest -q` and keep `streamlit run app.py` working.
5. Merge working core improvements before optional features. Keep this README accurate about what is live.

The detailed task split is in [team tasks](docs/team_tasks.md), and agent development rules are in [AGENTS.md](AGENTS.md). GitHub Actions runs the tests and container checks on pushes and pull requests.

## 🎓 Understand and present the project

The [architecture guide](docs/architecture.md) explains the data flow, every major file, the training and evaluation design, model choices, metrics, and technology choices. The [presentation script](docs/presentation.md) gives the team a short live demo plan and answers to likely class questions. The [local walkthrough](docs/local_walkthrough.md) shows how to train and compare two experiments step by step.
