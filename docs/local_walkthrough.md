# Local setup and experiment walkthrough

## Set up

Use Python 3.11 or 3.12. From a terminal:

```bash
git clone https://github.com/Oluwasegun-rm/modelsphere.git
cd modelsphere
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
python -m scripts.smoke_test
streamlit run app.py
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`. Streamlit prints a local address, usually `http://localhost:8501`. No API key or separate dataset download is needed. If `streamlit` is not found, check that the environment is active and installation finished. If SQLite cannot write its file, check folder permissions or set `MODELSPHERE_DB_PATH` to a writable path.

## Train the first model

1. Leave **Logistic regression**, **test fraction 0.20**, **seed 42**, and **C = 1.0** selected.
2. Click **Train and save**. The app shows training/test row counts and five **test** metrics.
3. Explain that test rows were held back from both scaling and model fitting. The scaler learned its values from training rows inside the pipeline.
4. Find the new row under **Compare saved experiments**. It contains settings and metrics, not the trained model or patient rows.

## Compare a second run

1. Switch to **Random forest**. Keep **test fraction 0.20** and **seed 42** so the holdout is the same.
2. Leave **100 trees** and **unlimited depth**, then click **Train and save**.
3. Compare the rows. Say which model had higher test accuracy or F1 **on this split**; do not claim one algorithm is universally better.
4. Change tree depth or tree count and run again to show how a hyperparameter changes the experiment.

History is stored in `.local/modelsphere.db` by default. The app creates it automatically. For a fresh local demo, set `MODELSPHERE_DB_PATH` to a new database path before starting Streamlit; avoid deleting another teammate's runs.

## Interpret the metrics

The dataset labels **malignant = 0** and **benign = 1**. Precision, recall, F1, and ROC AUC use **benign as positive**. A score from one held-out sample is not a guarantee on future data. Keep seed and split constant while comparing model settings.

## Verify before presenting

- `python -m pytest -q` passes.
- `python -m scripts.smoke_test` reports a saved run.
- The local or public app opens, trains the two scikit-learn baselines, and displays two saved rows. The PyTorch model, Optuna tuning, drift check, and AI insights are additional demo paths.
- The README's public URL and embedded GIF open correctly.
- Each speaker can explain the positive class, split, scaling pipeline, and database limitation.
