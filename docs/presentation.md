# Class presentation and demo script

Aim for **5–7 minutes**, then leave time for questions. Use your own words. Rehearse the public app before class; have the GIF and a local app ready if cloud history resets.

## Suggested speaking order

| Time | Speaker | Show | Main point |
| --- | --- | --- | --- |
| 0:00–0:45 | Oluwasegun | README and app | "ModelSphere compares classifier settings on held-out data." |
| 0:45–1:45 | WaunCPU | `src/training.py`, dataset summary | "The Wisconsin dataset has 569 rows, 30 numeric features, and two labels. We use a stratified 80/20 split with seed 42." |
| 1:45–2:45 | WaunCPU | Model controls and Optuna option | "Users can choose settings or try a small search. Optuna validates settings on training folds; final test rows remain untouched." |
| 2:45–3:45 | Oluwasegun | Train logistic regression | "The scaler and model are fit only on training rows. These scores come from untouched test rows." |
| 3:45–4:45 | Eicap1-prog | Train random forest; comparison table | "Each run saves its settings and scores. We keep seed and split fixed to compare fairly." |
| 4:45–5:30 | Eicap1-prog | Tracking and drift check | "SQLite stores run summaries. A second batch can be compared with training features for a simple distribution warning." |
| 5:30–6:15 | Oluwasegun | Tests and GitHub history | "Tests cover the whole workflow. The live app and GIF show the working scope." |

Adjust speaking assignments to the code and commits each person actually contributes. Do not claim another person's work.

## Live demo clicks

1. Open the public URL in README. Show dataset size and positive-class note.
2. Select logistic regression, C = 1.0, test fraction 0.20, seed 42. Click **Train and save**.
3. Read test accuracy and F1; explain what a held-out score means.
4. Select random forest, 100 trees, unlimited depth. Keep split and seed fixed. Click **Train and save**.
5. Scroll to the comparison table and point to settings and metrics for both runs.
6. Turn on **Tune automatically with Optuna** and run a small search. Explain that its validation score is from training folds and the metric cards are from held-out test rows.
7. Under **Check data drift**, select **Simulated shift** and click **Check drift**. Show the flagged features and explain that this is an artificial input change, not measured accuracy loss.
8. If the public app is slow, show `docs/demo.gif` and switch to the verified local app.

## Likely questions and short answers

**Why logistic regression?** It is a simple linear classifier and useful baseline. Smaller `C` means stronger regularization.

**Why random forest?** Trees can represent nonlinear patterns. Multiple trees reduce reliance on one tree.

**Why scale only logistic regression?** Its optimization is sensitive to feature magnitude. Tree splits do not need standard scaling. The pipeline fits scaling on training data to avoid leakage.

**What is leakage?** Test-set information influencing training or preprocessing, making a test score less trustworthy.

**What is stratification?** Keeping approximately the same benign/malignant proportions in train and test subsets.

**Does recall measure missed cancer cases?** No. Benign is class 1, so displayed recall is **benign recall**. Malignant recall would need a separate calculation.

**Why can the seed change scores?** It changes which examples are held out. Keep the split fixed when comparing settings.

**Does the app tune models or monitor drift?** It offers bounded Optuna tuning and a simple numeric input-drift signal. Optuna is not reinforcement learning. The drift signal does not predict future changes or measure accuracy without labels for the new batch.

**What persists?** Local SQLite history persists on durable local storage. Streamlit Community Cloud may reset its local file.

**What are the limitations?** The built-in example is one small dataset; uploaded data must have numeric features and a two-class target. One holdout split cannot establish external validity. The app has no confidence intervals, medical calibration, predictive alerts, or dynamic scaling. It is a classroom experiment, not a clinical tool.
