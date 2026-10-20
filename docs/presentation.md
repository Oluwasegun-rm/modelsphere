# Class presentation and demo script

Aim for **5–7 minutes**, then leave time for questions. Use your own words. Rehearse the public app before class; have the GIF and a local app ready if cloud history resets.

## Suggested speaking order

| Time | Speaker | Show | Main point |
| --- | --- | --- | --- |
| 0:00–0:40 | Oluwasegun | README and app | "ModelSphere compares classifier settings on held-out data." |
| 0:40–1:50 | Dawaun | Dataset, training controls, first result | "We use 569 rows and a stratified 80/20 split. Preprocessing learns from training rows only." |
| 1:50–2:40 | Dawaun | Optuna option | "Optuna checks settings on training folds; the held-out test rows are scored once." |
| 2:40–3:35 | Oluwasegun | PyTorch and CSV controls, or GIF | "The same workflow also accepts a small numeric CSV and a bounded Lightning neural model." |
| 3:35–4:45 | Eian | Saved-run comparison | "SQLite stores settings and scores. Keep the seed and split fixed for fair comparisons." |
| 4:45–5:55 | Eian | Drift check and AI insights | "Drift checks inputs; the AI explains aggregate saved results. Neither establishes future model accuracy." |
| 5:55–6:30 | Oluwasegun | GIF, tests, limitations | "The public demo and automated checks show the working scope; predictive alerts and autoscaling remain outside it." |

Adjust speaking assignments to the code and commits each person actually contributes. Do not claim another person's work.

## Live demo clicks

1. Open the public URL in README. Show dataset size and positive-class note.
2. Select logistic regression, C = 1.0, test fraction 0.20, seed 42. Click **Train and save**.
3. Read test accuracy and F1; explain what a held-out score means.
4. Select random forest, 100 trees, unlimited depth. Keep split and seed fixed. Click **Train and save**.
5. Scroll to the comparison table and point to settings and metrics for both runs. Briefly show **Upload CSV** and explain the target, numeric features, and positive-class choices; the GIF has a synthetic example without needing another live upload.
6. Show **Tune automatically with Optuna** and explain that validation uses training folds while the metric cards use held-out test rows. Run three trials only if time allows.
7. Show **Neural network (PyTorch)** controls. Explain that the small Lightning model uses the same split and test metrics; use the GIF or saved run to show its result if time is short.
8. Under **Check data drift**, select **Simulated shift** and click **Check drift**. Show the flagged features and explain that this is an artificial input change, not measured accuracy loss.
9. Under **AI Experiment Insights**, select two runs and click **Generate AI Insights**. The public app returned a real AI response in the October 20 check. Point out the label that separates suggestions from measured results. If the API is unavailable, the same button shows a local summary.
10. If the public app is slow, show `docs/demo.gif` and switch to the verified local app.

## Likely questions and short answers

**Why logistic regression?** It is a simple linear classifier and useful baseline. Smaller `C` means stronger regularization.

**Why random forest?** Trees can represent nonlinear patterns. Multiple trees reduce reliance on one tree.

**Why scale logistic regression and the neural network?** Both learn weights that are sensitive to feature magnitude. Tree splits do not need standard scaling. Preprocessing is fitted on training data to avoid leakage.

**What is leakage?** Test-set information influencing training or preprocessing, making a test score less trustworthy.

**What is stratification?** Keeping approximately the same benign/malignant proportions in train and test subsets.

**Does recall measure missed cancer cases?** No. Benign is class 1, so displayed recall is **benign recall**. Malignant recall would need a separate calculation.

**Why can the seed change scores?** It changes which examples are held out. Keep the split fixed when comparing settings.

**Does the app tune models or monitor drift?** It offers bounded Optuna tuning and a simple numeric input-drift signal. Optuna is not reinforcement learning. The drift signal does not predict future changes or measure accuracy without labels for the new batch.

**Does the AI see my data?** It sees an allowlisted summary: model settings, row counts, test metrics, tuning results, dataset kind, and feature counts, only after the button is clicked. Dataset rows, filenames, and column names are never sent. Its text is a suggestion that can be wrong, and it cannot change any setting.

**What if there is no API key or OpenAI is down?** The app keeps working and shows a local summary computed from the saved metrics.

**What persists?** Local SQLite history persists on durable local storage. Streamlit Community Cloud may reset its local file.

**What are the limitations?** The built-in example is one small dataset; uploaded data must have numeric features and a two-class target. One holdout split cannot establish external validity. The app has no confidence intervals, medical calibration, predictive alerts, or dynamic scaling. It is a classroom experiment, not a clinical tool.
