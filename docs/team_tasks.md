# Team plan for the next ModelSphere release

The immediate goal is a reliable workflow: inspect a built-in or uploaded binary-classification dataset, train and compare models, then add a small amount of honest automation. [PR #10](https://github.com/Oluwasegun-rm/modelsphere/pull/10) adds CSV upload; start new teammate branches from `main` **after** it merges. Each person uses their own GitHub account, branch, and PR.

## Oluwasegun Soji-John — integration and PyTorch

Issue: https://github.com/Oluwasegun-rm/modelsphere/issues/1

Own `app.py`, README, architecture and presentation docs, deployment, and demo. Review and merge the CSV, tuning, and monitoring PRs; show the new controls in Streamlit; retest the public URL after each merge. Implement a small CPU-only PyTorch/Lightning neural-network classifier in a separate module so the existing scikit-learn baselines stay intact. It must accept the same built-in or `DatasetSpec` data, use training-only preprocessing, report the same held-out metrics and positive class, and save a normal `Result`. Add an optional LLM explanation of saved metrics only after the core features work; send no uploaded rows or secret key to the repository.

## Dawaun Shields (@WaunCPU) — automatic tuning

Issue: https://github.com/Oluwasegun-rm/modelsphere/issues/2

Own `src/tuning.py`, `tests/test_tuning.py`, and any small changes needed in training code. Add an Optuna search for logistic regression and random forest with a modest default of about 10–15 trials. Split the chosen dataset once; search using cross-validation on the training rows; fit the best settings on all training rows; score the untouched test rows once. Return the existing `Result` with JSON-safe best settings and tuning details, including trial count and training-only validation score. Preserve `Result.parameters["dataset"]` for uploads. The module should work without Streamlit or SQLite. Coordinate any change to `train_model` or `requirements.txt` with Oluwasegun before changing the shared contract. A small test should prove the held-out test rows are excluded from tuning.

## Eian Capers (@Eicap1-prog) — data drift signal

Issue: https://github.com/Oluwasegun-rm/modelsphere/issues/3

Own `src/monitoring.py` and `tests/test_monitoring.py`; leave the existing SQLite format alone unless there is a clear need. Build a simple function that compares **reference** and **new** batches using the same numeric feature columns and returns a per-feature signal plus an overall summary. Document the calculation, threshold, missing-value handling, and minimum sample size. Test clearly shifted and similar batches, constant features, and mismatched columns. This is a distribution warning, not a prediction of future drift or a measurement of accuracy; new labels would be needed to measure accuracy. Keep the module independent of Streamlit; Oluwasegun will add its UI. Do not add email or Slack alerts yet.

## Shared handoff

1. Read `AGENTS.md`, `docs/architecture.md`, and `docs/upload.md` before editing. Run the built-in app once to understand its current behavior.
2. Tell the team your branch name and proposed function name. Stay inside the assigned files; ask before changing another person's files.
3. Open a PR when the feature works. Include a plain-language explanation, one example, test results, and anything Oluwasegun needs to connect in the dashboard.
4. After merge, Oluwasegun connects the feature, verifies the public app, updates README and the GIF if the workflow changed, and tells everyone what is now live.

## Later, only after the above is stable

Optional LLM commentary on metrics, PyTorch activation logging for a monitoring demonstration, an outbound alert, and a post-deployment health check. Dynamic Dockerfile generation and Swarm/Kubernetes autoscaling require separate infrastructure and are not claimed as complete.
