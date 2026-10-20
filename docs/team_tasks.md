# Team responsibilities and handoff

The working flow is: inspect a built-in or uploaded binary-classification dataset, train and compare models, optionally tune with Optuna, check a new batch for numeric drift, and request plain-language AI insights on saved aggregate results. CSV upload came through [PR #10](https://github.com/Oluwasegun-rm/modelsphere/pull/10), tuning through [PR #11](https://github.com/Oluwasegun-rm/modelsphere/pull/11), monitoring through [PR #12](https://github.com/Oluwasegun-rm/modelsphere/pull/12), PyTorch through [PR #14](https://github.com/Oluwasegun-rm/modelsphere/pull/14), and AI insights through [PR #15](https://github.com/Oluwasegun-rm/modelsphere/pull/15). Each person uses their own GitHub account, branch, and PR for future changes.

## Oluwasegun Soji-John — integration and PyTorch

Issue: https://github.com/Oluwasegun-rm/modelsphere/issues/1

Own `app.py`, README, architecture and presentation docs, deployment, and demo. The merged dashboard connects the teammate modules and a small CPU-only PyTorch/Lightning classifier in `src/neural.py`; it uses the same dataset contract, held-out metrics, and saved `Result` shape. The public app and refreshed GIF were checked for submission. Keep the scikit-learn baselines and the no-raw-rows AI boundary intact.

## Dawaun Shields (@WaunCPU) — automatic tuning

Issue: https://github.com/Oluwasegun-rm/modelsphere/issues/2

Own `src/tuning.py`, `tests/test_tuning.py`, and coordinated changes in training code. The merged Optuna search uses training-only cross-validation and returns the existing `Result` with JSON-safe best settings, trial count, and validation score. Preserve `Result.parameters["dataset"]` for uploads in future changes. Keep the module usable without Streamlit or SQLite, and coordinate any shared interface change with Oluwasegun.

## Eian Capers (@Eicap1-prog) — data drift signal

Issue: https://github.com/Oluwasegun-rm/modelsphere/issues/3

Own `src/tracking.py`, `src/monitoring.py`, `src/insights.py`, and their tests; leave the existing SQLite format alone unless there is a clear need. The merged drift function compares reference and new batches, returning per-feature signals and an overall summary. It handles missing values, small batches, constant features, and mismatched columns. The optional AI module summarizes allowlisted aggregate experiment results and falls back to a local explanation if the API is unavailable. Keep the modules independent of Streamlit where possible. Drift is a distribution warning, not a prediction of future drift or a measurement of accuracy; new labels would be needed to measure accuracy.

## Shared handoff

1. Read `AGENTS.md`, `docs/architecture.md`, and `docs/upload.md` before editing. Run the built-in app once to understand its current behavior.
2. Tell the team your branch name and proposed function name. Stay inside the assigned files; ask before changing another person's files.
3. Open a PR when the feature works. Include a plain-language explanation, one example, test results, and anything Oluwasegun needs to connect in the dashboard.
4. After merge, Oluwasegun connects the feature, verifies the public app, updates README and the GIF if the workflow changed, and tells everyone what is now live.

## Outside this submission

PyTorch activation logging for monitoring, predictive drift alerts, an outbound alert service, dynamic Dockerfile generation, and Swarm/Kubernetes autoscaling need additional implementation and infrastructure. The current static container and GitHub Actions checks are documented in `docs/container.md` and `docs/deployment.md`; they are not claimed as a complete automated deployment pipeline.
