# Team responsibilities and handoff

The working flow is: inspect a built-in or uploaded binary-classification dataset, train and compare models, optionally tune with Optuna, and check a new batch for numeric drift. CSV upload came through [PR #10](https://github.com/Oluwasegun-rm/modelsphere/pull/10), tuning through [PR #11](https://github.com/Oluwasegun-rm/modelsphere/pull/11), and monitoring through [PR #12](https://github.com/Oluwasegun-rm/modelsphere/pull/12). Each person uses their own GitHub account, branch, and PR for future changes.

## Oluwasegun Soji-John — integration and PyTorch

Issue: https://github.com/Oluwasegun-rm/modelsphere/issues/1

Own `app.py`, README, architecture and presentation docs, deployment, and demo. Connect teammate modules in Streamlit and verify the public URL after integration. A later small CPU-only PyTorch/Lightning classifier should live in a separate module so the existing scikit-learn baselines stay intact. It must accept the same built-in or `DatasetSpec` data, use training-only preprocessing, report the same held-out metrics and positive class, and save a normal `Result`. Add optional LLM commentary only after the core features work; send no uploaded rows or secret key to the repository.

## Dawaun Shields (@WaunCPU) — automatic tuning

Issue: https://github.com/Oluwasegun-rm/modelsphere/issues/2

Own `src/tuning.py`, `tests/test_tuning.py`, and coordinated changes in training code. The merged Optuna search uses training-only cross-validation and returns the existing `Result` with JSON-safe best settings, trial count, and validation score. Preserve `Result.parameters["dataset"]` for uploads in future changes. Keep the module usable without Streamlit or SQLite, and coordinate any shared interface change with Oluwasegun.

## Eian Capers (@Eicap1-prog) — data drift signal

Issue: https://github.com/Oluwasegun-rm/modelsphere/issues/3

Own `src/monitoring.py` and `tests/test_monitoring.py`; leave the existing SQLite format alone unless there is a clear need. The merged function compares reference and new batches, returning per-feature signals and an overall summary. It handles missing values, small batches, constant features, and mismatched columns. Keep it independent of Streamlit. This is a distribution warning, not a prediction of future drift or a measurement of accuracy; new labels would be needed to measure accuracy. Do not add email or Slack alerts without a tested delivery workflow.

## Shared handoff

1. Read `AGENTS.md`, `docs/architecture.md`, and `docs/upload.md` before editing. Run the built-in app once to understand its current behavior.
2. Tell the team your branch name and proposed function name. Stay inside the assigned files; ask before changing another person's files.
3. Open a PR when the feature works. Include a plain-language explanation, one example, test results, and anything Oluwasegun needs to connect in the dashboard.
4. After merge, Oluwasegun connects the feature, verifies the public app, updates README and the GIF if the workflow changed, and tells everyone what is now live.

## Later, only after the above is stable

Optional LLM commentary on metrics, PyTorch activation logging for a monitoring demonstration, an outbound alert, and a post-deployment health check. Dynamic Dockerfile generation and Swarm/Kubernetes autoscaling require separate infrastructure and are not claimed as complete.
