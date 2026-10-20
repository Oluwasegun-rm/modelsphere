# ModelSphere agent rules

## Purpose

Ship a credible, working course project by the deadline. Keep each addition understandable for an Introduction to Machine Learning class. Do not claim a feature or deployment exists until it has been run and checked.

## Ownership and coordination

- Oluwasegun Soji-John owns integration, dashboard, README, demo capture, deployment, and final verification.
- WaunCPU owns the training engine and bounded Optuna tuning in `src/training.py`, `src/tuning.py`, and their tests.
- Eicap1-prog owns experiment persistence, the transparent drift signal, and optional AI insights in `src/tracking.py`, `src/monitoring.py`, `src/insights.py`, and their tests.
- Oluwasegun owns integration of both modules in `app.py`, public verification, and any separate PyTorch/Lightning model; coordinate before changing another owner's files.
- Make changes on a named branch and open a pull request. Avoid editing another owner's files without coordinating first.
- Each student must make their own substantive commits from their own GitHub account. Do not fabricate author metadata or contributions.

## Working together with coding agents

- Start new work from the latest `main`. Ask your agent to read this file, `docs/architecture.md`, `docs/team_tasks.md`, and `docs/upload.md` before editing.
- Tell the agent your assigned task and file boundaries. It should explain the change to you in everyday language, show how to run it, and identify what it tested. You do not need to interpret a raw diff to approve your own work.
- Open a small PR with a short description and a simple example of the feature. Tell the team if you need to change a shared function's inputs or outputs before doing so.
- Do not edit `app.py`, `README.md`, `AGENTS.md`, or architecture docs from teammate branches; Oluwasegun will connect and document the finished modules. If a dependency is needed, mention it in the PR; WaunCPU may add Optuna to `requirements.txt` after coordinating.

## Technical rules

- Keep `streamlit run app.py` working after every merge.
- Use the built-in Breast Cancer Wisconsin dataset for the baseline. Keep this an educational classification demo and avoid clinical claims.
- Fix a random seed and use a stratified holdout split. Fit preprocessing only on training data to avoid leakage.
- Report the positive-class definition where metrics are shown. Never present training-set scores as test-set scores.
- Keep model settings JSON serializable. Use parameterized SQL. Do not commit `.env`, SQLite databases, generated caches, or uploaded data.
- The optional `DatasetSpec` in `src/datasets.py` carries uploaded numeric features, encoded labels, and display metadata. `train_model(..., dataset=None)` keeps the built-in demo. Return the existing `Result` shape; store only JSON-safe dataset metadata in `Result.parameters["dataset"]`. SQLite must not store uploaded rows.
- For tuning, use cross-validation on the training portion only. The held-out test portion is scored once after selecting settings. Keep trial counts small enough for the public app.
- For monitoring, compare reference and new data explicitly. A feature-distribution warning without new labels is not proof that model accuracy changed; call it a drift signal, not an AI prediction.
- Add targeted tests for training, metrics bounds, and database save/list behavior. Run `make test` before a pull request.
- If adding PyTorch, Optuna, monitoring, or AI insights, keep installation and runtime under control. Label heuristic explanations honestly; no API key should be required for the core app. Never commit API keys or send uploaded data rows to an LLM.

## Integration and submission

- Merge working parts before optional features. Resolve conflicts with both authors as needed.
- Test from a clean install, verify the live public URL, and update README feature status to match reality.
- Capture the actual working app as `docs/demo.gif` and embed it in README before submission.
- Invite the professor by exact handle `daconjam` and verify invitation status.
- Keep `docs/architecture.md` and README explicit about implemented versus planned features. Docker packaging and CI exist; dynamic autoscaling and a complete deployment pipeline do not.
