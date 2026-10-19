# ModelSphere agent rules

## Purpose

Ship a credible, working course project by the deadline. Keep the app small and verifiable. Do not claim a feature or deployment exists until it has been run and checked.

## Ownership and coordination

- Oluwasegun Soji-John owns integration, dashboard, README, demo capture, deployment, and final verification.
- WaunCPU owns the training engine and its tests in `src/training.py` and `tests/test_training.py`.
- Eicap1-prog owns experiment persistence and simple monitoring in `src/tracking.py`, `src/monitoring.py`, and related tests.
- Make changes on a named branch and open a pull request. Avoid editing another owner's files without coordinating first.
- Each student must make their own substantive commits from their own GitHub account. Do not fabricate author metadata or contributions.

## Technical rules

- Keep `streamlit run app.py` working after every merge.
- Use the built-in Breast Cancer Wisconsin dataset for the baseline. Keep this an educational classification demo and avoid clinical claims.
- Fix a random seed and use a stratified holdout split. Fit preprocessing only on training data to avoid leakage.
- Report the positive-class definition where metrics are shown. Never present training-set scores as test-set scores.
- Keep model settings JSON serializable. Use parameterized SQL. Do not commit `.env`, SQLite databases, generated caches, or uploaded data.
- Add targeted tests for training, metrics bounds, and database save/list behavior. Run `make test` before a pull request.
- If adding Optuna, monitoring, or AI insights, keep optional dependencies and runtime under control. Label heuristic explanations honestly; no API key should be required for the core app.

## Integration and submission

- Merge working parts before optional features. Resolve conflicts with both authors as needed.
- Test from a clean install, verify the live public URL, and update README feature status to match reality.
- Capture the actual working app as `docs/demo.gif` and embed it in README before submission.
- Invite the professor by exact handle `daconjam` and verify invitation status.
