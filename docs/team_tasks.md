# Tonight's task split

## Oluwasegun Soji-John — integration, dashboard, submission

Own `app.py`, README, deployment, and demo. Integrate pull requests, verify tests, publish a public app, record `docs/demo.gif`, and invite Dr. Dacon. Final acceptance: the public URL trains both models and README matches actual features.

## WaunCPU — training engine

Own `src/training.py` and training tests. Review the baseline, add clear model/metric behavior and test edge cases. If time permits, implement cross-validated Optuna tuning on the training split. Do not tune on the holdout set. Final acceptance: deterministic tests and clean integration API.

## Eicap1-prog — tracking and monitoring

Own `src/tracking.py`, tracking tests, and optional `src/monitoring.py`. Improve SQLite reliability and add a simple documented drift signal only after persistence works. Final acceptance: saves and lists experiments correctly; monitoring states its limits.

Everyone: work in separate branches, make genuine commits, open pull requests early, and announce interface changes before merging.
