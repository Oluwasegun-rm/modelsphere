# ModelSphere

**Team:** Texas Toast  
**Project:** A small, reproducible machine learning experimentation dashboard  
**Live demo:** Pending deployment  
**Demo recording:** Pending capture; see [demo plan](docs/demo.md)

## Overview

ModelSphere lets a user train a classification model on the built-in Breast Cancer Wisconsin (Diagnostic) dataset, inspect holdout metrics, save a run, and compare past runs. It uses Streamlit, scikit-learn, and SQLite. The dataset is loaded from scikit-learn; no dataset download is needed.

## Run locally

Python 3.11 or 3.12 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
make install
make run
```

Open the local address shown by Streamlit. To run tests: `make test`.

## How to use

1. Choose logistic regression or random forest in the sidebar.
2. Set the test fraction and random seed, then configure the model.
3. Select **Train and save** to see accuracy, precision, recall, F1, and ROC AUC.
4. Compare rows in the saved experiments table.

The app uses a stratified train/test split. Logistic regression scales features inside a pipeline fitted on the training set only. The test set is held out from fitting. Results are educational and are not medical advice.
Precision, recall, F1, and ROC AUC treat **benign (class 1)** as the positive class.

## Current scope and limitations

| Feature | Status |
| --- | --- |
| Built-in Wisconsin dataset | Implemented |
| Two classification algorithms and configurable parameters | Implemented |
| Holdout evaluation with five metrics | Implemented |
| SQLite saved run comparison | Implemented |
| CSV upload and target selection | Planned |
| Optuna hyperparameter tuning | Planned |
| Drift monitoring | Planned |
| AI insights | Planned |

SQLite history is stored at `.local/modelsphere.db` by default. On cloud hosting with ephemeral storage, saved runs may disappear after a restart. This demo has no user accounts and should not be used for sensitive data.

## Project structure

- `app.py`: Streamlit interface
- `src/training.py`: dataset, model training, metrics
- `src/tracking.py`: SQLite persistence
- `tests/`: automated checks
- `docs/`: architecture, deployment, demo plan, team work
- `scripts/`: helper scripts

## Team contributions

Work is split in [team tasks](docs/team_tasks.md). Each teammate should create a branch, make genuine commits for their own work, and open a pull request. GitHub history is the record of contributions; this README will be updated with each member's actual commits before submission.

Task issues: [integration and deployment](https://github.com/Oluwasegun-rm/modelsphere/issues/1), [training engine](https://github.com/Oluwasegun-rm/modelsphere/issues/2), and [tracking and monitoring](https://github.com/Oluwasegun-rm/modelsphere/issues/3). GitHub will allow formal assignment of the teammate issues after their pending collaborator invitations are accepted.

## Deployment

See [deployment guide](docs/deployment.md). Add the verified public demo URL here after deployment.

## Academic submission checklist

- [ ] All three members have genuine commits
- [ ] Record `demo.gif`, embed it here, and verify it plays on GitHub
- [ ] Publish and verify a functional public demo URL
- [ ] Invite Dr. Dacon (`daconjam`) as collaborator
- [ ] Update implemented feature status and contribution links

<!-- Once recorded, embed with: ![ModelSphere demo](docs/demo.gif) -->
