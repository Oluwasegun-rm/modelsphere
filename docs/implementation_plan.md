# Implementation plan against the course brief

This is the order for the next release. A feature counts as complete only after its PR passes tests, it works in the public app, and README and the walkthrough describe what users can actually do. Keep logistic regression and random forest as understandable baselines.

| Professor's instruction | Current status | Next step |
| --- | --- | --- |
| PyTorch environment and Docker | The single-app Docker image and container smoke check exist. PyTorch and Lightning are absent. | Add one small CPU PyTorch/Lightning classifier; verify installation and runtime on Streamlit Cloud before enabling it publicly. |
| AI monitoring and alerts | A reference-versus-new-batch numeric drift signal is integrated in the dashboard; no predictive model or alerts. | Verify the public example and uploaded batch. PyTorch hooks can later record activations; a hook alone does not detect drift. |
| Hyperparameter optimization | Optional bounded Optuna tuning uses training-only cross-validation and one final held-out test score. | Verify timing and saved results in the public app. Optuna search is not reinforcement learning. |
| Experiment tracking and LLM insights | SQLite saves settings and metrics; no LLM. | Keep this result format. Add optional LLM commentary on aggregate settings and scores, with no uploaded rows sent and no key in Git. |
| Dynamic containerization | A static Dockerfile works. | Document it accurately. Dockerfile generation, adaptive resources, and Swarm/Kubernetes scaling require an orchestrated deployment and remain later work. |
| CI/CD and deployment checks | GitHub Actions tests code and Docker app health; the app is hosted on Streamlit Cloud. | Add a public URL health check after merges if reliable. Do not claim automated container deployment or post-deployment model evaluation until built. |

## Sequence and checkpoints

1. **CSV workflow completed:** PR #10 merged, and a small valid upload was trained and saved in the public app.
2. **Teammate modules completed:** Dawaun's tuning PR #11 and Eian's monitoring PR #12 merged with their tests passing.
3. **Owner integration:** Dashboard controls for tuning and monitoring are being verified. Keep the default one-click demo path fast and retest the public URL after deployment.
4. **PyTorch/Lightning model:** Add a small CPU binary neural network in its own module. Use the same split, encoded positive class, metrics, and `Result` shape. Cap training epochs and include a deterministic test. Check package size and cloud startup before calling it live.
5. **Optional LLM insight:** Explain a saved run or comparison using only model name, settings, metrics, and counts. Provide a useful non-LLM fallback when the API key is missing or a request fails. Clearly label suggestions as generated commentary.
6. **Final presentation:** Update README, walkthrough GIF, technology table, and results using the public deployment. Each teammate should be able to explain their contribution and the limitation of the feature they built.

Do not install `transformers` or `docker-py` just to list them as technologies. Add them only if a tested feature genuinely uses them. The public app must remain available while optional work is integrated.
