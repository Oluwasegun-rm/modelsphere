# Implementation plan against the course brief

This is the order for the next release. A feature counts as complete only after its PR passes tests, it works in the public app, and README and the walkthrough describe what users can actually do. Keep logistic regression and random forest as understandable baselines.

| Professor's instruction | Current status | Next step |
| --- | --- | --- |
| PyTorch environment and Docker | The single-app Docker image and container smoke check exist. PyTorch and Lightning are absent. | Add one small CPU PyTorch/Lightning classifier; verify installation and runtime on Streamlit Cloud before enabling it publicly. |
| AI monitoring and alerts | No monitoring or alerts. | Eian builds an honest reference-versus-new-batch numeric drift signal. UI and optional alert follow after verification. PyTorch hooks can later record activations; a hook alone does not detect drift. |
| Hyperparameter optimization | Users set parameters manually. | Dawaun adds bounded Optuna tuning with training-only cross-validation and one final held-out test score. Optuna search is not reinforcement learning. |
| Experiment tracking and LLM insights | SQLite saves settings and metrics; no LLM. | Keep this result format. Add optional LLM commentary on aggregate settings and scores, with no uploaded rows sent and no key in Git. |
| Dynamic containerization | A static Dockerfile works. | Document it accurately. Dockerfile generation, adaptive resources, and Swarm/Kubernetes scaling require an orchestrated deployment and remain later work. |
| CI/CD and deployment checks | GitHub Actions tests code and Docker app health; the app is hosted on Streamlit Cloud. | Add a public URL health check after merges if reliable. Do not claim automated container deployment or post-deployment model evaluation until built. |

## Sequence and checkpoints

1. **Integrate CSV workflow:** Resolve PR #10 against `main`; confirm the built-in path still works, a small valid upload trains both models, invalid data shows a useful message, and no uploaded rows are stored. Merge and verify the public URL.
2. **Parallel teammate modules:** Dawaun works only in tuning/training files. Eian works only in monitoring files. Both start from the merged `main`, run tests, and open separate PRs with their callable function and example output in the description.
3. **Owner integration:** Add dashboard controls for tuning and monitoring after their module PRs pass. Keep the default one-click demo path fast.
4. **PyTorch/Lightning model:** Add a small CPU binary neural network in its own module. Use the same split, encoded positive class, metrics, and `Result` shape. Cap training epochs and include a deterministic test. Check package size and cloud startup before calling it live.
5. **Optional LLM insight:** Explain a saved run or comparison using only model name, settings, metrics, and counts. Provide a useful non-LLM fallback when the API key is missing or a request fails. Clearly label suggestions as generated commentary.
6. **Final presentation:** Update README, walkthrough GIF, technology table, and results using the public deployment. Each teammate should be able to explain their contribution and the limitation of the feature they built.

Do not install `transformers` or `docker-py` just to list them as technologies. Add them only if a tested feature genuinely uses them. The public app must remain available while optional work is integrated.
