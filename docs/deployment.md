# Deployment guide

The public app is [ModelSphere on Streamlit Community Cloud](https://efpmqnyhgu7aepxpdqxxi2.streamlit.app/). It runs `app.py` from the repository's `main` branch. On October 20, 2026, the public app trained and saved logistic regression, Optuna-tuned logistic regression, and PyTorch neural runs; accepted a small synthetic CSV; detected a simulated drift warning; and returned a real OpenAI Insights response. Random forest and uploaded-batch drift were verified in earlier live checks.

To update the deployment, merge a tested pull request into `main`, wait for Streamlit Community Cloud to rebuild, then open the public URL and repeat the relevant workflow. GitHub Actions checks tests and a Docker build, smoke run, and app health. It does not automatically deploy the Docker image or run post-deployment model tests.

The core app requires no secret. The optional OpenAI request uses `OPENAI_API_KEY` from Streamlit secrets or an environment variable. Never commit a key. SQLite needs a writable local directory; Community Cloud local history may reset after an app restart. Persistent multi-user history would require hosted storage and is outside this course demo.
