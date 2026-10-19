# Deployment guide

1. Push the repository to GitHub.
2. Create a Streamlit Community Cloud app using `app.py` on the default branch.
3. Verify the app opens in a private browser window, trains both models, and displays metrics.
4. Check whether experiment history survives a restart. Streamlit Community Cloud local storage may be ephemeral; document the observed behavior rather than promise persistent shared history.
5. Put the verified public URL in README.

The core app requires no secrets. A writable local directory is needed for SQLite. For a persistent multi-user deployment, move experiment storage to a hosted database; that is outside the initial course demo.
