# Run ModelSphere with Docker

Docker packages the current Streamlit app and its Python dependencies in one image.
Install Docker Desktop or another Docker engine, then run these commands from the
repository root:

```bash
docker build -t modelsphere .
docker run --rm -p 8501:8501 modelsphere
```

Open <http://localhost:8501>. Train a model and check that its saved run appears in
the comparison table. The container stores its SQLite file inside the container;
that history disappears when the container is removed. To keep history between
restarts, mount a local folder:

```bash
mkdir -p .local
docker run --rm -p 8501:8501 -v "$(pwd)/.local:/app/.local" modelsphere
```

On Windows PowerShell, replace `$(pwd)` with `${PWD}` if needed. The
`.local` directory is ignored by Git. Do not commit database files.

The GitHub Actions container workflow builds the image, runs the training and
SQLite smoke check inside it, and checks that the Streamlit server responds.
This is a single-container package. It does not yet generate Dockerfiles,
scale containers, or deploy the image automatically.
