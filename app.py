"""ModelSphere Streamlit dashboard."""

import pandas as pd
import streamlit as st

from src.training import dataset_summary, train_model
from src.tracking import list_experiments, save_experiment

st.set_page_config(page_title="ModelSphere", page_icon="🔬", layout="wide")
st.title("🔬 ModelSphere")
st.caption("Texas Toast · Reproducible classification experiments")

# The UI asks the training module for facts about the bundled dataset.
# It does not load or modify patient rows itself.
summary = dataset_summary()
st.subheader("Dataset")
st.write(f"Breast Cancer Wisconsin (Diagnostic): {summary['rows']} rows, {summary['features']} numeric features. Target: {', '.join(summary['classes'])}.")
st.caption("Metric positive class: benign (class 1). Scores come from the held-out test split.")
st.info("Educational demonstration only. Predictions are not medical advice.")

with st.sidebar:
    # These controls become arguments to one shared training function.
    # Keeping the split and seed fixed makes two model runs easier to compare.
    st.header("Configure experiment")
    model = st.selectbox("Classification model", ["Logistic regression", "Random forest"])
    test_size = st.slider("Test fraction", 0.1, 0.4, 0.2, 0.05)
    seed = st.number_input("Random seed", 0, 100000, 42, 1)
    if model == "Logistic regression":
        c = st.number_input("Regularization C", min_value=0.01, max_value=100.0, value=1.0)
        n_estimators, max_depth = 100, None
    else:
        n_estimators = st.slider("Trees", 10, 300, 100, 10)
        depth = st.slider("Maximum depth (0 = unlimited)", 0, 30, 0)
        max_depth = None if depth == 0 else depth
        c = 1.0
    run = st.button("Train and save", type="primary")

if run:
    try:
        # Training returns a small summary of settings and test scores.
        # Tracking saves that summary; it does not save a model or raw dataset rows.
        result = train_model(model, test_size=test_size, random_state=int(seed), c=c,
                             n_estimators=n_estimators, max_depth=max_depth)
        run_id = save_experiment(result)
        st.success(f"Experiment #{run_id} saved. Split: {result.train_count} train / {result.test_count} test.")
        cols = st.columns(len(result.metrics))
        for col, (name, value) in zip(cols, result.metrics.items()):
            col.metric(name.replace("_", " ").upper(), f"{value:.3f}")
    except (OSError, ValueError) as exc:
        st.error(f"Experiment could not be saved: {exc}")

st.subheader("Compare saved experiments")
try:
    # Read summaries from SQLite and display them as a simple table.
    experiments = list_experiments()
    if experiments:
        frame = pd.DataFrame([{"id": item["id"], "created (UTC)": item["created_at"],
                               "model": item["model"], "parameters": str(item["parameters"]),
                               **item["metrics"]} for item in experiments])
        st.dataframe(frame, hide_index=True, use_container_width=True)
    else:
        st.write("No saved experiments yet. Train a model to start comparing runs.")
except OSError as exc:
    st.warning(f"Experiment history unavailable: {exc}")
