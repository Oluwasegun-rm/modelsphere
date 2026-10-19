"""ModelSphere Streamlit dashboard."""

from dataclasses import replace

import pandas as pd
import streamlit as st

from src.training import dataset_summary, train_model
from src.tracking import list_experiments, save_experiment

st.set_page_config(page_title="ModelSphere", page_icon="🔬", layout="wide")
st.title("🔬 ModelSphere")
st.caption("Texas Toast · Reproducible classification experiments")

summary = dataset_summary()
st.subheader("Dataset")
st.write(f"Breast Cancer Wisconsin (Diagnostic): {summary['rows']} rows, {summary['features']} numeric features. Target: {', '.join(summary['classes'])}.")
st.caption("Metric positive class: benign (class 1). Scores come from the held-out test split.")
st.info("Educational demonstration only. Predictions are not medical advice.")

with st.sidebar:
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
        result = train_model(model, test_size=test_size, random_state=int(seed), c=c,
                             n_estimators=n_estimators, max_depth=max_depth)
        result = replace(result, parameters={**result.parameters,
                                             "test_size": float(test_size),
                                             "random_state": int(seed)})
        run_id = save_experiment(result)
        st.success(f"Experiment #{run_id} saved. Split: {result.train_count} train / {result.test_count} test.")
        cols = st.columns(len(result.metrics))
        for col, (name, value) in zip(cols, result.metrics.items()):
            col.metric(name.replace("_", " ").upper(), f"{value:.3f}")
    except (OSError, ValueError) as exc:
        st.error(f"Experiment could not be saved: {exc}")

st.subheader("Compare saved experiments")
st.caption("For a fair comparison, keep the test fraction and random seed the same and change one model setting at a time.")
try:
    experiments = list_experiments()
    if experiments:
        rows = []
        for item in experiments:
            settings = item["parameters"]
            model_settings = {key: value for key, value in settings.items()
                              if key not in ("test_size", "random_state")}
            rows.append({"id": item["id"], "created (UTC)": item["created_at"],
                         "model": item["model"], "test fraction": settings.get("test_size", "unknown"),
                         "seed": settings.get("random_state", "unknown"),
                         "model settings": str(model_settings), **item["metrics"]})
        frame = pd.DataFrame(rows)
        st.dataframe(frame, hide_index=True, use_container_width=True)

        latest = experiments[0]
        latest_settings = latest["parameters"]
        if "test_size" in latest_settings and "random_state" in latest_settings:
            previous = next((item for item in experiments[1:]
                             if item["parameters"].get("test_size") == latest_settings["test_size"]
                             and item["parameters"].get("random_state") == latest_settings["random_state"]), None)
            if previous:
                difference = latest["metrics"]["f1"] - previous["metrics"]["f1"]
                st.info(f"Automatic comparison: run #{latest['id']} has test F1 {latest['metrics']['f1']:.3f}; "
                        f"run #{previous['id']} has {previous['metrics']['f1']:.3f} "
                        f"(difference {difference:+.3f}). Both used the same test split and seed.")
            else:
                st.caption("Save another run with the same test fraction and seed to see an automatic comparison.")
        else:
            st.caption("Older runs may not show a test fraction or seed because those settings were not recorded yet.")
    else:
        st.write("No saved experiments yet. Train a model to start comparing runs.")
except OSError as exc:
    st.warning(f"Experiment history unavailable: {exc}")
