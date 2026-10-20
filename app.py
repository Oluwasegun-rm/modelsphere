"""ModelSphere Streamlit dashboard."""

import pandas as pd
import streamlit as st
from sklearn.datasets import load_breast_cancer

from src.datasets import BUILTIN_DATASET_ID, prepare_dataset, read_csv
from src.insights import MAX_EXPERIMENTS, generate_insights
from src.monitoring import detect_numeric_drift
from src.neural import MODEL_NAME as NEURAL_MODEL, train_neural_model
from src.training import dataset_summary, split_indices, train_model
from src.tracking import list_experiments, save_experiment
from src.tuning import tune_model

st.set_page_config(page_title="ModelSphere", page_icon="🔬", layout="wide")
st.title("🔬 ModelSphere")
st.caption("Texas Toast · Reproducible classification experiments")

st.subheader("Dataset")
source = st.radio("Data source", ["Built-in demo", "Upload CSV"], horizontal=True)
dataset = None
dataset_id = BUILTIN_DATASET_ID if source == "Built-in demo" else None
ready = source == "Built-in demo"
if ready:
    summary = dataset_summary()
    st.write(f"Breast Cancer Wisconsin (Diagnostic): {summary['rows']} rows, {summary['features']} numeric features. Target: {', '.join(summary['classes'])}.")
    st.caption("Metric positive class: benign (class 1). Scores come from the held-out test split.")
    st.info("Educational demonstration only. Predictions are not medical advice.")
else:
    st.caption("Upload a UTF-8 CSV under 5 MB. This demo supports numeric features and a two-class target. Uploaded rows are not saved with experiment history.")
    upload = st.file_uploader("Choose CSV file", type="csv")
    if upload is not None:
        try:
            frame = read_csv(upload.getvalue())
            st.write(f"{len(frame)} rows · {len(frame.columns)} columns")
            st.dataframe(frame.head(10), hide_index=True, use_container_width=True)
            quality = pd.DataFrame({
                "column": frame.columns,
                "type": [str(frame[column].dtype) for column in frame.columns],
                "missing": [int(frame[column].isna().sum()) for column in frame.columns],
            })
            with st.expander("Inspect columns and missing values"):
                st.dataframe(quality, hide_index=True, use_container_width=True)
            # Many classroom CSVs put the answer last. Users can still choose any column.
            target_name = st.selectbox("Target column (what to predict)", frame.columns,
                                       index=len(frame.columns) - 1)
            numeric_features = [column for column in frame.columns if column != target_name
                                and pd.api.types.is_numeric_dtype(frame[column])]
            text_features = [column for column in frame.columns if column != target_name
                             and column not in numeric_features]
            if text_features:
                st.caption(f"Text columns are not used in this version: {', '.join(text_features)}")
            feature_names = st.multiselect("Numeric feature columns", numeric_features,
                                           default=numeric_features[:50])
            labels = frame[target_name].astype("string").str.strip().replace("", pd.NA)
            class_counts = labels.dropna().value_counts()
            st.write("Target class counts", class_counts.rename_axis("class").reset_index(name="rows"))
            if len(class_counts) == 2:
                options = [str(label) for label in class_counts.index]
                positive_label = st.selectbox("Positive class for precision, recall and F1", options)
                dataset = prepare_dataset(frame, name=upload.name, target_name=target_name,
                                          feature_names=feature_names, positive_label=positive_label)
                dataset_id = dataset.dataset_id
                ready = True
                st.success(f"Ready to train: {len(dataset.target)} labelled rows and {len(feature_names)} features.")
                if dataset.excluded_target_rows:
                    st.warning(f"{dataset.excluded_target_rows} rows with no target will be excluded.")
                if dataset.duplicate_rows:
                    st.warning(f"{dataset.duplicate_rows} exact duplicate rows found. They will be kept; check whether they are intentional.")
                high_missing = [name for name, count in dataset.missing_by_feature.items()
                                if count / len(dataset.target) > 0.2]
                if high_missing:
                    st.warning(f"More than 20% of values are missing in: {', '.join(high_missing)}. Missing numeric values will be filled using training-data medians.")
                elif any(dataset.missing_by_feature.values()):
                    st.caption("Missing numeric values will be filled using training-data medians.")
                st.caption("Check that selected features are available before the outcome is known; exclude IDs and answer-revealing columns.")
            else:
                st.error("Choose a target with exactly two non-empty classes.")
        except ValueError as exc:
            st.error(f"Dataset needs attention: {exc}")

with st.sidebar:
    # These controls become arguments to one shared training function.
    # Keeping the split and seed fixed makes two model runs easier to compare.
    st.header("Configure experiment")
    model = st.selectbox("Classification model", ["Logistic regression", "Random forest", NEURAL_MODEL])
    test_size = st.slider("Test fraction", 0.1, 0.4, 0.2, 0.05)
    seed = st.number_input("Random seed", 0, 100000, 42, 1)
    if model == NEURAL_MODEL:
        st.caption("Small CPU neural network trained with PyTorch and Lightning.")
        epochs = st.slider("Training epochs", 1, 20, 10)
        hidden_units = st.selectbox("Hidden units", [8, 16, 32], index=1)
        learning_rate = st.selectbox("Learning rate", [0.0001, 0.001, 0.01], index=1)
        auto_tune = False
        c, n_estimators, max_depth = 1.0, 100, None
    else:
        auto_tune = st.checkbox("Tune automatically with Optuna", value=False)
        if auto_tune:
            trials = st.slider("Tuning trials", 3, 20, 12)
            st.caption("Tries settings on training rows only, then scores the test rows once.")
            c, n_estimators, max_depth = 1.0, 100, None
        elif model == "Logistic regression":
            c = st.number_input("Regularization C", min_value=0.01, max_value=100.0, value=1.0)
            n_estimators, max_depth = 100, None
        else:
            n_estimators = st.slider("Trees", 10, 300, 100, 10)
            depth = st.slider("Maximum depth (0 = unlimited)", 0, 30, 0)
            max_depth = None if depth == 0 else depth
            c = 1.0
    run = st.button("Tune, train and save" if auto_tune else "Train and save",
                    type="primary", disabled=not ready)

if run:
    try:
        # Training returns a small summary of settings and test scores.
        # Tracking saves that summary; it does not save a model or raw dataset rows.
        if model == NEURAL_MODEL:
            with st.spinner("Training the small neural network on CPU…"):
                result = train_neural_model(test_size=test_size, random_state=int(seed),
                                            epochs=epochs, hidden_units=hidden_units,
                                            learning_rate=learning_rate, dataset=dataset)
        elif auto_tune:
            result = tune_model(model, n_trials=trials, test_size=test_size,
                                random_state=int(seed), dataset=dataset)
        else:
            result = train_model(model, test_size=test_size, random_state=int(seed), c=c,
                                 n_estimators=n_estimators, max_depth=max_depth, dataset=dataset)
        run_id = save_experiment(result)
        st.success(f"Experiment #{run_id} saved. Split: {result.train_count} train / {result.test_count} test.")
        if auto_tune:
            tuning = result.parameters["tuning"]
            st.caption(f"Best training-only cross-validation ROC AUC: "
                       f"{tuning['validation_score']:.3f} across {tuning['trials']} trials. "
                       "The scores below come from the held-out test rows.")
        cols = st.columns(len(result.metrics))
        for col, (name, value) in zip(cols, result.metrics.items()):
            col.metric(name.replace("_", " ").upper(), f"{value:.3f}")
    except (OSError, ValueError) as exc:
        st.error(f"Experiment could not be saved: {exc}")

st.subheader("Compare saved experiments")
st.caption("Only runs for this dataset and feature selection appear here. For a fair comparison, keep the test fraction and random seed the same.")
try:
    # Older built-in runs have no dataset metadata; uploaded runs carry a fingerprint.
    experiments = [item for item in list_experiments()
                   if ready and item["parameters"].get("dataset", {}).get("id", BUILTIN_DATASET_ID) == dataset_id]
    if experiments:
        rows = []
        for item in experiments:
            settings = item["parameters"]
            tuning = settings.get("tuning") or {}
            model_settings = {key: value for key, value in settings.items()
                              if key not in ("test_size", "random_state", "dataset", "tuning")}
            rows.append({"id": item["id"], "created (UTC)": item["created_at"],
                         "model": item["model"], "method": "Optuna" if tuning else "Manual",
                         "validation ROC AUC": tuning.get("validation_score"),
                         "test fraction": settings.get("test_size"),
                         "seed": settings.get("random_state"),
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
        if ready:
            st.write("No saved experiments for this dataset yet. Train a model to start comparing runs.")
        else:
            st.write("Choose a valid dataset to see its experiment history.")
except OSError as exc:
    st.warning(f"Experiment history unavailable: {exc}")

st.subheader("Check data drift")
st.caption("Compare the numeric features used for training with a new batch. This checks input data, not model accuracy.")
if ready:
    example = "Similar data"
    if source == "Built-in demo":
        batch_source = st.radio("New batch", ["Built-in example", "Upload another CSV"], horizontal=True)
        if batch_source == "Built-in example":
            example = st.selectbox("Example", ["Similar data", "Simulated shift"])
            st.caption("The simulated shift adds an artificial change to eight features for demonstration.")
    else:
        batch_source = "Upload another CSV"
    new_upload = None
    if batch_source == "Upload another CSV":
        st.caption("Upload a second CSV with the same numeric feature columns. A target column is optional; at least 30 rows per feature are needed.")
        new_upload = st.file_uploader("New batch CSV", type="csv", key="drift_batch")
    check_drift = st.button("Check drift", disabled=batch_source == "Upload another CSV" and new_upload is None)
    if check_drift:
        try:
            if dataset is None:
                built_in = load_breast_cancer(as_frame=True)
                reference_features = built_in.data
                reference_target = built_in.target.to_numpy()
            else:
                reference_features = dataset.features
                reference_target = dataset.target
            reference_rows, example_rows = split_indices(reference_target, float(test_size), int(seed))
            reference_batch = reference_features.iloc[reference_rows]

            if batch_source == "Built-in example":
                new_batch = reference_features.iloc[example_rows].copy()
                if example == "Simulated shift":
                    for column in new_batch.columns[:8]:
                        new_batch[column] += 2 * reference_batch[column].std()
            else:
                new_batch = read_csv(new_upload.getvalue())
            report = detect_numeric_drift(reference_batch, new_batch)
            if report.status == "warning":
                st.warning(report.message)
            elif report.status == "watch":
                st.info(report.message)
            elif report.status == "not_enough_data":
                st.warning(report.message)
            else:
                st.success(report.message)
            st.dataframe(pd.DataFrame(report.to_dict()["features"]), hide_index=True,
                         use_container_width=True)
            st.caption(report.limits)
        except (TypeError, ValueError) as exc:
            st.error(f"Drift check could not run: {exc}")
else:
    st.write("Choose a valid dataset first.")

st.subheader("AI Experiment Insights")
st.caption("Optional plain-language commentary on saved runs. Model settings, row counts, test metrics, "
           "tuning results, dataset kind, and feature counts are sent to OpenAI, never your data rows, filenames or column "
           "names. It runs only when you click the button.")
if ready:
    try:
        insight_runs = [item for item in list_experiments()
                        if item["parameters"].get("dataset", {}).get("id", BUILTIN_DATASET_ID) == dataset_id]
    except OSError as exc:
        insight_runs = []
        st.warning(f"Experiment history unavailable: {exc}")
    if insight_runs:
        run_ids = [item["id"] for item in insight_runs]
        picked = st.multiselect(f"Runs to explain (up to {MAX_EXPERIMENTS}; pick two or more to compare)", run_ids,
                                default=run_ids[:2], max_selections=MAX_EXPERIMENTS,
                                format_func=lambda run_id: f"Run #{run_id}")
        if st.button("Generate AI Insights", disabled=not picked):
            chosen = [item for item in insight_runs if item["id"] in picked]
            try:
                with st.spinner("Writing insights..."):
                    st.session_state["insight_result"] = (tuple(sorted(picked)), generate_insights(chosen))
            except ValueError as exc:
                st.session_state.pop("insight_result", None)
                st.error(f"Insights could not be created: {exc}")
        saved = st.session_state.get("insight_result")
        if saved and saved[0] == tuple(sorted(picked)):
            outcome = saved[1]
            if outcome.source == "ai":
                st.markdown("**AI-generated suggestions** (not measured results; check them against the table above):")
            else:
                st.markdown("**Local summary** (no AI was used; computed from the saved metrics):")
                if outcome.note:
                    st.info(outcome.note)
            st.write(outcome.text)
        st.caption("Metrics come from held-out test rows. AI text is an explanation aid, may be wrong, and never changes any setting.")
    else:
        st.write("Save at least one experiment for this dataset to get insights.")
else:
    st.write("Choose a valid dataset first.")
