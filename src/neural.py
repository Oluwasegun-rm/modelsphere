"""Small CPU neural network for the same binary classification workflow.

PyTorch defines the model and loss; Lightning runs a short, repeatable training
loop. Preprocessing is fitted on training rows only, just as in the scikit-learn
models. This module is imported only when the neural model is selected.
"""

from __future__ import annotations

import math

from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from src.datasets import DatasetSpec
from src.training import Result, load_features_and_target, split_indices

MODEL_NAME = "Neural network (PyTorch)"


def train_neural_model(*, test_size: float = 0.2, random_state: int = 42,
                       epochs: int = 10, hidden_units: int = 16,
                       learning_rate: float = 0.001,
                       dataset: DatasetSpec | None = None) -> Result:
    """Fit a one-hidden-layer classifier and score the untouched test rows."""
    if not isinstance(epochs, int) or isinstance(epochs, bool) or not 1 <= epochs <= 20:
        raise ValueError("Epochs must be a whole number from 1 to 20.")
    if hidden_units not in (8, 16, 32):
        raise ValueError("Hidden units must be 8, 16, or 32.")
    if not isinstance(learning_rate, (int, float)) or not math.isfinite(learning_rate) or not 0 < learning_rate <= 0.1:
        raise ValueError("Learning rate must be a positive number no greater than 0.1.")
    if not isinstance(test_size, (int, float)) or not math.isfinite(test_size) or not 0.1 <= test_size <= 0.4:
        raise ValueError("Test fraction must be between 0.1 and 0.4.")
    if not isinstance(random_state, int) or isinstance(random_state, bool) or not 0 <= random_state <= 2**32 - 1:
        raise ValueError("Random seed must be a nonnegative whole number.")

    try:
        import lightning as L
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:
        raise ValueError("The neural model needs PyTorch and Lightning installed.") from exc

    features, target = load_features_and_target(dataset)
    train_rows, test_rows = split_indices(target, float(test_size), random_state)
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    x_train = scaler.fit_transform(imputer.fit_transform(features[train_rows]))
    x_test = scaler.transform(imputer.transform(features[test_rows]))
    y_train, y_test = target[train_rows], target[test_rows]

    L.seed_everything(random_state, workers=True)
    torch.set_num_threads(1)

    class BinaryClassifier(L.LightningModule):
        def __init__(self) -> None:
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(x_train.shape[1], hidden_units), nn.ReLU(), nn.Linear(hidden_units, 1),
            )
            self.loss = nn.BCEWithLogitsLoss()

        def forward(self, x):
            return self.network(x)

        def training_step(self, batch, batch_idx):
            x, y = batch
            return self.loss(self(x), y)

        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters(), lr=float(learning_rate))

    training_data = TensorDataset(
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32),
    )
    loader = DataLoader(training_data, batch_size=min(64, len(training_data)), shuffle=True)
    model = BinaryClassifier()
    trainer = L.Trainer(
        accelerator="cpu", devices=1, max_epochs=epochs, deterministic=True,
        logger=False, enable_checkpointing=False, enable_progress_bar=False,
        enable_model_summary=False, num_sanity_val_steps=0,
    )
    trainer.fit(model, loader)

    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(x_test, dtype=torch.float32))
        probabilities = torch.sigmoid(logits).cpu().numpy().reshape(-1)
    predictions = (probabilities >= 0.5).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, pos_label=1, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, pos_label=1, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
    }
    parameters = {
        "epochs": epochs, "hidden_units": hidden_units, "learning_rate": float(learning_rate),
        "test_size": float(test_size), "random_state": random_state,
    }
    if dataset is not None:
        parameters["dataset"] = dataset.metadata()
    return Result(MODEL_NAME, parameters, metrics, len(train_rows), len(test_rows))
