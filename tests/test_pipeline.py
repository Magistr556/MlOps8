import json
from pathlib import Path

import joblib
import numpy as np


def test_model_exists():
    assert Path("models/model.joblib").exists()


def test_model_prediction():
    model = joblib.load("models/model.joblib")
    X_test = np.load("data/features/X_test.npy")

    predictions = model.predict(X_test)

    assert len(predictions) == len(X_test)
    assert np.isfinite(predictions).all()


def test_metrics():
    metrics_path = Path("reports/metrics.json")
    assert metrics_path.exists()

    with open(metrics_path, encoding="utf-8") as file:
        metrics = json.load(file)

    assert "MAE" in metrics
    assert "RMSE" in metrics
    assert "R2" in metrics

    assert metrics["MAE"] >= 0
    assert metrics["RMSE"] >= 0
