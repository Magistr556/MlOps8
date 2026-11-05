import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def main():
    X_test = np.load("data/features/X_test.npy")
    y_test = np.load("data/features/y_test.npy")

    model = joblib.load("models/model.joblib")
    predictions = model.predict(X_test)

    metrics = {
        "MAE": float(mean_absolute_error(y_test, predictions)),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, predictions))),
        "R2": float(r2_score(y_test, predictions)),
    }

    output_path = Path("reports/metrics.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    print("Evaluation completed!")
    print(json.dumps(metrics, indent=4))


if __name__ == "__main__":
    main()
