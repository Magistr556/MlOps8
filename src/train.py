from pathlib import Path

import joblib
import numpy as np
import yaml
from sklearn.ensemble import RandomForestRegressor


def main():
    with open("params.yaml", encoding="utf-8") as file:
        params = yaml.safe_load(file)["train"]

    X_train = np.load("data/features/X_train.npy")
    y_train = np.load("data/features/y_train.npy")

    model = RandomForestRegressor(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        min_samples_split=params["min_samples_split"],
        random_state=params["random_state"],
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    output_dir = Path("models")
    output_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, output_dir / "model.joblib")

    print("Model training completed!")
    print("Training samples:", len(y_train))
    print("Number of features:", X_train.shape[1])


if __name__ == "__main__":
    main()
