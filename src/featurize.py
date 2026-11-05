from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def main():
    with open("params.yaml", encoding="utf-8") as file:
        params = yaml.safe_load(file)["featurize"]

    train_df = pd.read_csv("data/processed/train.csv")
    test_df = pd.read_csv("data/processed/test.csv")

    X_train = train_df.drop(columns=["G3"])
    y_train = train_df["G3"]

    X_test = test_df.drop(columns=["G3"])
    y_test = test_df["G3"]

    numeric_features = X_train.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_features = X_train.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    numeric_transformer = StandardScaler() if params["scale_numeric"] else "passthrough"

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, numeric_features),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_features,
            ),
        ]
    )

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    output_dir = Path("data/features")
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "X_train.npy", X_train_processed)
    np.save(output_dir / "X_test.npy", X_test_processed)
    np.save(output_dir / "y_train.npy", y_train.to_numpy())
    np.save(output_dir / "y_test.npy", y_test.to_numpy())

    joblib.dump(preprocessor, output_dir / "preprocessor.joblib")

    print("Numeric features:", len(numeric_features))
    print("Categorical features:", len(categorical_features))
    print("Processed training shape:", X_train_processed.shape)
    print("Processed testing shape:", X_test_processed.shape)


if __name__ == "__main__":
    main()
