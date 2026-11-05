from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split


def main():
    with open("params.yaml", encoding="utf-8") as file:
        params = yaml.safe_load(file)["prepare"]

    input_path = Path("data/raw/student-mat.csv")
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path, sep=";")

    if "G3" not in df.columns:
        raise ValueError("Target column G3 is missing")

    if df.empty:
        raise ValueError("Dataset is empty")

    train_df, test_df = train_test_split(
        df,
        test_size=params["test_size"],
        random_state=params["random_state"],
    )

    train_df.to_csv(output_dir / "train.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)

    print(f"Dataset shape: {df.shape}")
    print(f"Training rows: {len(train_df)}")
    print(f"Testing rows: {len(test_df)}")


if __name__ == "__main__":
    main()
