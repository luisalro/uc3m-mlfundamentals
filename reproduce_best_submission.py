"""Reproduce the best Kaggle submission (predictions_v9.csv).

Final approach extracted from the project notebook:
- Backward Sequential Feature Selection (8 numeric features)
- KNN imputation + standardization for numeric variables
- Most-frequent imputation + one-hot encoding for categorical variables
- Tuned ExtraTreesRegressor
- Average predictions across 10 random seeds
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_FEATURES = [
    "feat_1", "feat_6", "feat_7", "feat_8",
    "feat_9", "feat_11", "feat_14", "feat_15",
]
CATEGORICAL_FEATURES = ["feat_16", "feat_18", "feat_19"]
SEEDS = [42, 0, 7, 13, 99, 123, 256, 512, 777, 999]

BEST_PARAMS = {
    "n_estimators": 1000,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "max_features": 0.6,
    "max_depth": 40,
    "ccp_alpha": 0.0001,
    "bootstrap": False,
}


def build_pipeline(seed: int) -> Pipeline:
    numeric = Pipeline([
        ("imputer", KNNImputer(n_neighbors=5)),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric, NUMERIC_FEATURES),
        ("cat", categorical, CATEGORICAL_FEATURES),
    ])
    model = ExtraTreesRegressor(
        random_state=seed,
        n_jobs=-1,
        **BEST_PARAMS,
    )
    return Pipeline([("pre", preprocessor), ("reg", model)])


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=root / "data" / "train_set.csv")
    parser.add_argument("--test", type=Path, default=root / "data" / "test_set.csv")
    parser.add_argument("--output", type=Path, default=root / "submissions" / "predictions_reproduced.csv")
    args = parser.parse_args()

    train = pd.read_csv(args.train)
    test = pd.read_csv(args.test)
    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X_train = train[features]
    y_train = train["target"]
    X_test = test[features]

    predictions = []
    for seed in SEEDS:
        pipeline = build_pipeline(seed)
        pipeline.fit(X_train, y_train)
        predictions.append(pipeline.predict(X_test))

    averaged = np.mean(predictions, axis=0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"ID": test["ID"], "target": averaged}).to_csv(args.output, index=False)
    print(f"Saved {len(test):,} predictions to {args.output}")


if __name__ == "__main__":
    main()
