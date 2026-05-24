
from datasets import load_dataset
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)

from xgboost import XGBClassifier

dataset = load_dataset(
    "darpan1703/gl_tourism_project_dataset"
)

df = dataset["train"].to_pandas()

df = df.drop(
    columns=["Unnamed: 0", "CustomerID"],
    errors="ignore"
)

X = df.drop(columns=["ProdTaken"])
y = df["ProdTaken"]

categorical_cols = X.select_dtypes(
    include=["object"]
).columns.tolist()

numerical_cols = X.select_dtypes(
    include=["int64","float64"]
).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            numerical_cols
        ),
        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_cols
        )
    ]
)

pipeline = Pipeline([
    (
        "preprocessor",
        preprocessor
    ),
    (
        "classifier",
        XGBClassifier(
            eval_metric="logloss"
        )
    )
])

pipeline.fit(X,y)

joblib.dump(
    pipeline,
    "models/best_model.pkl"
)

print("Training Completed")
