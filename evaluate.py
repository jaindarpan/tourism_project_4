
from datasets import load_dataset
import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score
)

dataset = load_dataset(
    "darpan1703/gl_tourism_project_dataset"
)

df = dataset["test"].to_pandas()

df = df.drop(
    columns=["Unnamed: 0", "CustomerID"],
    errors="ignore"
)

X_test = df.drop(columns=["ProdTaken"])
y_test = df["ProdTaken"]

model = joblib.load(
    "models/best_model.pkl"
)

pred = model.predict(X_test)

prob = model.predict_proba(X_test)[:,1]

print(
    "Accuracy:",
    accuracy_score(y_test,pred)
)

print(
    "ROC AUC:",
    roc_auc_score(y_test,prob)
)
