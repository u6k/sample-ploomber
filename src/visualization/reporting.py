import mlflow
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)

# %% tags=["parameters"]
upstream = ["train_model"]
product = None


# %%
df_pred_test = pd.read_parquet(upstream["train_model"]["df_pred_test"])
df_pred_test


# %%
model_score = {
    "accuracy": accuracy_score(df_pred_test["target"], df_pred_test["pred"]),
    "precision": precision_score(df_pred_test["target"], df_pred_test["pred"], average="macro"),
    "recall": recall_score(df_pred_test["target"], df_pred_test["pred"], average="macro"),
    "f1": f1_score(df_pred_test["target"], df_pred_test["pred"], average="macro"),
}

model_score


# %%
cls_report = classification_report(df_pred_test["target"], df_pred_test["pred"])
print(cls_report)


# %%
matrix_labels = [0, 1, 2]

con_matrix = confusion_matrix(
    df_pred_test["target"],
    df_pred_test["pred"],
    labels=matrix_labels
)

df_con_matrix = pd.DataFrame(
    con_matrix,
    columns=[f"pred_{label}" for label in matrix_labels],
    index=[f"act_{label}" for label in matrix_labels]
)

df_con_matrix


# %%
with open(product["classification_report"], "w") as f:
    f.write(cls_report)

with open(product["confusion_matrix"], "w") as f:
    f.write(df_con_matrix.to_string())


# %%
mlflow.set_experiment("sample-ploomber")
with mlflow.start_run(run_name="reporting"):
    mlflow.log_metrics(model_score)
    mlflow.log_artifact(product["classification_report"])
    mlflow.log_artifact(product["confusion_matrix"])
