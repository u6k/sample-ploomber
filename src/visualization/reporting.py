import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

# %% tags=["parameters"]
upstream = ["train_model"]


# %%
df_pred_test = pd.read_parquet(upstream["train_model"]["df_pred_test"])
df_pred_test


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
    columns=[f"pred_{l}" for l in matrix_labels],
    index=[f"act_{l}" for l in matrix_labels]
)

df_con_matrix


# %%
with open(product["classification_report"], "w") as f:
    f.write(cls_report)

with open(product["confusion_matrix"], "w") as f:
    f.write(df_con_matrix.to_string())
