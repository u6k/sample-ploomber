import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

# %% tags=["parameters"]
upstream = {
    "train_model": {
        "pred_test": "../../data/processed/df_pred_test.parquet",
    }
}
product = None


# %%
df_pred_test = pd.read_parquet(upstream["train_model"]["pred_test"])
df_pred_test


# %%
print(classification_report(df_pred_test["target"], df_pred_test["pred"]))


# %%
confusion_matrix(df_pred_test["target"], df_pred_test["pred"])
